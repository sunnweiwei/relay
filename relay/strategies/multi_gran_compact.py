"""Multi-granularity prompt compaction for append-only Responses loops.

A relay port of FoldAgent's ``pg_compaction`` ("prompt-based, single-granularity,
prompt-controlled" context compaction). There is NO level selector and NO discrete
granularities: compaction fires on a fixed **token cadence** (default every 32k
tokens of raw working context) and a single carefully written prompt tells a
dedicated compactor *how coarse vs. how fine* to compress each span
("multi-granularity" in the continuous, prompt-decided sense — the compactor picks
one of TRAJECTORY / TRANSITION / STATE styles per span).

How it maps onto relay's stateless, append-only strategy model:

* The first system/developer items are a protected prefix — never compacted, never
  counted toward the cadence (:func:`_protected_prefix`).
* Each request carries the full trajectory. Walking from the last sealed note, raw
  interactions accumulate (counted with the compactor's own Qwen tokenizer) until a
  tool-safe boundary (:func:`_safe_boundaries`) first crosses ``compact_threshold``;
  that whole span is folded into ONE memory note and replaced. Previous notes stay
  verbatim and are never re-summarized — notes simply accumulate:
  ``[prefix, note_1, ..., note_k, raw_tail]``. This replays, deterministically, the
  same fold sequence the online agent would have produced (it folds at the START of a
  step, before generating, so the model always acts on a bounded context).
* Notes are stored in the checkpoint artifact, so :meth:`materialize` reconstructs the
  active context without any LLM call and relay's exact-prefix cache reuses folds
  across requests. Only spans BEYOND the recovered artifact are (re)folded.
* ``max_compaction`` caps the number of folds; the final allowed note gets a
  commit-now ``FOOTER`` so the working agent stops exploring and finishes before the
  window overflows.

Unlike the other relay manager strategies (which reuse the task upstream via
``responses._client`` and only override the model NAME), the compactor here is a
genuinely separate endpoint speaking Chat Completions — configured via
``RELAY_MULTI_GRAN_BASE_URL`` / ``RELAY_MULTI_GRAN_API_KEY`` / ``RELAY_MULTI_GRAN_MODEL``
(mirroring FoldAgent's ``--compact_base_url`` / ``--compact_model_name``). When
``RELAY_MULTI_GRAN_BASE_URL`` is unset it falls back to inheriting the task upstream
(rlm-style), so the model-name-only override still works.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .base import BaseStrategy, GeneratedCheckpoint, PreparedInput
from .compact import _safe_boundaries

_STATE_VERSION = 1
_ARTIFACT_KIND = "multi_gran_compact"

# ---------------------------------------------------------------------------
# Prompts (ported verbatim from FoldAgent agents/pg_compaction.py)
# ---------------------------------------------------------------------------
COMPACT_SYSTEM_PROMPT = (
    "You are the MEMORY of a deep-research agent working on a hard, multi-hop "
    "BrowseComp-Plus question — the answer is a specific entity/fact that has to be "
    "pinned down by chaining evidence from retrieved documents (each cited by a "
    "bracketed docid like [20]). The agent works by repeatedly issuing `search` / "
    "`open_page` actions and reading the returned documents. Its context has grown "
    "too long, so your job is to REWRITE one old span of its interactions into a "
    "compact note that the agent will read back as its OWN memory of that work.\n\n"
    "--- Voice: write as the agent's own recollection ---\n"
    "Write the note in the second person, narrating what the agent itself did and "
    "saw. It must read as the agent's own memory, not as a "
    "third-party summary.\n\n"
    "You are NOT the research agent. Do NOT answer the question, name a final answer"
    ", decide the next search, or draw any "
    "conclusion the documents don't literally state. You only condense what already "
    "happened. Earlier turns in this conversation are notes you already wrote for "
    "previous spans — continue in the same voice and do not repeat their content.\n\n"
    "--- Pick one of three compaction styles for THIS span ---\n"
    "You compact in one of three styles. Each time you are called, pick the ONE style "
    "that fits the span you are folding now and follow the example to compact the span.\n\n"
    "1. TRAJECTORY — replay it, step by step. Keep the SHAPE of what happened: the same "
    "ordered sequence of actions and observations, each one condensed, one entry per "
    "step, in order. For each `search`, its query then a thinned list of the returned "
    "documents (title, a one-sentence snippet, docid — including the ones that look "
    "unrelated); for each `open_page`, the specific sentences carrying names, numbers, "
    "dates, quotes and claims, each with its docid; keep every candidate entity, partial "
    "lead and unconfirmed clue, flagging any seen only in a snippet and not yet opened.\n"
    "   Example: You opened document [66261] to verify your top candidate Francesca Zappia and "
    "confirmed she lives in Indiana, studies Computer Science at the University of "
    "Indianapolis, and plays 'way too much Pokémon'; you also confirmed that though "
    "a bio said she started writing at eight, she clarified she began the story for "
    "Made You Up at ten or eleven, and that its plot has Alex dealing with paranoid "
    "schizophrenia. You then opened [14211], an unrelated interview about PaRappa "
    "the Rapper, and [10683] on novelist Gabrielle Zevin, confirming her lifelong "
    "video-game interest but nothing about writing before age ten or a boss "
    "romance. You then ran five searches trying to tie the 'started at eight' "
    "detail to a workplace-romance trope; all returned irrelevant lists, so no "
    "single document yet matches every criterion.\n"
    "2. TRANSITION — compress the whole span into ONE high-level action and observation. "
    "Do NOT replay step by step and do NOT enumerate the individual searches and pages. "
    "Lift the span up one level of abstraction, the way you would recall a finished "
    "sub-investigation: the ACTION is the through-line of the whole span — the one thing "
    "you were really trying to do across all those steps (the clue you set out to "
    "establish, the candidate you set out to verify), stated as a single intent, not as "
    "the queries you typed; the OBSERVATION is what that effort established — the concrete "
    "findings it produced, each with its exact docid, stating each fact once. Even if the "
    "span ranged over several threads, find the single line of inquiry that ties them "
    "together and write ONE action-observation for it all.\n"
    "   Example: You set out to verify whether the candidate landmark — the 1798 Monument "
    "(Pikeman statue) at Wexford's Bull Ring — satisfies the surrounding-business clues in the question. This established that the "
    "required nearby businesses sit at the Bull Ring — Ryans Opticians [14181], Stone "
    "Solicitors [24818], Meylers Fish Company [5636] and the former Red Square Pizza "
    "(now closed) [36301] — while the Asian street-food businesses Aroi and Mi were "
    "reported in Wexford town centre around Anne Street [81865].\n"
    "3. STATE — one sentence of progress. Do NOT describe what you did, what you saw, or "
    "list evidence. In a SINGLE sentence, state where the investigation now stands: what "
    "has been pinned down so far and what clues have been verified.\n"
    "   Example: You have pinned the landmark down as the 1798 Monument (Pikeman statue) at Wexford's "
    "Bull Ring [34106] — unveiled in 1905 and restored in 2009, on a site that once "
    "served as an open-air pike factory — and you have confirmed the surrounding businesses"
    ": the optician Ryans [14181], the Stone Solicitors office [24818], the Meylers fish "
    "company [5636], and the Asian fusion restaurant Aroi [81865].\n"
    "--- Strict rules ---\n"
    "- Record ONLY what the documents in this span literally state and what the agent "
    "literally did. Never infer, chain clues together, or conclude beyond the span. "
    "- Do NOT answer the question or name a final answer — not even a tentative, "
    "partial, or best-guess one.\n"
    "- Preserve concrete facts, numbers, names and dates verbatim. Invent nothing.\n\n"
    "- Never say something is unverified. Be positive about the search results. Your only job is to record what happened, not to comment on whether the steps returned useful information."
    "--- Output ---\n"
    "Do not reason first, output the note directly.\n\n"
    "The research question:\n{question}"
)

COMPACT_USER_TEMPLATE = (
    "# New span of my raw interactions to compact\n{span}\n\n"
    "# Instruction\n"
    "Follow the guidance and pick one of three compaction styles to compact:"
    "Use TRAJECTORY when: the span's exploration is still IN PROGRESS — it has not yet "
    "verified a clue that research question gives. You are still searching and do not yet "
    "know which step will have important information. So, replay the span step by step with exact documents docid\n"
    "Use TRANSITION when: the span's exploration REACHED its goal — it verified a clue provided by the research question. The purpose is met, so "
    "the individual steps no longer matter — only the net outcome: a handful of concrete "
    "verification of clues, each with its exact docid. \n"
    "Use STATE when: the span CLOSES OUT a whole phase and moves the investigation into a new "
    "stage — several clues are verified an accumulated, and this span's "
    "role is to consolidate them into a new position rather than to produce new evidence.\n"
    "Output the note text directly:"
)

# Prepended to every folded note when it is injected as a user turn: a trust
# directive telling the working agent this is a faithful record of its OWN prior work.
MEMORY_HEADER = (
    "[Your own memory — trustworthy] The note below is a faithful, compacted "
    "record of research YOU already did: the searches you ran, the documents you "
    "opened, and the concrete facts you found (with their exact docids). Treat the "
    "facts it records as established — don't re-run a search just to re-confirm "
    "something the note already settled. Build your work FORWARD from it, and rely "
    "on it when deciding your final answer. When the note flags a clue as seen only "
    "in a search snippet (not yet opened) or still unconfirmed, move it forward by "
    "opening that document with open_page — not by issuing more searches.\n\n"
)

# Appended to the note of the FINAL allowed compaction: a hard nudge to stop
# exploring and commit the current best answer before the window overflows.
FOOTER = (
    "\n\n[**STRICT RULE**: Context budget nearly exhausted, finish now] Your context window is almost full. Call `finish` function with your single most likely answer even if some clues are still unconfirmed."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class MultiGranCompact(BaseStrategy):
    """Cadence-triggered sequential note compaction with a separate compactor."""

    compact_threshold: int = 30_000
    max_compaction: int = 10
    compact_model: str | None = None
    compact_base_url: str | None = None
    compact_api_key: str | None = None
    reasoning_effort: str | None = None
    tokenizer_name: str = "Qwen/Qwen3.5-9B"
    max_note_tokens: int = 16_384
    verbose: bool = False
    name: str = field(default="multi_gran_compact", init=False)

    def __post_init__(self) -> None:
        if self.compact_threshold <= 0:
            raise ValueError("compact_threshold must be positive")
        if self.max_compaction <= 0:
            raise ValueError("max_compaction must be positive")
        # Lazily built once and reused across requests (the strategy instance is
        # shared by the engine); rebuilding the conv per call keeps the vLLM prefix
        # cache warm without any persistent client state.
        self._client_cache: Any = None

    @classmethod
    def from_env(cls) -> MultiGranCompact:
        return cls(
            compact_threshold=int(os.getenv("RELAY_MULTI_GRAN_THRESHOLD", "30000")),
            max_compaction=int(os.getenv("RELAY_MULTI_GRAN_MAX_COMPACTION", "10")),
            compact_model=os.getenv("RELAY_MULTI_GRAN_MODEL") or None,
            compact_base_url=os.getenv("RELAY_MULTI_GRAN_BASE_URL") or None,
            compact_api_key=os.getenv("RELAY_MULTI_GRAN_API_KEY") or None,
            reasoning_effort=os.getenv("RELAY_MULTI_GRAN_REASONING_EFFORT") or None,
            tokenizer_name=os.getenv(
                "RELAY_MULTI_GRAN_TOKENIZER", "Qwen/Qwen3.5-9B"
            ),
            max_note_tokens=int(os.getenv("RELAY_MULTI_GRAN_MAX_NOTE_TOKENS", "16384")),
            verbose=_env_bool("RELAY_MULTI_GRAN_VERBOSE", False),
        )

    def cache_scope(self) -> dict[str, Any]:
        return {
            "compact_threshold": self.compact_threshold,
            "max_compaction": self.max_compaction,
            "compact_model": self.compact_model,
            "compact_base_url": self.compact_base_url,
            "tokenizer_name": self.tokenizer_name,
            "prompt_version": 1,
        }

    # ------------------------------------------------------------- materialize --
    def materialize(
        self,
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> list[dict[str, Any]]:
        if checkpoint is None:
            return deepcopy(trajectory)
        protected_len = _leading_prefix(trajectory)
        protected = deepcopy(trajectory[:protected_len])
        notes, _ = self._notes(checkpoint, trajectory, protected_len)
        last_end = notes[-1]["end"] if notes else len(protected)
        return [
            *protected,
            *self._note_items(notes),
            *deepcopy(trajectory[last_end:]),
        ]

    # ----------------------------------------------------------------- prepare --
    def prepare(
        self,
        responses: Any,
        request: dict[str, Any],
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> PreparedInput:
        protected_len = _leading_prefix(trajectory)
        protected = deepcopy(trajectory[:protected_len])
        if checkpoint is None:
            notes: list[dict[str, Any]] = []
        else:
            notes, _ = self._notes(checkpoint, trajectory, protected_len)

        generated = self._fold(
            responses, request, trajectory, notes, protected_len, force_tail=False
        )
        last_end = notes[-1]["end"] if notes else protected_len
        active = [
            *protected,
            *self._note_items(notes),
            *deepcopy(trajectory[last_end:]),
        ]
        changed = bool(generated)
        current = (
            GeneratedCheckpoint(
                covered_items=len(trajectory),
                artifact=self._artifact(notes),
            )
            if changed
            else None
        )
        return PreparedInput(
            active,
            compacted=active != trajectory,
            checkpoints=tuple(generated),
            checkpoint=current,
        )

    # ----------------------------------------------------------------- compact --
    def compact(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        protected_len = _leading_prefix(active)
        protected = deepcopy(active[:protected_len])
        notes: list[dict[str, Any]] = []
        self._fold(
            responses, request, active, notes, protected_len, force_tail=True
        )
        last_end = notes[-1]["end"] if notes else protected_len
        return [
            *protected,
            *self._note_items(notes),
            *deepcopy(active[last_end:]),
        ]

    # -------------------------------------------------------------- fold engine --
    def _fold(
        self,
        responses: Any,
        request: dict[str, Any],
        seq: list[dict[str, Any]],
        notes: list[dict[str, Any]],
        protected_len: int,
        *,
        force_tail: bool,
    ) -> list[GeneratedCheckpoint]:
        """Fold cadence-sized spans into notes, appending to ``notes`` in place.

        Replays the online fold sequence: from the last sealed note, cut at the first
        tool-safe boundary whose accumulated tokens reach ``compact_threshold`` and
        fold that span into one note. ``force_tail`` (used by :meth:`compact`) also
        folds a below-threshold remainder into a final note. Returns the newly
        generated checkpoints (one per fold) for the prefix cache.
        """
        total = len(seq)
        # One tokenizer pass over the sequence -> O(1) prefix-sum span counts.
        item_tokens = [self._count(_item_text(item)) for item in seq]
        cumulative = [0]
        for count in item_tokens:
            cumulative.append(cumulative[-1] + count)

        def span_tokens(start: int, end: int) -> int:
            return cumulative[end] - cumulative[start]

        boundaries = [b for b in _safe_boundaries(seq) if b > protected_len]
        last_end = notes[-1]["end"] if notes else protected_len
        generated: list[GeneratedCheckpoint] = []

        while len(notes) < self.max_compaction and last_end < total:
            over = span_tokens(last_end, total) >= self.compact_threshold
            if not over and not force_tail:
                break
            cut = next(
                (
                    b
                    for b in boundaries
                    if b > last_end and span_tokens(last_end, b) >= self.compact_threshold
                ),
                None,
            )
            if cut is None:
                if not force_tail:
                    break
                # No boundary reaches the threshold: fold the whole remaining tail
                # up to the last safe boundary (only if the tail is tool-complete).
                tail_cut = next(
                    (b for b in reversed(boundaries) if b > last_end), None
                )
                if tail_cut is None:
                    break
                cut = tail_cut
            span = seq[last_end:cut]
            if not span:
                break
            note_text = self._compact_span(responses, request, seq, notes, span)
            if not note_text:
                # Compaction failed: leave the span raw and stop (retry next request).
                break
            notes.append({"end": cut, "text": note_text})
            generated.append(
                GeneratedCheckpoint(
                    covered_items=cut,
                    artifact=self._artifact(notes),
                )
            )
            if self.verbose:
                print(
                    f"[MultiGranCompact] fold {len(notes)} span={len(span)} items "
                    f"~{span_tokens(last_end, cut)} tok -> note ~{self._count(note_text)} tok"
                )
            last_end = cut

        return generated

    # ------------------------------------------------------------- compactor io --
    def _compact_span(
        self,
        responses: Any,
        request: dict[str, Any],
        seq: Sequence[dict[str, Any]],
        prior_notes: Sequence[dict[str, Any]],
        span: Sequence[dict[str, Any]],
    ) -> str:
        """Render ``span`` and ask the compactor for one memory note.

        The compactor conversation ``[system(+question), note_1, ..., note_k]`` is
        rebuilt from ``prior_notes`` so it is byte-stable across calls (vLLM prefix
        cache) while prior notes double as the "already-known context".
        """
        question = _question(seq)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": COMPACT_SYSTEM_PROMPT.format(question=question)}
        ]
        for note in prior_notes:
            messages.append({"role": "assistant", "content": note["text"]})
        messages.append(
            {"role": "user", "content": COMPACT_USER_TEMPLATE.format(span=_render_span(span))}
        )
        raw = self._chat(responses, request, messages).strip()
        text = _THINK_RE.sub("", raw)
        return text.replace("<note>", "").replace("</note>", "").strip()

    def _chat(
        self, responses: Any, request: dict[str, Any], messages: list[dict[str, Any]]
    ) -> str:
        model = self.compact_model or request.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("a model is required for multi-gran compaction")
        client = self._client(responses)
        try:
            if model.lower().startswith("gpt-5"):
                return self._chat_responses(client, model, messages)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.max_note_tokens,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            if self.verbose:
                print(f"[MultiGranCompact] compactor call failed: {exc}")
            return ""

    def _chat_responses(
        self, client: Any, model: str, messages: list[dict[str, Any]]
    ) -> str:
        # Responses-API variant for gpt-5.x: the system prompt rides in `instructions`
        # while `input` keeps a placeholder empty system turn (some endpoints return an
        # empty output otherwise), followed by the non-system turns.
        instructions = "\n\n".join(
            m.get("content", "") or "" for m in messages if m.get("role") == "system"
        )
        conv = [
            {"role": m["role"], "content": m.get("content", "") or ""}
            for m in messages
            if m.get("role") != "system"
        ]
        create_kwargs: dict[str, Any] = {}
        if self.reasoning_effort:
            create_kwargs["reasoning"] = {"effort": self.reasoning_effort}
        resp = client.responses.create(
            model=model,
            instructions=instructions or None,
            input=[{"role": "system", "content": "You are a helpful assistant."}] + conv,
            temperature=0.7,
            top_p=0.8,
            **create_kwargs,
        )
        return getattr(resp, "output_text", None) or ""

    def _client(self, responses: Any) -> Any:
        if self._client_cache is not None:
            return self._client_cache
        from openai import OpenAI

        base_url = self.compact_base_url
        api_key = self.compact_api_key
        if base_url is None or api_key is None:
            # Fall back to the task upstream (rlm-style) for whichever is unset.
            inner = getattr(responses, "_client", None)
            if base_url is None and inner is not None:
                inherited = getattr(inner, "base_url", None)
                if inherited is not None:
                    base_url = str(inherited)
            if api_key is None:
                inherited_key = getattr(inner, "api_key", None) if inner else None
                api_key = inherited_key or os.getenv("OPENAI_API_KEY")
        self._client_cache = OpenAI(base_url=base_url, api_key=api_key)
        return self._client_cache

    # ---------------------------------------------------------------- artifacts --
    def _note_items(self, notes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        capped = len(notes) >= self.max_compaction
        items: list[dict[str, Any]] = []
        for index, note in enumerate(notes):
            content = f"{MEMORY_HEADER}{note['text']}"
            if capped and index == len(notes) - 1:
                content += FOOTER
            items.append({"type": "message", "role": "user", "content": content})
        return items

    def _artifact(self, notes: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "kind": _ARTIFACT_KIND,
            "compact_threshold": self.compact_threshold,
            "max_compaction": self.max_compaction,
            "notes": deepcopy(list(notes)),
        }

    def _notes(
        self,
        checkpoint: GeneratedCheckpoint,
        trajectory: Sequence[dict[str, Any]],
        protected_len: int,
    ) -> tuple[list[dict[str, Any]], int]:
        artifact = checkpoint.artifact
        if artifact.get("version") != _STATE_VERSION or artifact.get(
            "kind"
        ) != _ARTIFACT_KIND:
            raise ValueError("invalid multi-gran compact artifact")
        if (
            artifact.get("compact_threshold") != self.compact_threshold
            or artifact.get("max_compaction") != self.max_compaction
        ):
            raise ValueError("multi-gran compact thresholds changed during a trajectory")
        if checkpoint.covered_items > len(trajectory):
            raise ValueError("multi-gran compact checkpoint exceeds the trajectory")
        values = artifact.get("notes")
        if not isinstance(values, list):
            raise TypeError("invalid multi-gran compact notes")
        if len(values) > self.max_compaction:
            raise ValueError("multi-gran compact checkpoint exceeds max_compaction")

        notes: list[dict[str, Any]] = []
        previous_end = protected_len
        for value in values:
            if not isinstance(value, dict):
                raise TypeError("invalid multi-gran compact note")
            end = value.get("end")
            text = value.get("text")
            if (
                not isinstance(end, int)
                or not isinstance(text, str)
                or end <= previous_end
                or end > checkpoint.covered_items
            ):
                raise ValueError("invalid multi-gran compact note")
            notes.append({"end": end, "text": text})
            previous_end = end
        return notes, checkpoint.covered_items

    def _count(self, text: str) -> int:
        return _count_tokens(text, self.tokenizer_name)


# ---------------------------------------------------------------------------
# Rendering + token counting helpers (model-agnostic; Responses input items).
# ---------------------------------------------------------------------------
def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return "" if content is None else str(content)


def _item_text(item: dict[str, Any]) -> str:
    typ = item.get("type", "message")
    if typ == "reasoning":
        return ""
    if typ in {"function_call", "custom_tool_call"}:
        return f"{item.get('name', '')}({item.get('arguments', '') or item.get('input', '')})"
    if typ in {"function_call_output", "custom_tool_call_output"}:
        return _content_text(item.get("output", ""))
    return _content_text(item.get("content", ""))


def _render_span(items: Sequence[dict[str, Any]]) -> str:
    """Plain, model-agnostic [Action]/[Observation] transcript of a span.

    Reasoning items are encrypted/empty so they render nothing; assistant messages
    and tool calls are actions, user messages and tool outputs are observations.
    """
    parts: list[str] = []
    for item in items:
        typ = item.get("type", "message")
        if typ == "reasoning":
            continue
        if typ in {"function_call", "custom_tool_call"}:
            parts.append(f"[Action]\n{_item_text(item)}")
            continue
        if typ in {"function_call_output", "custom_tool_call_output"}:
            parts.append(f"[Observation]\n{_item_text(item)}")
            continue
        text = _item_text(item)
        if item.get("role") == "assistant":
            parts.append(f"[Action]\n{text}")
        else:
            parts.append(f"[Observation]\n{text}")
    return "\n\n".join(parts)


def _question(trajectory: Sequence[dict[str, Any]]) -> str:
    """The research question = the first user message (the task/problem statement)."""

    for item in trajectory:
        if item.get("type", "message") == "message" and item.get("role") == "user":
            text = _item_text(item)
            if text:
                return text
    return _render_span(list(trajectory)[: _leading_prefix(list(trajectory))])


def _leading_prefix(trajectory: Sequence[dict[str, Any]]) -> int:
    """Protected-prefix length: leading system/developer/user turns before the first model action.

    Mirrors FoldAgent's ``prompt_turn`` (system + the initial task/question): everything
    up to — but not including — the first assistant turn, reasoning item, or tool call is
    never compacted and never counted toward the fold cadence. Unlike relay's generic
    ``_protected_prefix`` (system/developer only), this keeps the literal task in context.
    """
    end = 0
    for item in trajectory:
        if item.get("type", "message") != "message":
            break
        if item.get("role") not in {"system", "developer", "user"}:
            break
        end += 1
    return end


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


@functools.lru_cache(maxsize=4)
def _get_tokenizer(name: str) -> Any:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def _count_tokens(text: str, name: str) -> int:
    """Exact token count under the compactor tokenizer, len//3 fallback offline."""

    if not text:
        return 1
    try:
        tokenizer = _get_tokenizer(name)
    except Exception as exc:  # pragma: no cover - only when the tokenizer is unavailable
        if not getattr(_count_tokens, "_warned", False):
            print(
                f"[MultiGranCompact] WARNING: could not load {name} ({exc}); "
                "falling back to len//3 token estimate"
            )
            _count_tokens._warned = True  # type: ignore[attr-defined]
        return _estimate_tokens(text)
    return max(1, len(tokenizer.encode(text, add_special_tokens=False)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")
