# Context-management evaluation on three benchmarks

This directory contains the code and results of an evaluation of Relay's context-management
strategies on three long-horizon agent benchmarks. Codex is the agent harness and Relay sits
between Codex and the model. Every request the model sees, including the strategies' own
summary and sub-model calls, must fit a fixed input window. The strategy is the only thing
that varies between runs of the same task.

The results are a pilot. Each cell (one task, one method) was run once, with one model
(`gpt-6-luna`, medium reasoning effort). Read the limitations below before you draw
conclusions from the tables.

## Results

`evals/results/pilot/context-length.html` is the full report. It shows each method's score overall and
by the task's context length, gives paired differences on the same tasks, and reports how
often each method hit the request or time cap. `evals/results/pilot/context-length.json` holds the same
numbers, and `evals/results/pilot/cells.csv` lists every run.

Scores use one outcome policy on all three benchmarks (`outcomes.py`):

- A run without a score because of the run itself counts as 0. Examples: the method raised
  an error, the provider refused the content the method sent, the model's output was cut at
  `max_output_tokens`, or the verifier timed out on the submitted artifact.
- A run without a score because of the service around it is excluded and counted
  separately. Examples: the upstream dropped the connection, rate limits outlasted the
  retries, the quota guard stopped the run, or the input counter could not account for a
  request.

Method comparisons use the tasks on which all five core methods (Compaction, Folding, CLM,
ACM, Self-Compact) have a score under this policy.

Two methods were added on 2026-11-06 (see Protocol): a truncation-only baseline and ACM with
a forced compression. They ran a week after the other methods, and the account's weekly
usage limit stopped them part of the way: on 404 and 403 of the 825 Browse questions, the
first in the evaluation's fixed random order, and on 23 Science tasks. Their rows below use
the tasks they have, and their contrasts use the tasks on which both methods have a score.

### BrowseComp-Plus: 825 questions

The metric is strict accuracy. The interval is a 95% Wilson interval. All 825 questions have a
score for every core method.

| Method | Accuracy (95% CI) | Runs scored | Task failures (0) | Excluded | Answer contains gold (lenient) | Runs with truncation fallback |
| --- | --- | --- | --- | --- | --- | --- |
| Compaction | 49.7% (46.3–53.1%) | 825 | 5 | 0 | 54.1% | 0% |
| Folding | 40.2% (36.9–43.6%) | 825 | 21 | 0 | 46.5% | 1% |
| CLM | 39.5% (36.2–42.9%) | 825 | 2 | 0 | 48.1% | 0% |
| ACM | 38.2% (34.9–41.5%) | 825 | 2 | 0 | 45.6% | 92% |
| Self-Compact | 16.8% (14.4–19.6%) | 825 | 10 | 0 | 19.4% | 0% |
| ACM (forced), added | 36.7% (32.0–41.6%) | 390 | 2 | 13 | 44.8% | 6% |
| Truncation only, added | 32.2% (27.8–37.0%) | 394 | 2 | 10 | 36.0% | 85% |
| RLM (reference) | 0.0% | 708 | 4 | 117 | 0.0% | 0% |

These are paired differences between core methods on the 825 questions (95% bootstrap
interval):

- Compaction minus Folding: +9.5% (+6.4 to +12.6).
- Compaction minus CLM: +10.2% (+6.9 to +13.5).
- Compaction minus ACM: +11.5% (+8.1 to +14.9).

These are differences from the truncation-only baseline, on the 394 questions it ran (389 for
ACM (forced)):

| Method minus Truncation only | Difference (95% CI) |
| --- | --- |
| Compaction | +20.1% (+15.2 to +25.1) |
| Folding | +9.1% (+4.1 to +14.0) |
| CLM | +7.1% (+2.5 to +11.7) |
| ACM | +6.1% (+1.3 to +10.9) |
| ACM (forced) | +4.6% (+0.0 to +9.5) |
| Self-Compact | −13.2% (−18.3 to −7.9) |

ACM (forced) minus ACM is −1.5% (−6.7 to +3.6) on 390 questions: forcing the compression did
not help ACM. Truncation-only runs ended early (median 9 main requests, against 16 for ACM
and 29 for Compaction).

### ARC-AGI-3: 25 public games × 6 methods

The metric is the share of a game's levels completed. No run completed a whole game, and on
14 of the 19 common games no method completed a level. This benchmark is at its floor here
and does not separate the methods. The added methods did not run on it.

### Terminal-Bench-Science: 53 tasks with all core methods scored

The metric is the share of the task's verifier tests that pass. It is a diagnostic partial
score: the official reward is 1 only when every test passes, and only two runs reached
reward 1 (ACM and Compaction, one each). The metric has a floor, since RLM runs that barely
worked on the task still pass about 16% of the tests.

| Method | Tests passed (95% CI) | Runs scored | Runs with truncation fallback |
| --- | --- | --- | --- |
| ACM | 38.1% (29.6–46.8%) | 57 | 65% |
| Compaction | 33.0% (24.4–41.0%) | 54 | 0% |
| Folding | 31.9% (23.2–40.7%) | 55 | 0% |
| CLM | 27.7% (18.6–37.0%) | 55 | 0% |
| Self-Compact | 18.9% (11.4–27.1%) | 56 | 0% |
| ACM (forced), added | 43.7% (28.7–59.2%) | 17 | 9% |
| Truncation only, added | 42.4% (28.8–56.0%) | 18 | 61% |
| RLM (reference) | 15.8% (8.0–24.3%) | 48 | 0% |

The paired interval of ACM minus Compaction, from −1.8% to +14.0%, includes 0. The added
methods ran on too few Science tasks to rank them: on the 16 tasks where both have a score,
ACM minus Truncation only is +0.7% (−8.7 to +9.4), and Compaction minus Truncation only is
−6.1% (−15.1 to +2.4).

### Earlier sampled comparison

`evals/results/pilot/sample-2026-10-26/` holds an earlier comparison of `gpt-6-luna` and `gpt-6.1-sol`
over all 11 methods on five sampled tasks per benchmark. It was stopped early: 80 of the 330
planned runs have a score. It also used an earlier protocol (no overflow fallback, smaller
caps). It is listed for completeness and does not support a ranking.

## Limitations

- **ACM depends on the truncation fallback.** All runs enabled Relay's common overflow
  fallback. When a method's own forced recovery still leaves a request over the window,
  Relay drops the oldest tool call groups. Released ACM has no forced recovery, and in 90%
  of its Browse runs the model never asked it to compress, so the fallback managed the
  context in most ACM runs. On Browse, ACM still beat the truncation-only baseline by 6.1%,
  so its guidance and occasional compression add something over the fallback alone.
- **The added methods are partial and a week later.** The truncation-only baseline and ACM
  (forced) ran on 2026-11-06, about half of Browse and a third of the ready Science tasks,
  until the account's weekly usage limit stopped the queues. Their questions are the first
  in a fixed random order, so they are a random subset, but their comparison with methods run
  a week earlier can include drift in the model or its serving. A same-day rerun of released
  ACM, meant to measure that drift, was queued next but the account limit rejected it before
  any run, so the drift is not measured.
- **RLM does not measure the RLM method.** Relay replaces the conversation with the RLM's
  handoff. In many runs the main agent never sees the task, and its final answer asks for
  the question. RLM is shown for reference and is left out of the paired comparisons.
- **The Browse score is strict.** It is an exact match of the normalized final answer, not
  the official LLM judge. The lenient column counts answers that contain the gold answer's
  words in order. That check catches correct answers with extra words, but it can also
  credit answers that list several candidates. The lenient column covers scored runs only.
  Compared with strict accuracy on the same runs, the gap differs by method, from 2.3% for
  Self-Compact to 8.5% for CLM. The strict ranking can therefore overstate some differences.
- **27 Browse runs were repeated because the prompt profile was not applied.** Browse runs
  replace Codex's generic coding instructions with a short Browse prompt
  (`relay/harnesses/browse_profile.py`). In 27 runs (4 or 5 per method, on six questions) Codex
  sent a different version of its prompt, and the profile left it unchanged. These cells,
  and the 29 Browse cells first excluded as infrastructure failures, were run again on
  2026-11-06 with a profile that refuses unchecked versions. The reruns replace the earlier
  runs (`cells.csv`, `source = reruns-20261106`), so every core method now has all 825
  questions.
- **Each benchmark used one fixed condition, and runs under other conditions are excluded.**
  - Browse excluded five questions (183, 514, 1026, 692, 1099) that ran under earlier
    conditions.
  - ARC excluded 17 cells that ran with 100 requests and 1,800 s.
  - Science excluded 29 cells that ran with 64 requests and 1,800 s.
- **Not every Science cell ran.** The pinned Science population has 70 tasks. 54 cells on
  nine tasks were never run because their environments could not be prepared on the
  evaluation cluster. They are not zeros.
- **The Science sandbox shares the network.** The task sandbox in `science_container.py`
  (bubblewrap) shares the host network. Some agents downloaded papers or installed packages.
  No run was seen fetching a reference solution, but the network was not blocked.
- **The Science code here is a consolidated version.** Science cells ran in task-specific
  runtime snapshots, because several tasks needed their own isolation adapters
  (`science_*_isolation.py`). The code here is the consolidated version of those adapters,
  and some task adapters differ in detail from the snapshot that ran.
- **The runs used Relay as of late October.** Every run used a snapshot of Relay from commit
  `c6bd44f` with the evaluation's changes. Upstream has since changed ACM (its signal
  handling, and it now compresses the user's later messages and carries images) and Folding,
  so the repository's ACM, ACM (forced) and Folding differ in detail from the code that ran.
  The runs of 2026-11-06 used the same snapshot as the earlier runs, with ACM (forced) built on
  that snapshot's ACM.
- **The ARC prompt had a wrong example.** Its coordinate example used `ACTION2`; only
  `ACTION6` takes coordinates. `arc_sample.py` keeps the corrected text, but the full ARC
  runs used the original prompt. The ARC service also has no reset action, so `GAME_OVER`
  ends a run.
- **ACE never builds a playbook here.** Each run starts with an empty ACE playbook, and a
  single `codex exec` turn never updates it. ACE in this setup is compaction plus an empty
  playbook prompt.
- **The input counter is a local convention.** It is not the provider's tokenizer
  (see below). In every run, the provider-reported input of every main request stayed within
  the dispatch limit (the window plus the tolerance). Some runs did exceed the nominal
  window: 1,395 Browse runs (745 ACM, 321 ACM (forced), 288 truncation only), 20 ARC runs and
  43 Science runs. All of these stayed within the tolerance.

## Protocol

| | BrowseComp-Plus | ARC-AGI-3 | Terminal-Bench-Science |
| --- | --- | --- | --- |
| Population | 830 questions, all 100,195 corpus documents | 25 public games, seed 0 | 70 tasks at revision `4dc977a3` |
| Order | SHA256 of `20261026:<id>` | same | same |
| Input window | 23,560 tokens (+2,356 at dispatch) | 32,000 (+3,200) | 32,000 (+3,200) |
| Main requests | 100 | 200 | 128 |
| Wall time | 1,800 s | 3,600 s | 7,200 s (verifier time from `task.toml`) |
| Other limits | | 300 game actions | |
| Tools | local BM25 `search` and `fetch` | local game HTTP service | the task's own environment |
| Score | strict answer match | official scorecard; levels completed | official verifier; tests passed |

The methods are `compaction`, `folding`, `clm`, `acm`, `selfcompact` and `rlm`, with Relay's
default settings for each. The model is `gpt-6-luna` with medium reasoning effort.

Two methods were added on 2026-11-06 and ran on BrowseComp-Plus and Terminal-Bench-Science:

- `truncation` (`relay/strategies/truncation.py`) leaves the context as Codex sent it. When a
  request is over the window, Relay's overflow fallback drops the oldest tool call groups. It
  is the baseline that every context-management method should beat.
- `acm_forced` is ACM with the forced recovery that released ACM lacks. When a request is over
  the window, the history since the last summary is compressed as `manage_context` would
  compress it, up to half the window so that the summarizer's own request fits, and the
  summary joins the conversation as a message marked as the system's.

The same day, 56 earlier Browse cells were run again: the 29 excluded as infrastructure
failures and the 27 in which the Browse profile was not applied. A rerun replaces the earlier
run of its cell in the results. These runs used the same runtime snapshot as the earlier
runs, with three changes: the two methods, the retry of dropped connections on Relay's own
requests, and the Browse profile refusing Codex instruction versions it has not checked. The
Codex instructions served on 2026-11-06 differ from the checked version by one word in a
section that the profile replaces, so the agent's instructions are the same. The agents'
sandbox also hid two more local paths (a source checkout and agent session files), which adds
their names to the permissions text the agent reads.

### BrowseComp-Plus retrieval

`browse_server.py` builds a SQLite FTS5 index of the full public corpus. Search keeps up to
eight of the query's longest terms of four or more letters and joins them with AND. If that
returns nothing, it retries with OR. It returns the top 10 documents by BM25, each with a
64-word snippet capped at 512 tokens. Fetch returns the start of a document, capped at 8,192
tokens. Workers never see answers or relevance labels. `browse_grade.py` grades runs after
they finish.

### Input accounting

`relay/core/budget.py` counts every request the same way during planning, during strategy
work, and at dispatch:

- **Text:** `o200k_base` tokens.
- **Tool declarations and response schemas:** counted from their decoded fields, with two
  units per container and one per separator.
- **Encrypted reasoning items:** the `reasoning_tokens` the provider reported for the
  response that produced them. Nothing is decrypted.
- **Framing:** eight units per input item and three per request.
- **Inline images** (ARC: PNG; Science continuation: PNG or JPEG):
  `128 + ceil(1.25 * ceil(w/32) * ceil(h/32))` per image. This is an empirical upper bound
  calibrated on `gpt-6-luna`.

A request over the window gets one forced recovery from its strategy. If it still does not
fit and the fallback is enabled, Relay drops the oldest tool call groups. A request that
still does not fit fails the run.

## Running

Install the evaluation extras:

```bash
pip install -e '.[eval,rlm]'
```

Run one cell with `run.py`. It starts an isolated Relay with the bounded budget, then runs
Codex in a sandbox that cannot read the benchmark's private files:

```bash
export RELAY_PILOT_ALL_CALL_BUDGET=1 RELAY_PILOT_INPUT_SLACK=2356 RELAY_PILOT_OVERFLOW_FALLBACK=truncate
python -m evals.pilot.run --root "$RELAY_PILOT_ROOT" --out runs/q1/acm --prompt prompt.txt \
  --strategy acm --model gpt-6-luna --window 23560 --max-requests 100 --seconds 1800 \
  --input-budget-counter o200k_fields_reasoning_usage_framing8_v3 --harness-profile browse_minimal_v1
```

The benchmark drivers wrap this runner:

- **BrowseComp-Plus:** `browse_server.py build`, then `browse_suite.py` per run, or
  `browse_queue.py` for a fixed list of cells (the runs of 2026-11-06 used it).
- **ARC-AGI-3:** `arc_server.py`, then `arc_suite.py`. `arc_sample.py` freezes a game
  sample.
- **Terminal-Bench-Science:** `science_container.py build`, then `science_sample_suite.py`
  per cell, or `science_sample_matrix.py` for a queue.

The runner uses Codex's ChatGPT sign-in. A quota guard stops admitting runs and stops
inference at configurable plan-usage thresholds (`RELAY_PILOT_QUOTA_*`). It never uses
credits or overage. The full-population runs were scheduled by cluster-specific queue
controllers. Those controllers are not included, but every cell ran through the runner and
drivers here.

## Reproducing the results

Set `RELAY_PILOT_ROOT` to the evaluation's working tree (data, runtimes and run outputs),
then run:

```bash
python -m evals.pilot.browse_context_collect --out rows-browse.json
python -m evals.pilot.context_collect arc --out rows-arc.json
python -m evals.pilot.context_collect science --out rows-science.json
python -m evals.pilot.context_report --browse rows-browse.json --arc rows-arc.json \
  --science rows-science.json --html evals/results/pilot/context-length.html \
  --json evals/results/pilot/context-length.json --cells evals/results/pilot/cells.csv
```

The raw trajectories are not in this repository. They are in the Hugging Face dataset
[`Corning/relay-context-eval-trajectories`](https://huggingface.co/datasets/Corning/relay-context-eval-trajectories),
one archive per benchmark and method, with a `runs.csv` that maps each row of `cells.csv` to
its files. They contain benchmark questions, answers and hidden test output in plain text, so
access is gated: you accept the terms of use (no training, no republishing) and access is
granted automatically.
