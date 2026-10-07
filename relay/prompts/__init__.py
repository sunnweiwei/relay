"""Compaction prompts, verbatim.

Codex's: codex-rs/prompts/templates/compact/{prompt,summary_prefix}.md at commit
d25c114d494ddb693290b76bf5e5f64ecbdb38fc. Claude Code's: the prompt its `/compact` and
auto-compaction send (2.1.292, the default "control" variant), as it sends it. Crush's: its
summary system prompt (internal/agent/templates/summary.md, 0.97.1). WorkBuddy's: CodeBuddy
Code's context-summary-max-token-prompt (2.161.4). Goose's: prompts/compaction.md (1.53.0), the
conversation it embeds left out. nanobot's: consolidator_archive.md (0.3.5). pi's: its
summarization system prompt and instructions (1.0.4). Hermes Agent's: its compression prompt
(0.19.0), the turns it embeds and its focus topic left out. DeepSeek Harness's: compaction-basic's
(0.2.0-rc.2). OpenCode's (and Kilo's): its compaction agent's prompt and instructions (1.18.35),
the conversation it embeds left out. Kimi Code's: its compaction instruction (2.1.1). Gemini
CLI's: its compression prompt and the request for a new snapshot (0.63.0); Kimi Code's note on
recovering the context from its event log, its path and line windows left to fill in.
"""

from pathlib import Path

_DIR = Path(__file__).parent

SUMMARIZATION_PROMPT = (_DIR / "codex_compact.md").read_text(encoding="utf-8")
SUMMARY_PREFIX = (_DIR / "codex_summary_prefix.md").read_text(encoding="utf-8")
CLAUDE_CODE_PROMPT = (_DIR / "claude_code_compact.md").read_text(encoding="utf-8")
CRUSH_PROMPT = (_DIR / "crush_summary.md").read_text(encoding="utf-8")
WORKBUDDY_PROMPT = (_DIR / "workbuddy_summary.md").read_text(encoding="utf-8")
GOOSE_PROMPT = (_DIR / "goose_compaction.md").read_text(encoding="utf-8")
NANOBOT_PROMPT = (_DIR / "nanobot_archive.md").read_text(encoding="utf-8")
PI_PROMPT = (_DIR / "pi_compaction.md").read_text(encoding="utf-8")
HERMES_PROMPT = (_DIR / "hermes_compression.md").read_text(encoding="utf-8")
DSH_PROMPT = (_DIR / "dsh_compaction.md").read_text(encoding="utf-8")
OPENCODE_PROMPT = (_DIR / "opencode_compaction.md").read_text(encoding="utf-8")
KIMI_PROMPT = (_DIR / "kimi_compaction.md").read_text(encoding="utf-8")
GEMINI_PROMPT = (_DIR / "gemini_compression.md").read_text(encoding="utf-8")
KIMI_RECOVERY = (_DIR / "kimi_context_recovery.md").read_text(encoding="utf-8")
