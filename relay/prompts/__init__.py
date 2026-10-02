"""Compaction prompts vendored verbatim from openai/codex.

Source: codex-rs/prompts/templates/compact/{prompt,summary_prefix}.md at commit
d25c114d494ddb693290b76bf5e5f64ecbdb38fc.
"""

from pathlib import Path

_DIR = Path(__file__).parent

SUMMARIZATION_PROMPT = (_DIR / "codex_compact.md").read_text(encoding="utf-8")
SUMMARY_PREFIX = (_DIR / "codex_summary_prefix.md").read_text(encoding="utf-8")
