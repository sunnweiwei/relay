"""Explicit evaluation-only reduction of known Codex boilerplate.

This changes the experimental harness and must be versioned in results. It
does not change tool contracts, permissions, task history, or method prompts.
Other developer text is preserved. A Codex coding instruction other than the
one this profile was checked against is an error: forwarding it unchanged
would silently run a different harness condition under the profile's name.
"""

from __future__ import annotations

import re
import hashlib

from ..protocols.base import Body

PROFILE = "browse_minimal_v1"
CODING_PREFIX = "You are Codex, an agent based on GPT-6. You and the user share one workspace, and your job"
CODING_SHA256 = "a2735be23ce52098b1f735146accfb6b92ce819c8c69a07b8449a08d71260d56"
# Codex model metadata served from 2026-11-06 changes one word of the skills section. The
# profile replaces the whole text, so the agent reads the same instructions.
LATER_CODING_SHA256 = ("e52174f354e3970ad8de5935fff8d04ddf2890a35659283d163befb56490198c",)
BROWSE_INSTRUCTIONS = (
    "You are an agent solving a BrowseComp-Plus research task. Follow the task, "
    "tool contracts, and filesystem/network permissions. Use the provided "
    "search and fetch interface to gather evidence and finish the requested "
    "answer. Follow the supplied context-management instructions and the "
    "task's delegation limits."
)


def apply_profile(body: Body, profile: str | None) -> Body:
    if profile is None:
        return body
    if profile != PROFILE:
        raise ValueError(f"Unknown evaluation harness profile: {profile}")
    result = []
    for item in body.get("input") or []:
        if item.get("type", "message") != "message" or item.get("role") != "developer":
            result.append(item)
            continue
        content = item.get("content")
        if not isinstance(content, list):
            result.append(item)
            continue
        parts = []
        for part in content:
            if part.get("type") != "input_text":
                parts.append(part)
                continue
            original = text = part.get("text", "")
            if text.startswith(CODING_PREFIX):
                if hashlib.sha256(text.encode()).hexdigest() not in (CODING_SHA256, *LATER_CODING_SHA256):
                    raise ValueError(f"{PROFILE}: unrecognized Codex coding instructions; refusing to "
                                     "forward them under this profile")
                text = BROWSE_INSTRUCTIONS
            else:
                for tag in ("skills_instructions", "recommended_plugins", "multi_agent_role"):
                    text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)
            # Unchanged parts (notably permissions) remain byte-identical.
            if text == original or text.strip():
                parts.append({**part, "text": text})
        if parts:
            result.append({**item, "content": parts})
    return {**body, "input": result}
