"""Native Gemini text/tool-history adapter; unsupported parts fail closed."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


def _key(item: dict[str, Any]) -> str:
    return json.dumps(item, sort_keys=True, separators=(",", ":"))


class GeminiInput:
    def __init__(self, body: dict[str, Any], model: str) -> None:
        self.body = deepcopy(body)
        self.original: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
        items = []
        pending: dict[str, list[str]] = {}
        completed: dict[str, tuple[str, str]] = {}
        for index, content in enumerate(body.get("contents", [])):
            role = content.get("role", "user")
            if role not in {"user", "model"}:
                raise ValueError("unsupported Gemini content role")
            for offset, part in enumerate(content.get("parts", [])):
                kinds = set(part) - {"thoughtSignature", "thought"}
                if kinds == {"text"}:
                    if part.get("thought"):
                        raise ValueError("Gemini thought text is not yet supported")
                    item = {"type": "message", "role": "assistant" if role == "model" else "user",
                            "content": part["text"]}
                elif kinds == {"functionCall"} and role == "model":
                    call = part["functionCall"]
                    call_id = call.get("id") or f"gemini_{index}_{offset}"
                    pending.setdefault(call["name"], []).append(call_id)
                    item = {"type": "function_call", "call_id": call_id, "name": call["name"],
                            "arguments": json.dumps(call.get("args", {}), sort_keys=True)}
                elif kinds == {"functionResponse"} and role == "user":
                    result = part["functionResponse"]
                    candidates = pending.get(result["name"], [])
                    response_id = result.get("id")
                    response_body = json.dumps(result.get("response", {}), sort_keys=True)
                    if (isinstance(response_id, str) and response_id in completed
                            and completed[response_id] == (result["name"], response_body)):
                        # Gemini CLI can repeat a completed tool result in the
                        # same history. Keep one canonical call/output pair.
                        continue
                    call_id = response_id if response_id in candidates else None
                    if call_id is None and candidates:
                        # Gemini CLI can assign a tool-result ID after the model
                        # omitted the call ID. Match that result by tool name.
                        synthetic = next((candidate for candidate in candidates
                                          if candidate.startswith("gemini_")), None)
                        if response_id is None or synthetic is not None:
                            call_id = synthetic or candidates[0]
                    if call_id is None:
                        raise ValueError("Gemini tool response has no matching call")
                    candidates.remove(call_id)
                    completed[call_id] = (result["name"], response_body)
                    item = {"type": "function_call_output", "call_id": call_id,
                            "output": response_body}
                else:
                    raise ValueError("unsupported Gemini part; images/files are not yet supported")
                # Lists handle identical text messages without collapsing their identity.
                key = _key(item)
                self.original.setdefault(key, []).append((index, role, deepcopy(part)))
                items.append(item)
        self.request = {"model": model, "input": items}
        # Keep system instructions and tool schemas in the cache scope and manager prompt.
        if "systemInstruction" in body:
            self.request["instructions"] = json.dumps(body["systemInstruction"], sort_keys=True)
        if "tools" in body:
            self.request["instructions"] = self.request.get("instructions", "") + "\nGemini tools: " + json.dumps(body["tools"], sort_keys=True)

    def render(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        originals = deepcopy(self.original)
        contents = []
        previous_group = None
        for item in items:
            matches = originals.get(_key(item), [])
            if matches:
                group, role, part = matches.pop(0)
            elif item.get("type", "message") == "message":
                group = None
                item_role = item.get("role")
                if item_role not in {"assistant", "user"}:
                    raise ValueError("unsupported generated Gemini message role")
                role = "model" if item_role == "assistant" else "user"
                value = item.get("content", "")
                if isinstance(value, list):
                    if any(p.get("type") not in {"input_text", "output_text"} for p in value):
                        raise ValueError("unsupported generated Gemini message content")
                    value = "".join(p.get("text", "") for p in value)
                if not isinstance(value, str):
                    raise ValueError("unsupported generated Gemini message")
                part = {"text": value}
            else:
                raise ValueError("strategy generated an unsupported Gemini item")
            if contents and group is not None and group == previous_group:
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": role, "parts": [part]})
            previous_group = group
        forwarded = deepcopy(self.body)
        forwarded["contents"] = contents
        return forwarded
