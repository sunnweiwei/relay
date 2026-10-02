# Compact threshold question for discussion

The committed Relay `Compact` uses `compact_threshold` both to trigger
compaction (`_over_threshold`) and as the maximum token count of each summary
request (`_largest_fitting_boundary`). A complete message or paired tool
transaction plus the compaction prompt (and sometimes the previous summary and
instructions) may exceed that same threshold. In that case no candidate
boundary fits, `ValueError` is raised, and the proxy responds with HTTP 400
before calling the summary model.

Questions for the team:

1. Why must a summary request fit under the *trigger* threshold? Is the
   intended limit instead the summary model's context window or an independent
   summary input budget?
2. Should the latest complete user message remain outside the summarized
   prefix? The current implementation summarizes the whole active trajectory
   and later retains recent user text separately.
3. Should the prepared input be checked against the trigger threshold to avoid
   another compaction on the next model call?

For Pi, mini-SWE, Gemini CLI, and Claude Code, the follow-up live diagnostic
separates the trigger threshold from a bounded summary input budget in the
experiment runner only. Its result can diagnose Harness/Relay integration but
does not count as validation of the committed Compact policy. The strategy
module `relay/strategies/compact.py` is unchanged.

2026-10-02 GPT-6 Luna diagnostic observations: Pi, mini-SWE, and Gemini CLI
each completed a real model run with compaction and an exact cache checkpoint
reuse. Mini-SWE needed its native bash submission sentinel in the task prompt;
Gemini CLI needed a 36,000-token cumulative task input budget because its
first request already contained over 8,000 tokens. All three remained bounded
per request. These observations do not settle the threshold policy question.

## 2026-10-02 local-file tool case

The same two actual user messages and two local JSON files were run with real
GPT-6 Luna. Pi, OpenCode, mini-SWE and Gemini CLI completed both turns,
compacted after a real tool result, and restored an exact checkpoint on the
second user turn. Gemini CLI's 2250 answer used a thousands separator (`2,250`);
the existing run was re-evaluated offline after fixing that answer predicate.
No extra API request was made for the correction.

Codex completed both answers and compacted. It hit a checkpoint inside the
first user turn, but the second user's `exec resume --last` request changed two
earlier `reasoning` items. Raw field capture showed that their only change was
an absent `content` field becoming `null`; the cache namespace/scope and all
other item fields stayed the same. Cache-key normalization now equates only
those two representations for reasoning items. A later reasoning-enabled run
passed all checks and matched the checkpoint on the second user turn. The
earlier `model_reasoning_effort = "none"` run also passed, but is no longer the
only passing Codex configuration.

Claude Code required a tool bridge for GPT Responses. Empty `Read.pages` from
the model was rejected by the Claude tool, so the bridge now omits that empty
optional argument. Both answers then passed. However, the resumed Claude Code
history omitted an earlier `Glob` call/output pair that existed when the first
checkpoint was stored; exact-prefix Cache missed and the second turn
re-summarized. The adapter does not equate those different histories.
Restricting Claude Code to `Read` alone was also tried within the same bounded
case. It repeatedly read files until the request cap, so the runner was
returned to `Read,Glob,Grep`; no further paid retries were made.

The per-run `wire_trace` in `runs/*-live-local-files/result.json` records
request/turn, item types, tool names and call IDs, tool-result error flags,
summary forwarding, cache depth and item digests. It intentionally does not
store full Harness request bodies or provider credentials.
