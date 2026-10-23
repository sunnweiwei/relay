// Relay for Claude Code (relay/integrations/claude_code.py). As a session starts, this plugin moves
// Claude Code's auto-compaction trigger to the bottom, so the compaction hook runs before every
// request and asks Relay's strategy: "not now" skips the compaction (it leaves no trace), anything
// else is the conversation Claude Code continues from. The strategy's instructions join Claude
// Code's system prompt, which it builds once a session; a shell result the strategy only gives new
// text is written as it comes in. The summaries Relay asks for are made
// with Claude Code's own model. When Relay cannot answer, nothing happens, unless Claude Code is
// near its own limit, where its own compaction runs.

type Message = { role: string; text: string; toolUses: any[]; toolResults?: any[]; handle?: string };
type Reply = {
  messages?: ({ ref: number } | { role: string; text: string; toolUses: any[]; toolResults?: any[] })[];
  summarize?: { key: string; fork: string | null; complete: string };
  skip?: string;
};

const ROUNDS = 8; // summary requests per compaction (Relay asks for one at a time)
const MARGIN = 33000; // below the window, where Claude Code would have compacted at the latest
const TRIGGER = '0.01'; // Claude Code's trigger, in percent of its window: below any request
const SECTION = 'context_management'; // the last section of Claude Code's system prompt
const TRIES = 5; // asking for the instructions, a second apart
// The prompts of the forks making Relay's summaries. A fork carries the whole conversation, so
// it reaches the trigger too; it runs as it is, as Codex's summary request does.
const summarizing = new Set<string>();
let placed: string | null = null; // the instructions the system prompt got (null: none asked yet)
// By agent ('' the session's own loop): the prompt tokens of its latest request, as the upstream
// counted them (a sub-agent's are not in $.session.usage); requests since its latest compaction, and
// compactions in a row each within RAPID requests of the one before. Claude Code ends a turn with a
// third such compaction (its thrashing breaker); these are counted as it counts them, by turn.
const RAPID = 3;
const sizes = new Map<string, number>();
const since = new Map<string, number>();
const rapid = new Map<string, number>();

export const register = (on: any) => {
  on('session.start', async ($: any, e: any, next: any) => {
    if (!(await proxied($))) await $.env.set('CLAUDE_AUTOCOMPACT_PCT_OVERRIDE', TRIGGER);
    return next(e);
  });
  on('turn.start', async ($: any, e: any, next: any) => {
    since.delete(e.agentId ?? '');
    rapid.delete(e.agentId ?? '');
    return next(e);
  });
  on('turn.step', async function* ($: any, e: any, next: any) {
    const step = yield* next(e);
    const u = step?.usage;
    if (u && typeof u.input_tokens === 'number')
      sizes.set(e.agentId ?? '', u.input_tokens + (u.cache_read_input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0));
    return step;
  });
  on('prompt.section', { name: SECTION }, async ($: any, e: any, next: any) => {
    const section = await next(e);
    if (await proxied($)) return section;
    placed = '';
    for (let attempt = 0; attempt < TRIES; attempt++) {
      try {
        placed = (await post($, 'instructions', { harness: 'claude_code', version: await version($) })).instructions ?? '';
        break;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (attempt === TRIES - 1 || message.startsWith('Relay answered')) {
          $.ui.log(`relay: ${message}`);
          break;
        }
        await $.clock.sleep(1000); // (Relay may be starting, from the session's start)
      }
    }
    if (!placed) return section;
    return { text: section.text ? `${section.text}\n\n${placed}` : placed };
  });
  // A shell result as it comes in: where all the strategy changes is its text, that is written
  // as the result (no compaction); anything else waits for the next request.
  on('tool.call', { tool: 'Bash' }, async ($: any, e: any, next: any) => {
    const done = await next(e);
    if (typeof done?.text !== 'string' || typeof done.result?.stdout !== 'string' || (await proxied($))) return done;
    try {
      const reply = await post($, 'result', {
        harness: 'claude_code',
        session: await $.session.id(),
        agent: e.agentId ?? null,
        model: await $.session.model(),
        instructions: placed ?? '',
        messages: await $.session.messages(e.agentId ? { agentId: e.agentId } : {}),
        call: { tool_use_id: e.tool_use_id, text: done.text },
      });
      if (typeof reply.text === 'string') return { ...done, result: { ...done.result, stdout: reply.text, stderr: '' } };
    } catch (error) {
      $.ui.log(`relay: ${error instanceof Error ? error.message : String(error)}`);
    }
    return done;
  });
  on('session.compact', async ($: any, e: any, next: any) => {
    if (e.trigger === 'precompute') return { skip: 'Relay decides before each request' };
    if (await proxied($)) return next(e);
    if (e.agentId && summarizing.has(e.messages.at(-1)?.text)) return { skip: 'Relay is summarizing' };
    const { context } = await $.session.usage();
    const agent = e.agentId ?? '';
    since.set(agent, (since.get(agent) ?? Infinity) + 1);
    try {
      const done = await decide($, e, context);
      if (done.messages) {
        rapid.set(agent, (since.get(agent) ?? Infinity) <= RAPID ? (rapid.get(agent) ?? 0) + 1 : 0);
        since.set(agent, 0);
      }
      return done;
    } catch (error) {
      $.ui.log(`relay: ${error instanceof Error ? error.message : String(error)}`);
      return (context.tokens ?? 0) >= context.window - MARGIN ? next(e) : { skip: 'Relay did not answer' };
    }
  });
};

// A session whose requests go through Relay's proxy (one set up before the hook, say): Relay
// decides there already.
async function proxied($: any): Promise<boolean> {
  const base = (await $.env.get('ANTHROPIC_BASE_URL')) || '';
  return base !== '' && base.startsWith(await url($));
}

async function url($: any): Promise<string> {
  return (await $.env.get('RELAY_HOOK_URL')) || 'http://127.0.0.1:8787';
}

async function version($: any): Promise<string | null> {
  return (await $.session.version())?.version ?? null;
}

async function post($: any, endpoint: string, request: any) {
  const response = await $.http.fetch(`${await url($)}/relay/v1/${endpoint}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(`Relay answered ${response.status}: ${response.text.slice(0, 300)}`);
  return JSON.parse(response.text);
}

async function decide($: any, e: any, context: { tokens?: number; window: number }) {
  const request = {
    harness: 'claude_code',
    version: await version($),
    session: await $.session.id(), // the strategy's state is kept by session and agent
    agent: e.agentId ?? null,
    trigger: e.trigger,
    model: await $.session.model(),
    // the upstream's count of the agent's latest request
    tokens: e.agentId ? sizes.get(e.agentId) ?? null : context.tokens ?? null,
    // another compaction now would be the second in a row within RAPID requests: Relay holds it
    // (unless forced), so the next request does not trip Claude Code's breaker
    hold: (since.get(e.agentId ?? '') ?? Infinity) <= RAPID && (rapid.get(e.agentId ?? '') ?? 0) >= 1,
    window: context.window,
    instructions: placed ?? '',
    messages: e.messages, // (each with its handle: its row, which a resumed session brings back)
    summaries: {} as Record<string, string>,
  };
  for (let round = 0; round < ROUNDS; round++) {
    const reply: Reply = await post($, 'compact', request);
    if (reply.messages) {
      return { messages: reply.messages.map((m) => ('ref' in m ? e.messages[m.ref] : m)) };
    }
    if (!reply.summarize) return { skip: reply.skip ?? 'not now' };
    const { key, fork, complete } = reply.summarize;
    let answer;
    if (fork) {
      summarizing.add(fork);
      try {
        answer = await $.model.fork({ prompt: fork });
      } finally {
        summarizing.delete(fork);
      }
    }
    if (!answer?.isAnswered) answer = await $.model.complete({ model: request.model, prompt: complete, maxTokens: 16000 });
    if (!answer.isAnswered) throw new Error(`the summary request failed (${answer.reason})`);
    request.summaries[key] = answer.text;
  }
  throw new Error(`more than ${ROUNDS} summary requests`);
}
