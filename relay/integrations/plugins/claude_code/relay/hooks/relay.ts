// Relay for Claude Code (relay/integrations/claude_code.py). `relay install claude_code --via hook`
// moves Claude Code's auto-compaction trigger to the bottom, so this hook runs before every request
// and asks Relay's strategy: "not now" skips the compaction (it leaves no trace), anything else is
// the conversation Claude Code continues from. The summaries Relay asks for are made with Claude
// Code's own model. When Relay cannot answer, nothing happens, unless Claude Code is near its own
// limit, where its own compaction runs.

type Message = { role: string; text: string; toolUses: any[]; toolResults?: any[]; handle?: string };
type Reply = {
  messages?: ({ ref: number } | { role: string; text: string })[];
  summarize?: { key: string; fork: string | null; complete: string };
  skip?: string;
};

const ROUNDS = 8; // summary requests per compaction (Relay asks for one at a time)
const MARGIN = 33000; // below the window, where Claude Code would have compacted at the latest
// The prompts of the forks making Relay's summaries. A fork carries the whole conversation, so
// it reaches the trigger too; it runs as it is, as Codex's summary request does.
const summarizing = new Set<string>();

export const register = (on: any) => {
  on('session.compact', async ($: any, e: any, next: any) => {
    if (e.trigger === 'precompute') return { skip: 'Relay decides before each request' };
    if (e.agentId && summarizing.has(e.messages.at(-1)?.text)) return { skip: 'Relay is summarizing' };
    const { context } = await $.session.usage();
    try {
      return await decide($, e, context);
    } catch (error) {
      $.ui.log(`relay: ${error instanceof Error ? error.message : String(error)}`);
      return (context.tokens ?? 0) >= context.window - MARGIN ? next(e) : { skip: 'Relay did not answer' };
    }
  });
};

async function decide($: any, e: any, context: { tokens?: number; window: number }) {
  const url = `${(await $.env.get('RELAY_HOOK_URL')) || 'http://127.0.0.1:8787'}/relay/v1/compact`;
  const request = {
    harness: 'claude_code',
    session: await $.session.id(), // the strategy's state is kept by session and agent
    agent: e.agentId ?? null,
    trigger: e.trigger,
    model: await $.session.model(),
    tokens: context.tokens ?? null, // the upstream's count of the last request
    window: context.window,
    messages: e.messages.map(({ handle, ...message }: Message) => message),
    summaries: {} as Record<string, string>,
  };
  for (let round = 0; round < ROUNDS; round++) {
    const response = await $.http.fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error(`Relay answered ${response.status}: ${response.text.slice(0, 300)}`);
    const reply: Reply = JSON.parse(response.text);
    if (reply.messages) {
      return {
        messages: reply.messages.map((m) => ('ref' in m ? e.messages[m.ref] : { role: m.role, text: m.text, toolUses: [] })),
      };
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
