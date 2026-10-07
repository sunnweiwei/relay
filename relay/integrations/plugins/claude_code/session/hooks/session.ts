// Relay for Claude Code through the proxy (relay/core/local.py). Before each model request this
// reports to Relay what Claude Code knows of the session that the request does not show: where
// the conversation's whole transcript is (a sub-agent's own), the instruction files as they read
// now, the files read last (as Claude Code re-reads them after compacting), the skills run and
// the background agents. Relay's compaction then lays the request out as Claude Code's own does.
// File contents go only when a file changed since the last report (Relay keeps what it had). A
// report that fails costs nothing but that: Relay lays out what the request shows.

const FILES = 10; // Claude Code re-reads the 5 files read last that the context does not show...
const FILE_CHARS = 20000; // ... about 5k tokens each
const SKILL_CHARS = 20000;
const sent = new Map<string, string>(); // per conversation: the files and instructions last sent, by mtime
const projects = new Map<string, string>(); // per session: its folder of transcripts

export const register = (on: any) => {
  on('turn.step', async function* ($: any, e: any, next: any) {
    try {
      await report($, e.agentId);
    } catch (error) {
      $.ui.log(`relay: ${error instanceof Error ? error.message : String(error)}`);
    }
    return yield* next(e);
  });
};

async function report($: any, agentId?: string) {
  const url = `${(await $.env.get('RELAY_URL')) || 'http://127.0.0.1:8787'}/relay/v1/local`;
  const [session, cwd] = await Promise.all([$.session.id(), $.session.cwd()]);
  const found = await $.session.messages(agentId ? { agentId } : undefined);
  const rows = 'deny' in found ? [] : found;
  const uses = rows.flatMap((message: any) => message.toolUses ?? []);
  // Requests do not say whose conversation they continue: Relay tells by its first user message.
  const opening = rows.find((message: any) => message.role === 'user' && !message.toolResults?.length)?.text ?? '';
  const paths: string[] = [];
  for (const use of [...uses].reverse()) {
    const path = use.tool === 'Read' && !use.isError ? String(use.input?.file_path ?? '') : '';
    if (path && !paths.includes(path)) paths.push(path);
    if (paths.length === FILES) break;
  }
  const { context } = await $.session.usage({ breakdown: 'summary' });
  const memory = (context.breakdown?.memoryFiles ?? []).map((file: any) => ({ path: file.path, kind: String(file.type).toLowerCase() }));
  const stamps = await Promise.all([...paths, ...memory.map((m: any) => m.path)].map(async (path) => {
    try {
      return `${path}@${(await $.fs.stat(path)).mtimeMs}`;
    } catch {
      return `${path}@gone`;
    }
  }));
  const key = `${session}:${agentId ?? ''}`;
  const changed = sent.get(key) !== stamps.join('\n');
  const body: any = { harness: 'claude_code', session, agent: agentId ?? null, opening, cwd,
                      transcript: await transcript($, session, cwd, agentId) };
  if (changed) {
    body.files = [];
    for (const path of paths) {
      try {
        body.files.push({ path, content: String(await $.fs.read(path)).slice(0, FILE_CHARS) });
      } catch {} // gone since
    }
    body.instructions = [];
    for (const file of memory) {
      try {
        body.instructions.push({ ...file, content: String(await $.fs.read(file.path)) });
      } catch {}
    }
  }
  body.skills = uses
    .filter((use: any) => use.tool === 'Skill' && !use.isError)
    .map((use: any) => ({ path: String(use.input?.skill ?? ''), content: String(use.text ?? '').slice(0, SKILL_CHARS) }));
  body.tasks = (await $.agent.list())
    .filter((agent: any) => agent.status === 'running' && agent.id !== agentId)
    .map((agent: any) => ({ id: agent.id, kind: 'agent', description: agent.description, status: agent.status }));
  const response = await $.http.fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`Relay answered ${response.status}: ${response.text.slice(0, 300)}`);
  if (changed) sent.set(key, stamps.join('\n'));
  else if (paths.length && JSON.parse(response.text).files === false) sent.delete(key); // Relay lost them: again next time
}

// Where Claude Code keeps the conversation: the session's transcript in its project's folder (found
// rather than derived: Claude Code shortens long folder names), a sub-agent's under the session's.
async function transcript($: any, session: string, cwd: string, agentId?: string): Promise<string | null> {
  let folder = projects.get(session);
  if (!folder) {
    const root = `${(await $.env.get('CLAUDE_CONFIG_DIR')) || `${await $.env.get('HOME')}/.claude`}/projects`;
    const guess = `${root}/${cwd.replace(/[^a-zA-Z0-9]/g, '-')}`;
    if (await $.fs.exists(`${guess}/${session}.jsonl`)) folder = guess;
    else {
      for (const entry of await $.fs.list(root)) {
        if (entry.kind === 'dir' && (await $.fs.exists(`${root}/${entry.name}/${session}.jsonl`))) {
          folder = `${root}/${entry.name}`;
          break;
        }
      }
    }
    if (!folder) return null;
    projects.set(session, folder);
  }
  return agentId ? `${folder}/${session}/subagents/agent-${agentId}.jsonl` : `${folder}/${session}.jsonl`;
}
