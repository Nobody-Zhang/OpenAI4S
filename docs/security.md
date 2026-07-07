# Security

> ⚠️ Read this before exposing the daemon beyond `localhost`.

The daemon runs agent-authored code with **no OS-level sandbox** (no Seatbelt / bubblewrap) — `kernel/execute`, `compute/jobs`, and `host.bash` are equivalent to a shell on the host. This is fine for a single-user local tool bound to `127.0.0.1` (the default). On top of that, [`openai4s.security`](../openai4s/security) adds software layers reverse-engineered from Claude Science — all **opt-out via env**, all **fail-open** when no base model is set:

| layer | env (default) | what it does |
|---|---|---|
| **Pre-exec classifier** | `OPENAI4S_SAFETY` (`heuristic`) | screens every *agent-authored* cell before it runs (`heuristic` / `llm` / `off`); your own Notebook cells are never screened |
| **`dlopen` audit hook** | `OPENAI4S_SAFETY_AUDIT_HOOK` (on) | `sys.addaudithook` refuses `ctypes.dlopen` of a `.so` from an agent-writable path |
| **Biosecurity screener** | `OPENAI4S_BIOSECURITY` (on) | trajectory screener (ALLOW / ESCALATE / BLOCK) on biosecurity-relevant content |
| **Injection detector** | `OPENAI4S_INJECTION_SCAN` (on) | annotates tool-returned content (web / PDF / MCP) so the model treats it as **data, not instructions** |
| **Egress allowlist** | `OPENAI4S_EGRESS` (`off`) | fences `web_fetch` / `web_search` / `bash` to science APIs & package indexes; blocked domains recover via `host.request_network_access(domain=…)`, which **you** approve |

Additional enforcement: an opencode-style **permission broker** gates risk-bearing tools, a **secret-file guard** blocks `.env` / `*.key` / `id_rsa` from all file tools, and every file/shell op is **workspace-jailed**.

### Secret reads and secret logs

The agent can introspect its own SQLite store through the read-only `host.query`, so secret-bearing tables are **denylisted** and never reach it:

- `host.query` refuses any statement that references `settings` (the live LLM API key + saved model profiles), `connectors` (MCP server env / launch command), `memories`, `host_call_log`, or `permission_rules`. `host.query.schema()` also hides these tables. The check runs against a copy with single-quoted string literals and comments stripped, so a denied word appearing only inside a literal (e.g. `SELECT 'settings' AS note`) is not falsely rejected, while an identifier-quoted table reference (`FROM "settings"`) still trips it.
- Because the denylist is a table-name match, a query that reads the unrelated `agents.connectors` *column* is also refused; no bundled skill relies on that read.

Credential values passed to `host.credentials.set(name, value)` are held only in an in-memory vault (never persisted). To keep that true end to end, the **RPC audit log** redacts them: `credentials_get` / `credentials_list` are not logged at all, and `credentials_set` is logged for audit **with its args redacted** — the plaintext value never enters `host_call_log`. The replay tape recorder likewise skips `credentials_set`, so an exported notebook cannot carry a plaintext credential.

## Remote access

The daemon binds `127.0.0.1` by default. Reach the UI over an SSH tunnel — **never** expose `0.0.0.0` on an untrusted network:

```bash
ssh -L 8760:127.0.0.1:8760 user@your-host
```

If you must bind a non-loopback address (`OPENAI4S_HOST=0.0.0.0`) or set `OPENAI4S_REQUIRE_TOKEN=1`, the server prints a one-time access token at startup and rejects any request without `?token=…` (`401`).
