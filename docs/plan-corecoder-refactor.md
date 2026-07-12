# Plan: CoreCoder-style agents refactor — host executes only Jupyter/R instructions

Status: **implemented historical stage.** Its kernel/R/shell/completion
decisions remain active. Its fenced-tool-only decision was superseded by the
hybrid control plane: provider-native JSON tools now handle orchestration while
Python/R Code-as-Action remains the scientific layer. See
[`backend-refactor-architecture.md`](backend-refactor-architecture.md).

## Goal

Referencing CoreCoder's minimal agent architecture (declarative Tool registry, one
small loop core, sub-agent-as-capability), restructure the agents architecture so
that **the host executes exactly two kinds of instructions**:

1. **Python cells** on the persistent Jupyter-style kernel (`kernel/worker.py`), and
2. **R cells** on a new first-class persistent R kernel (`kernel/r_worker.R`),

removing the host-as-shell-executor role (`_m_bash` running `subprocess` in the
daemon process). Shell work moves *into the kernel*: `host.bash` becomes a
kernel-local SDK helper, and the ReAct `bash` tool is removed.

## Decisions (settled from the architecture map)

| # | Question | Decision |
|---|---|---|
| D1 | Wire protocol | **Keep the text-fence contract** (```python / ```r cells + ```tool JSON). No native function calling — the Code-as-Action identity (CLAUDE.md) and the pure-stdlib 3-wire client stay untouched. CoreCoder is referenced *structurally*: declarative registry, one action-parsing core, narrowed executor surface. |
| D2 | R executor shape | **Full R worker speaking the same JSON-per-line frame protocol** (`kernel/r_worker.R`), driven by the *existing* `Kernel` manager via a new `argv` override. No sentinel dialect, no python-wrapper. v1 emits no `host_call` frames (analysis kernel — no `host` object in R). |
| D3 | Protocol-wire safety in R | Spawn via `sh -c 'exec "$0" --vanilla "$1" 3>&1 4<&0 </dev/null 1>&2' <Rscript> <r_worker.R>`: protocol OUT on fd 3, protocol IN on fd 4, user stdin `/dev/null`, stray fd-1 writes land on stderr — the same fd discipline `worker.py` gets from its dup2 swap, done with shell redirections. `exec` keeps the pid = R's pid so `interrupt()`'s SIGINT reaches R. |
| D4 | Kernels per session | **Two, side by side.** The Python kernel remains the control plane (`host.*`, `submit_output`). The R kernel is a lazy analysis peer, spawned on the first ```r cell; both namespaces persist. `env_use` keeps swapping the Python kernel; selecting an R-only env retargets the R kernel instead of being refused. |
| D5 | Unlabeled ``` fence | Still means Python (back-compat). R requires an explicit ```r / ```R label. |
| D6 | R completion | **Observation-only.** `host.submit_output` stays reachable from Python cells / the tool path only; the system prompt says so. Matches the SDK's pre-built ANALYSIS-mode gate (`build_host(mode="r")`). |
| D7 | R safety gating | The pre-exec gates run on R cells too: `classify_code` + biosecurity trajectory screen are text-level and fail-open; no R-specific static tier in v1. |
| D8 | `host.bash` | **Kernel-local.** `sdk/host.py` implements `bash()` inside the worker process (`subprocess.run(shell=True, cwd=os.getcwd())` — the worker cwd is the session workspace in the gateway; PATH already carries the active env's `bin` via `_child_env`). It keeps the static `precheck_command` gate (moved to `openai4s/security/shellcheck.py`) and the `egress.scan_command` fence. `_m_bash` and the ReAct `bash` tool are deleted. Remote-GPU ssh provisioning keeps working unchanged (the agent still calls `host.bash("ssh …")`; it now runs in the kernel). |
| D9 | R figures/artifacts | Workspace-diff capture only (`ggsave` into the workspace); no matplotlib-style device capture, no provenance for R cells in v1. `_capture_snippet` (a Python cell) must not run after R cells. |
| D10 | `env_setup` pip install | Out of scope: env *provisioning* by the host (same class as spawning kernels), kept as-is. |
| D11 | R interpreter resolution | Selected env's `rscript` → discovered env named `r` → `Rscript` on PATH. Never silently substitute Python (constraint 12). `r_kernel` capability flag becomes a live probe. |

## Result-frame contract (unchanged, now bilingual)

Both workers emit `{type:"response", id, stdout, stderr, error, interrupted,
trace:{error_lineno,error_call}, usage:{wall_s,cpu_s,peak_rss_kb}}`. worker.R:
`sink()` for stdout/messages during eval, expression-by-expression eval of
`parse(text=code, keep.source=TRUE)` for a best-effort `error_lineno` (srcref of
the failing expression), `tryCatch(interrupt=)` → `interrupted`, `proc.time()`
for cpu. Inbound JSON parsed with `jsonlite` (pinned in `envs/r.yml`); a
jsonlite-less R reports a structured error frame (outbound JSON is hand-escaped,
dependency-free).

## New/changed modules

- `openai4s/agent/actions.py` (new) — the shared action-parsing core both loops
  import: `extract_action(reply) -> ("python"|"r", code) | None`,
  `count_code_blocks`, shared nudge/multi-cell-note text. Kills the drift between
  the duplicated filters in `loop.py` and `gateway.py` (`_extract_code` stays as a
  thin wrapper for compat).
- `openai4s/kernel/manager.py` — optional `argv: list[str]` ctor override (default
  unchanged); everything else (execute loop, restart/generation, interrupt,
  soft-fail) is already language-neutral.
- `openai4s/kernel/r_worker.R` (new) + `openai4s/kernel/r_kernel.py` (new) —
  `resolve_r_interpreter(env) -> str|None`, `spawn_r_kernel(cwd, rscript) -> Kernel`.
- `openai4s/sdk/host.py` — kernel-local `bash()`.
- `openai4s/security/shellcheck.py` (new home of `precheck_command`).
- `openai4s/tools/` — `bash` removed from `REGISTRY`; `render_tools_prompt` reworded.
- `openai4s/host_dispatch.py` — `_m_bash` deleted; `_m_env_use` R-env branch
  retargets the R kernel; `r_kernel` capability probed.
- Both loops + both system prompts: ```r routing, bash rewording, R guidance.
- `gateway.py` — `_execute_and_log(language=...)`, R kernel lifecycle on
  `SessionState`, `kernel_id "r — <env>"` labels, watchdog parameterized by kernel,
  R cells skip `_capture_snippet`.

## Tests

- `tests/test_r_kernel.py` (new): host-side plumbing against a **fake interpreter**
  (a python script speaking the frame protocol over fd3/fd4) — spawn, execute,
  restart, interrupt, shutdown; plus `skipif(no Rscript)` real-R integration
  (persistent namespace, error lineno, interrupt).
- `test_agent.py`: ```r routing via a stubbed R kernel; python-first precedence;
  R cells are non-terminal.
- `test_tools.py`: registry without bash; ```tool bash → unknown tool; prompt text.
- `test_kernel.py`: kernel-local `host.bash` runs without dispatcher RPC.
- Updated deliberately: `len(REGISTRY)` floor, bash precheck test imports,
  `test_orchestration_skills` `r_kernel` assertion, egress-bash tests.

## Non-goals (v1)

Native wire tool-calls; R host_call/`host` object; R provenance/figure devices;
REPL R cells in the read-only notebook; re-homing `_m_env_setup`.
