# Backend extension guide

OpenAI4S has two action planes and one rule decides where new behaviour belongs:

- Use a native JSON `Tool` for orchestration, permissions, external services,
  metadata, or a human-approval boundary.
- Use Python/R Code-as-Action for computation, exploration, analysis,
  simulation, and long-running scientific execution.
- Successful task completion always remains
  `host.submit_output(...)` from a Python cell. Never add a native completion
  tool or infer completion from prose/tool results.

## Dependency map

```text
provider wire -> AgentEngine -> action router
                                  |          |
                                  |          +-> Python/R kernel
                                  |                    |
                                  +-> Tool class       +-> synchronous host RPC
                                         |                         |
                                         +------ HostDispatcher ---+
                                                    |
                                   host service classes / repositories
                                                    |
                                           Store compatibility facade
```

`HostDispatcher` owns the shared policy envelope: wire decoding, permissions,
human approval, audit/replay, injection screening, and UI activity events.
Business behaviour belongs in a tool or service class. `Store` remains the
compatible public facade and connection/migration owner; SQL behaviour belongs
in domain repositories sharing that connection and lock.

## Add a native control tool

Create one module under `openai4s/tools/`. The class must contain its schema,
security policy, and behaviour so a maintainer can understand the capability by
opening one file.

```python
from openai4s.tools.base import Tool


class CreateExperimentTool(Tool):
    name = "create_experiment"
    host_method = "create_experiment"
    description = "Create an approved scientific workflow record."
    parameters = {
        "properties": {
            "type": {"type": "string"},
        },
        "required": ["type"],
    }
    read_only = False
    requires_approval = True
    permission_target_key = "type"

    def execute(self, context, arguments: dict) -> dict:
        return context.create_experiment(arguments["type"])
```

Then add the class—not a pre-created instance—to `TOOL_TYPES` in
`openai4s/tools/registry.py`. The registry is the only built-in composition
point and creates the runtime instances in a deterministic order.

Rules enforced by registration:

- `bash` and `submit_output` can never be native tools;
- tool names must be portable across supported providers;
- network tools must declare untrusted-result screening;
- approval is required unless the class explicitly proves a safe read-only
  boundary;
- model-originated calls enter through `Tool.invoke()` and the dispatcher;
  application code must not call `execute()` as a policy bypass.

Add direct tests for the class behaviour and policy metadata, plus an engine
test for the provider-neutral call/result group when the wire contract changes.

## Add an in-kernel `host.*` capability

The worker-facing signature belongs in `openai4s/sdk/`. A cohesive namespace
such as compute should have its own module; `sdk/host.py` composes and
compatibly re-exports it.

The host-side implementation belongs in a class under `openai4s/host/`:

```python
class ExperimentService:
    def __init__(self, store_provider):
        self._store_provider = store_provider

    def create(self, spec: dict) -> dict:
        store = self._store_provider()
        return store.create_experiment(**spec)
```

Construct the service once in `HostDispatcher.__init__`, using small provider
callbacks when session state can be replaced at runtime. Keep the existing
`_m_<method>` method only as a thin compatibility adapter:

```python
def _m_create_experiment(self, spec: dict) -> dict:
    return self._experiment_service.create(spec)
```

Return `{"error": message}` only for the established soft-fail contract.
Uncaught exceptions are converted at the kernel protocol boundary. Do not
duplicate permission, audit, replay, or injection policy inside the service;
all calls already cross the dispatcher envelope.

## Add persisted data

Create a focused repository under `openai4s/storage/`. Repositories receive the
existing SQLite connection, the existing `RLock`, and a clock callback. They do
not open another connection for application writes.

```python
class ExperimentRepository:
    def __init__(self, connection, lock, *, clock_ms):
        self._connection = connection
        self._lock = lock
        self._clock_ms = clock_ms

    def get(self, experiment_id: str) -> dict | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        return dict(row) if row else None
```

`Store` owns schema creation and migrations, constructs the repository with its
shared connection/lock, and exposes a thin forwarding method. Composite writes
that span several aggregates must keep their existing single-lock and commit
boundary. When old code dynamically called another `Store` method, inject a
late-bound lambda rather than freezing a bound method and silently breaking
subclass/monkeypatch compatibility.

Repository tests should lock down SQL-visible results, commit/rollback
boundaries, ordering, timestamp evaluation, JSON fallback, and legacy error
shapes. Default tests remain offline.

## Add Web session behaviour

HTTP and WebSocket code is an adapter. Stateful behaviour belongs in a service
under `openai4s/server/` with narrow protocols/callbacks for persistence,
kernel lifecycle, event broadcast, and configuration. `SessionRunner` may keep
a private forwarding method when tests or integrations depend on it, but the
algorithm should be visible in the service module.

Preserve event payload keys and order-sensitive lifecycle rules. Changes to
kernel execution, host RPC, artifact capture, review, streaming, or resume need
both focused tests and a real browser run against `./start.sh`.

## Definition of done

For every backend extension or extraction:

1. The class file contains the behaviour; the registry/dispatcher/facade only
   composes or forwards.
2. Core imports remain standard-library-only.
3. Public SDK, `host.*`, CLI, REST/WebSocket, SQLite, and saved-session contracts
   remain compatible or receive an explicit migration.
4. `host.submit_output()` remains the sole successful terminal signal.
5. Run focused tests, the full offline suite, and the browser flow when session,
   kernel, RPC, artifact, or UI behaviour is involved.
6. Commit one cohesive change at a time.

Avoid module-level tool singletons, duplicate agent loops, host-side shell
execution, independent repository connections, provider response types leaking
into `AgentEngine`, and scientific computation disguised as JSON tools.
