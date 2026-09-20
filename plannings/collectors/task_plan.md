# Collectors Event PubSub Implementation Plan

## Goal

Incrementally implement the approved collectors event-pubsub design while preserving existing Gateway, Framework, Router, and trajectory contracts. Each phase must form a reviewable vertical slice; Direct and Global transports remain isolated from the local path.

## Current Phase

### Phase 5: Final Migration and Cleanup

- [x] Add prompt-level aggregation for empty and failed episodes
- [x] Validate mixed-version behavior, failure recovery, and performance budgets
- [x] Audit compatibility paths; retain those whose removal criteria are not met
- **Status:** complete

## Completed Phases

### Phase 4: Router and Admission Consumers

- [x] Freeze Router ownership modes, projection parity, admission observation, and rollback contracts
- [x] Add Router state projection behind explicit legacy, shadow, and projector modes
- [x] Add admission observations without moving control ownership prematurely
- [x] Remove the compatibility writer in the Sticky ownership slice when projector mode owns commits
- [x] Verify shadow isolation, projector ownership, fail-closed behavior, parity, and disabled-path compatibility
- [x] Run focused Phase 4 tests, Phase 1-3 regressions, formatting, lint, compile, and scope review
- **Status:** complete

### Phase 3: Global Telemetry

- [x] Implement the bounded Global telemetry pipeline and isolated Reporter path
- **Status:** complete

### Phase 2: Direct State Sync

- [x] Implement recoverable Direct state sync and the Router session shadow
- **Status:** complete

### Phase 1: Minimal Local Vertical Slice

- [x] Audit the existing uncommitted local-event and Gateway integration against the Phase 1 contract
- [x] Complete the minimal `uni_agent.events` envelope, context, publisher, subscription, and local bus APIs
- [x] Keep one event runtime owned by `_GatewayActor`; inject only publisher/context dependencies into `GatewaySession`
- [x] Preserve list-returning Gateway finalization through an internal `SessionFinalizationResult`
- [x] Project `GenerationFinished` into a bounded, explicit-completeness metrics fragment
- [x] Preserve the fragment through Framework trajectory selection and postprocessing
- [x] Verify immutability, context isolation, unsubscribe, subscriber isolation, and incomplete-fragment behavior
- [x] Verify a real Gateway generation reaches the retained Framework trajectory without changing the disabled/legacy path
- **Status:** complete

### Phase 0: Baseline and Contract Freeze

- [x] Freeze compatibility, identity, event, fragment, transport, ownership, and configuration contracts
- [x] Record focused compatibility and performance baselines
- [x] Document the pinned veRL continuation-token incompatibility separately from collector work
- **Status:** complete_with_known_dependency_blocker

## Phase 1 Exit Criteria

- A Gateway generation emits local facts and produces a bounded fragment attached to the final retained Framework trajectory.
- Existing public Gateway finalization still returns `list[Trajectory]`.
- Missing observations remain explicit through `complete` and `incomplete_reasons`; they are never converted to zero.
- The disabled/legacy path adds no Actor, RPC, thread, or output change.
- No Direct or Global transport abstraction is introduced in this phase.
- Focused tests, relevant compatibility tests, formatting, lint, and compile checks pass, except for independently documented baseline/environment defects.

## Phase 2 Exit Criteria

- Direct transport uses versioned DTOs, per-source stream epochs, contiguous applied cursors, bounded queues, bounded retries, and authoritative snapshot recovery.
- A remote Endpoint validates and delivers only to the named local state-sync subscription; it never republishes.
- Gateway session lifecycle is projected into a Router shadow view without writing Router routing, inflight, sticky, or KV state.
- Direct remains disabled by default and creates no bridge thread or RPC when disabled.
- Global telemetry types, queues, ACKs, and Reporters remain outside Phase 2.
- Focused Direct, cross-Ray, Phase 1 compatibility, formatting, lint, compile, and diff checks pass, subject to the recorded veRL dependency gaps and noisy historical Router P99 comparison.

## Phase 3 Exit Criteria

- Global transport uses versioned telemetry batches, bounded source/broker/subscriber queues, bounded pending calls, and explicit accepted ACKs.
- An accepted ACK means only that the next Global queue admitted the batch; it never claims Reporter persistence or state application.
- Remote Global delivery targets a named local subscription through `deliver()` and cannot re-enter forwarding.
- Reporter latency, failure, or saturation cannot block publishers or unrelated subscribers; loss and incomplete outcomes remain observable.
- The first production slice forwards existing Gateway observability facts without changing Direct state sync or Router/Admission ownership.
- Global remains disabled by default and creates no Actor, RPC, queue pump, or Reporter when disabled.
- Focused Global, Phase 1/2 compatibility, formatting, lint, compile, and diff checks pass, subject only to previously recorded dependency blockers.

## Phase 4 Exit Criteria

- Router mode is explicit: legacy allocates no Phase 4 runtime, shadow cannot write Router state, and projector is the only Sticky writer after cutover.
- Sticky put, replica invalidation, and cache clear stay inside one ownership boundary; a Projector commit failure blocks later route expansion.
- `RouteCommitted` and `ReplicaCapacityChanged` are emitted only after synchronous Router commits, while observer failures remain non-blocking and visible.
- Admission state is bounded and explicitly observe-only; it owns no capacity mutation, Grant, Hold, queue, or scheduling decision.
- Existing Inflight, KV, Metrics, Direct, and Global ownership remains unchanged, and the final focused/static regression suite passes.

## Phase 5 Exit Criteria

- Prompt summaries retain episode outcomes and available metrics even when an episode yields no trajectory or all sessions fail; missing observations remain explicit rather than becoming zero.
- Fragment merging is idempotent by fragment identity and revision, and mixed legacy/current results do not double-count metrics or drop failure denominators.
- Recovery and shutdown paths preserve bounded queues, explicit incompleteness, and the ownership boundaries established in Phases 1-4.
- Compatibility adapters are removed only where focused parity tests prove all in-repository callers have migrated; externally meaningful public return contracts remain stable.
- Focused compatibility, failure-recovery, cross-phase regression, performance, formatting, lint, compile, and diff checks pass, subject only to recorded baseline dependency blockers.

## Guardrails

- Implement vertical slices; avoid broad package-by-package scaffolding.
- Keep metric names inside Projectors and business code limited to facts.
- Prefer small immutable DTOs and explicit ownership over shared mutable state.
- Preserve user changes outside the collectors slice, including the existing `verl`, `working/`, and unrelated planning state.
- Comments explain non-obvious constraints only; names and types should carry ordinary intent.

## Errors Encountered

| Error | Resolution |
|---|---|
| The automatic active plan resolves to `.planning/2026-09-20-collector-event-pubsub-implementation`, while the user selected `plannings/collectors/` | Treat `plannings/collectors/` as the sole plan for this task and audit the existing code as candidate Phase 1 work |
| The pinned veRL snapshot lacks `merge_context_tokens()` used by continuation code | Keep it as an independent baseline blocker; do not modify or restore the user-modified submodule during Phase 1 |
| Initial local publish P99 exceeded the frozen Phase 1 budgets | Removed avoidable hot-path allocations without changing event semantics; five follow-up batches passed both budgets |
| The user-modified veRL tree and the existing read-only export each miss different modules required by broad Ray/continuation suites | Preserve `verl`; use compatible focused suites and the new dedicated Gateway-to-Ray Direct integration, and record the unavailable coverage |
| A broad regression exposed extra Gateway Actor constructor kwargs on the disabled path | Preserve the original constructor call exactly when Direct is disabled; pass Direct dependencies only in the enabled branch |
| Router P99 reruns exceeded the historical tail threshold while median improved | Record the variance; Phase 2 does not modify `acquire_server()` or `release_server()`, and all compatible Router behavior suites pass |
| Initial Phase 3 scoped lint found an import-order error in `uni_agent/events/__init__.py` | Reorder the new Global export block and rerun the scoped checks |
| Initial Global test lint found one overlong assertion | Split the assertion without changing coverage |
| Initial Global module format check reported one file | Apply the repository formatter and rerun the scoped checks |
| Initial telemetry runtime lint found one overlong status line | Split broker and Reporter awaits into separate statements |
| Integrated Phase 3 lint found one Gateway import-order difference | Apply the scoped repository import organizer and rerun checks |
| Combined Gateway Global/Direct suite exceeded the first tool yield and its final output was not retained | Let the process exit cleanly and rerun with resumable session capture |
| Combined Gateway suite hit the known missing `merge_context_tokens()` veRL dependency | Preserve the user-modified submodule and classify the failure with the existing baseline blocker |
| Global Ray integration emitted a blocking `ray.get` warning from its communication thread | Use ObjectRef futures in Global source and broker pumps so async actor event loops are not synchronously waited |
| Framework regression reproduced the two known stdout-vs-caplog failures | Deselect only those exact baseline nodes; all compatible Framework tests must pass |
| Full Gateway actor regression reproduced six veRL continuation failures and two stdout-vs-caplog failures | Keep the eight baseline failures separate; require every compatible test to pass |
| Non-draining Global source shutdown could clear the active batch before its pump retired it | Track active ownership explicitly and drop only the remaining queue after the in-flight call completes |
| End-to-end test compared non-atomic broker and Reporter status snapshots too early | Poll the full cross-Actor predicate; do not imply runtime status is a transaction |
| Event-family config wiring left one unused Gateway import | Remove it and rerun scoped lint |
| One resumable test wait was submitted with malformed tool syntax | Reissued the wait against the same live session and retained the successful final result |
| Phase 5 baseline collection could not import `verl.workers.rollout.replica` from the user-modified submodule | Preserve `verl` and rerun against the previously recorded read-only compatible veRL export |
| Phase 5 compatibility baseline reproduced the known Loguru stdout-vs-`caplog` failure | Classify the exact node as an existing baseline defect and require the other 31 compatibility tests plus new Phase 5 coverage to pass |
