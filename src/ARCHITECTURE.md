# Architecture Note

This note defines module ownership and stable boundaries inside `src`. Each package should expose a narrow role. Cross-cutting changes should be checked against these boundaries before code is moved or shared.

## Module Map

### `src/configs`

- **Owns:** Shared configuration values, the scheduler user-table column contract, radio scenario presets, default file paths, PA catalog loading, PA helper behavior used by higher layers, and config-owned scheduler solver policy such as K-MILP tolerances and limits.
- **Consumes:** Repository-local default tables, source-file references, and config model types needed to build those configs and catalogs.
- **Does not own:** Derived deployment state, path-loss runtime values, candidate enumeration, scheduler execution, or study orchestration.

### `src/models`

- **Owns:** Shared data models such as `RadioConfig`, `UserRequest`, `PAParams`, `PASwitchPolicy`, `SchedulerMode`, scheduler result contracts, and derived runtime radio state such as `DeploymentParams`, path-loss helpers, and stable fingerprinting of resolved state.
- **Consumes:** Resolved config values, link distance, and other already-resolved values that need deterministic fingerprints.
- **Does not own:** Canonical scenario defaults, PA catalog loading behavior, solver policy, or scheduler backend internals.

### `src/candidate_table`

- **Owns:** The stored distance-binned single-user frontier table, JSON save/load/load-or-build behavior, strict same-PA pruning, fixed-distance frontier generation, single-user candidate enumeration, per-candidate rate/SINR/power evaluation, scheduler-facing user-table normalization, per-user lookup, and the `BatchUserParameterSpace` handoff artifact.
- **Consumes:** Canonical single-user defaults and PA catalog loading from `configs`, shared request/radio/deployment models from `models`, and the repository-local stored candidate-table artifact under `outputs/`.
- **Does not own:** Multi-user scheduling, finite-buffer demand generation, day-run orchestration, campaign chunking, post-run analysis, notebook studies, or thesis figure presentation.

### `src/schedulers`

- **Owns:** The final public multi-user scheduler dispatch API, source-facing scheduler mode selection, the round-robin baseline, the K-MILP backend, scheduler-internal prepared-problem models, feasibility-bound checks, frame-utilization summaries, and private K1/K2 implementation layers inside K-MILP.
- **Consumes:** Trusted `BatchUserParameterSpace` artifacts from `candidate_table`, shared scheduler/result/PA contracts from `models`, radio defaults and K-MILP solver policy from `configs`, and run-reporting helpers for compact console logs.
- **Does not own:** Candidate-table generation or lookup, finite-frame demand/test-case generation, run orchestration, campaign chunking, post-run analysis, notebook presentation, or CLI parsing.

### `src/user_generation`

- **Owns:** Finite-frame user population generation from load, user count, distance, and optional rate-shaping inputs.
- **Consumes:** Scheduler-facing column contracts and lightweight generation config models.
- **Does not own:** Radio configuration, link-budget physics, single-user solving, or multi-user schedule search.

### `src/experiment_runner`

- **Owns:** The official experiment execution workflow, CLI-adjacent run config resolution, finite-frame user assembly, candidate-table lookup, scheduler dispatch, compact result reporting, scheduler-comparison point IDs, run ordering, exact-scenario chunk grouping, and manifest contracts for generated campaign outputs.
- **Consumes:** Shared defaults from `configs`, shared run contracts from `models`, demand generation from `user_generation`, trusted single-user batch artifacts from `candidate_table`, the shared scheduler dispatcher in `schedulers`, and stable source contracts such as shared PA policy names when needed for campaign definitions.
- **Does not own:** User-table sanitization, PHY evaluation, scheduler internals, post-run analysis, thesis figure generation, notebook presentation helpers, raw HPC XML submissions, or completed ZIP extraction.

### `src/run_reporting.py`

- **Owns:** Shared logger setup, worker log inheritance, active run-scope lookup, and the aligned console message format used by thin run entry points and scheduler diagnostics.
- **Consumes:** Lean run-level context from orchestration and scheduler diagnostics.
- **Does not own:** Run execution, export-schema assembly, solver-space construction, or scheduler search logic.

## Main Dependency Flow

1. `src/configs` defines the shared radio assumptions, the scheduler user-table column contract, PA source paths, PA catalog behavior, and scheduler solver policy.
2. `src/models` provides the shared value objects, scheduler mode enum, scheduler result contract, and derived deployment/fingerprint helpers consumed by the solver layers.
3. `src/candidate_table` combines `configs` and `models` to build one prepared single-user context, evaluate candidate rate/SINR/power, build or load the strict-pruned distance-binned frontier table, snap each user upward to the next supported distance bin, filter by the user's required rate, and package the trusted batch artifact.
4. `src/schedulers` owns the shared scheduler mode selection and dispatches the trusted batch artifact to `round_robin` or `k_milp`.
5. `src/schedulers.round_robin` runs the deterministic rolling-quantum OFDMA baseline and returns the shared public scheduler result directly.
6. `src/schedulers.k_milp` runs the final K-MILP backend. Its K1 compressed TDMA-plan HiGHS MILP and K2 restricted pattern-count MILP are internal implementation stages, not public scheduler modes.
7. `src/user_generation` is an upstream finite-frame demand generator that can feed scheduler-ready user tables into the single-user and multi-user workflow.
8. `src/experiment_runner` orchestrates one finite-frame case by composing `user_generation`, `candidate_table`, and `schedulers`; its `scheduler_comparison` submodule defines bounded thesis campaigns and the chunk/manifests they emit.
9. Top-level `post_processing` starts after experiments have produced stable artifacts; it rebuilds local analysis tables and thesis outputs without owning scheduler execution or living inside the model source tree.

## Boundary Checks

- If logic defines the shared scheduler user-table column contract, it belongs in `configs`.
- If logic evaluates one resolved candidate's rate, SINR, or power, builds or searches one user's discrete candidate space from a prepared context, builds/saves/loads the precomputed distance-binned frontier table, normalizes scheduler-facing user tables, looks up per-user feasible spaces, or packages the trusted batch artifact, it belongs in `candidate_table`.
- If logic selects a concrete scheduler backend behind the shared public contract, it belongs in `schedulers`.
- If logic reasons jointly across users under the final round-robin or K-MILP scheduler contracts, it belongs in the corresponding `schedulers` backend package.
- K1/K2 helper logic belongs inside `schedulers.k_milp`; it should not become separate public scheduler API.
- If logic builds shared default configs, PA catalogs, or scheduler solver policy, it belongs in `configs`.
- If logic builds finite-frame user populations from load, distance, and user-count inputs, it belongs in `user_generation`.
- If logic orchestrates one finite-frame experiment run or defines scheduler-comparison campaign chunks, it belongs in `experiment_runner`.
- If logic only formats or configures shared run logging, it belongs in `run_reporting.py`.
- If logic reads completed thesis result artifacts and derives analysis tables, summaries, or figures, it belongs in the top-level `post_processing` package.
- Scheduler-comparison campaign contracts belong in `experiment_runner`, not `post_processing`.
- Shared config values and PA behavior belong in `configs`; shared models and derived radio physics belong in `models`.
