# Architecture Note

This note defines module ownership and stable boundaries inside `src`. Each package should expose a narrow role. Cross-cutting changes should be checked against these boundaries before code is moved or shared.

## Module Map

### `src/configs`

- **Owns:** Shared configuration values, the scheduler user-table column contract, radio scenario presets, default file paths, day-cycle session-generation presets, the default `SyntheticSessionGenerationConfig`, PA catalog loading, and PA helper behavior used by higher layers.
- **Consumes:** Repository-local default tables, source-file references, and config model types needed to build those configs and catalogs.
- **Does not own:** Derived deployment state, path-loss runtime values, candidate enumeration, or study orchestration.

### `src/models`

- **Owns:** Shared data models such as `RadioConfig`, `UserRequest`, `PAParams`, and derived runtime radio state such as `DeploymentParams`, path-loss helpers, and stable fingerprinting of resolved state.
- **Consumes:** Resolved config values, link distance, and other already-resolved values that need deterministic fingerprints.
- **Does not own:** Canonical scenario defaults, PA catalog loading behavior, or solver and study policy.

### `src/downlink_candidate_evaluation`

- **Owns:** Per-candidate rate, MCS requirement, SINR-chain, and power-feasibility models such as `CandidateRateModel` and `CandidatePowerModel`.
- **Consumes:** A resolved deployment, RRC envelope, candidate, PA model, and MCS table.
- **Does not own:** Search-space construction, whole-study caching policy, batching across users, or notebook-facing summaries.

### `src/single_user_solver`

- **Owns:** `PreparedSingleUserContext`, static search catalogs, discrete candidate enumeration, and the active candidate table for one `SingleUserRequest`.
- **Consumes:** A resolved radio config and search shape, a PA catalog, `models` deployment helpers, and `downlink_candidate_evaluation` models.
- **Does not own:** Canonical default scenario selection, batch user tables, multi-user reasoning, or notebook-facing study presentation.

### `src/single_user_parameter_space`

- **Owns:** The higher-level single-user study layer: request-table normalization, canonical default single-user engine state, `BatchUserParameterSpace`, and notebook-facing helpers such as `SingleUserScenario` and `SingleUserStudyResult`.
- **Consumes:** `configs` radio defaults and PA catalog builders, `models` value objects, and `single_user_solver`.
- **Does not own:** Per-candidate PHY equations, shared radio physics primitives, or joint TDMA scheduling across users.

### `src/multi_user_tdma_scheduler`

- **Owns:** `PreparedJointScheduleProblem`, repeated-window resolution, TDMA-space quantization and exact pruning, and the joint one-row-per-user search that returns `MultiUserTdmaSchedulerResult`.
- **Consumes:** Trusted `BatchUserParameterSpace` artifacts from `single_user_parameter_space` and PA switching policy from `models`.
- **Does not own:** Canonical single-user defaults, single-user candidate evaluation, or generation of user demand profiles.

### `src/day_cycle_simulation`

- **Owns:** Synthetic daily load and session generation, including 15-minute load expansion, the `SyntheticSessionGenerationConfig` model, and `SyntheticSession`.
- **Consumes:** Hourly or per-bin load tables and shared day-cycle preset catalogs from `configs`.
- **Does not own:** Radio configuration, link-budget physics, single-user solving, or multi-user schedule search.

### `src/day_run`

- **Owns:** The top-level day simulation workflow: CLI-adjacent run config resolution, day demand assembly, per-bin fan-out, and the authoritative day-run JSON export.
- **Consumes:** Shared defaults from `configs`, shared run contracts from `models`, demand generation from `day_cycle_simulation`, trusted single-user batch artifacts from `single_user_parameter_space`, and the joint TDMA scheduler in `multi_user_tdma_scheduler`.
- **Does not own:** User-table sanitization, PHY evaluation, scheduler internals, shared console formatting, or generic logging setup.

### `src/run_reporting.py`

- **Owns:** Shared logger setup, worker log inheritance, active-bin log scope, and the aligned console message format used by thin run entry points.
- **Consumes:** Run-level status updates and lean result contracts from the orchestration layer.
- **Does not own:** Day-run execution, export-schema assembly, solver-space construction, or scheduler search logic.

## Main Dependency Flow

1. `src/configs` defines the shared radio assumptions, the scheduler user-table column contract, day-cycle presets, PA source paths, and PA catalog behavior.
2. `src/models` provides the shared value objects and the derived deployment and fingerprint helpers consumed by the solver layers.
3. `src/single_user_solver` combines `configs`, `models`, and `downlink_candidate_evaluation` to build one prepared single-user context and enumerate candidate rows.
4. `src/single_user_parameter_space` chooses the canonical shared single-user engine state from `configs`, batches many user requests, and produces trusted per-user parameter-space artifacts.
5. `src/multi_user_tdma_scheduler` consumes those batch artifacts, resolves the repeated slot window, and solves the joint schedule.
6. `src/day_cycle_simulation` is an upstream demand generator that can feed scheduler-ready user tables into the single-user and multi-user workflow.
7. `src/day_run` orchestrates one full synthetic day by composing `day_cycle_simulation`, `single_user_parameter_space`, and `multi_user_tdma_scheduler`, while `src/run_reporting.py` owns the shared console reporting used by that run layer.
## Boundary Checks

- If logic evaluates one resolved candidate's rate, SINR, or power, it belongs in `downlink_candidate_evaluation`.
- If logic builds or searches one user's discrete candidate space from a prepared context, it belongs in `single_user_solver`.
- If logic defines the shared scheduler user-table column contract, it belongs in `configs`.
- If logic chooses default single-user settings, batches many requests, or prepares notebook-facing single-user studies, it belongs in `single_user_parameter_space`.
- If logic reasons jointly across users under a shared slot budget, it belongs in `multi_user_tdma_scheduler`.
- If logic builds shared default configs or PA catalogs, it belongs in `configs`.
- If logic builds daily load curves or synthetic session demand, it belongs in `day_cycle_simulation`.
- If logic orchestrates one full synthetic day run across packages, it belongs in `day_run`.
- If logic only formats or configures shared run logging, it belongs in `run_reporting.py`.
- Shared config values and PA behavior belong in `configs`; shared models and derived radio physics belong in `models`.
