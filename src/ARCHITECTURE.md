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

### `src/candidate_table_generation`

- **Owns:** The precomputed distance-binned single-user frontier table, the JSON save/load/load-or-build boundary for that table, the strict same-PA pruning policy used to store it, and the fixed-distance candidate-table build workflow.
- **Consumes:** Canonical single-user defaults from `configs`, shared models from `models`, and active candidate enumeration from `single_user_solver`.
- **Does not own:** User-table normalization, per-user rate lookup, notebook studies, or joint TDMA scheduling.

### `src/single_user_lookup`

- **Owns:** Scheduler-facing user-table normalization, per-user lookup into the precomputed distance-binned frontier, user-specific rate-feasible filtering, and the `BatchUserParameterSpace` handoff artifact.
- **Consumes:** Shared defaults and PA catalog loading from `configs`, shared request and PA models from `models`, and the precomputed frontier table from `candidate_table_generation`.
- **Does not own:** Per-candidate PHY equations, fixed-distance frontier generation, notebook studies, or joint TDMA scheduling across users.

### `src/multi_user_scheduler`

- **Owns:** Shared scheduler mode selection and the thin public dispatch contract that routes one trusted batch artifact to a concrete multi-user scheduler backend.
- **Consumes:** Shared scheduler enums from `models`, trusted `BatchUserParameterSpace` artifacts from `single_user_lookup`, and concrete backend packages such as `multi_user_tdma_scheduler`.
- **Does not own:** TDMA prepared-problem models, TDMA or OFDMA result adaptation details, TDMA pruning or search logic, or OFDMA solver internals.

### `src/multi_user_tdma_scheduler`

- **Owns:** `PreparedJointScheduleProblem`, single-frame TDMA-space quantization and exact pruning, the exact one-row-per-user TDMA search, and the TDMA package-level runner that returns the shared public scheduler result directly.
- **Consumes:** Trusted `BatchUserParameterSpace` artifacts from `single_user_lookup` and PA switching policy from `models`.
- **Does not own:** Canonical single-user defaults, single-user candidate evaluation, or generation of user demand profiles.

### `src/multi_user_ofdma_scheduler`

- **Owns:** `PreparedJointOfdmaProblem`, trusted-batch OFDMA slot-scheduling preparation, the greedy slot-level OFDMA solver, and the OFDMA package-level runner that adapts that backend result onto the shared public scheduler result.
- **Consumes:** Trusted `BatchUserParameterSpace` artifacts from `single_user_lookup`, shared PA models from `models`, and the repository's fixed radio geometry from `configs`.
- **Does not own:** Frame-level proxy rows, slot-PRB packing witnesses, notebook analytics, or the shared public scheduler dispatch contract.

### `src/day_cycle_simulation`

- **Owns:** Synthetic daily load and session generation, including 15-minute load expansion, the `SyntheticSessionGenerationConfig` model, and `SyntheticSession`.
- **Consumes:** Hourly or per-bin load tables and shared day-cycle preset catalogs from `configs`.
- **Does not own:** Radio configuration, link-budget physics, single-user solving, or multi-user schedule search.

### `src/day_run`

- **Owns:** The top-level day simulation workflow: CLI-adjacent run config resolution, day demand assembly, per-bin fan-out, and the authoritative day-run JSON export.
- **Consumes:** Shared defaults from `configs`, shared run contracts from `models`, demand generation from `day_cycle_simulation`, trusted single-user batch artifacts from `single_user_lookup`, and the shared scheduler dispatcher in `multi_user_scheduler`.
- **Does not own:** User-table sanitization, PHY evaluation, scheduler internals, shared console formatting, or generic logging setup.

### `src/run_reporting.py`

- **Owns:** Shared logger setup, worker log inheritance, active-bin log scope, and the aligned console message format used by thin run entry points.
- **Consumes:** Run-level status updates and lean result contracts from the orchestration layer.
- **Does not own:** Day-run execution, export-schema assembly, solver-space construction, or scheduler search logic.

### `src/thesis_analysis`

- **Owns:** Thesis-facing post-run analysis over completed experiment artifacts, including scheduler-comparison ZIP extraction, chunk CSV preprocessing, coverage and breakpoint summaries, display-table derivation, and final figure-generation support.
- **Consumes:** Completed result artifacts such as `data/scheduler_comparison_hpc_sweep.zip` and the stable CSV/JSON contracts emitted by experiment runners.
- **Does not own:** Campaign point generation, scheduler execution, HPC chunk dispatch, raw scheduler internals, notebook presentation prose, or raw image archives.

## Main Dependency Flow

1. `src/configs` defines the shared radio assumptions, the scheduler user-table column contract, day-cycle presets, PA source paths, and PA catalog behavior.
2. `src/models` provides the shared value objects and the derived deployment and fingerprint helpers consumed by the solver layers.
3. `src/single_user_solver` combines `configs`, `models`, and `downlink_candidate_evaluation` to build one prepared single-user context and enumerate candidate rows.
4. `src/candidate_table_generation` consumes that solver output once to build the strict-pruned distance-binned frontier table.
5. `src/single_user_lookup` loads the persisted frontier artifact through `candidate_table_generation`, snaps each user upward to the next supported distance bin, filters by the user's required rate, and packages the trusted batch artifact.
6. `src/multi_user_scheduler` owns the shared scheduler mode selection and dispatches the trusted batch artifact to the selected backend.
7. `src/multi_user_tdma_scheduler` converts those batch artifacts onto one shared frame slot lattice, solves the TDMA joint schedule, and returns the shared public scheduler result directly.
8. `src/multi_user_ofdma_scheduler` preserves the same trusted batch artifacts as slot-level OFDMA scheduler input, runs the greedy OFDMA slot scheduler, and adapts the OFDMA backend result onto the shared public scheduler result.
9. `src/day_cycle_simulation` is an upstream demand generator that can feed scheduler-ready user tables into the single-user and multi-user workflow.
10. `src/day_run` orchestrates one full synthetic day by composing `day_cycle_simulation`, `single_user_lookup`, and `multi_user_scheduler`, while `src/run_reporting.py` owns the shared console reporting used by that run layer.
11. `src/thesis_analysis` starts after experiments have produced stable artifacts; it rebuilds local analysis tables and thesis outputs without owning scheduler execution.
## Boundary Checks

- If logic evaluates one resolved candidate's rate, SINR, or power, it belongs in `downlink_candidate_evaluation`.
- If logic builds or searches one user's discrete candidate space from a prepared context, it belongs in `single_user_solver`.
- If logic defines the shared scheduler user-table column contract, it belongs in `configs`.
- If logic builds, saves, or loads the precomputed distance-binned frontier table, it belongs in `candidate_table_generation`.
- If logic normalizes scheduler-facing user tables, looks up per-user feasible spaces, or packages the trusted batch artifact, it belongs in `single_user_lookup`.
- If logic selects a concrete scheduler backend behind the shared public contract, it belongs in `multi_user_scheduler`.
- If logic reasons jointly across users under a shared slot budget or adapts the TDMA backend onto the shared public scheduler result, it belongs in `multi_user_tdma_scheduler`.
- If logic prepares trusted batch user spaces as slot-level OFDMA scheduler input, runs the OFDMA slot scheduler, or adapts the OFDMA backend onto the shared public scheduler result, it belongs in `multi_user_ofdma_scheduler`.
- If logic builds shared default configs or PA catalogs, it belongs in `configs`.
- If logic builds daily load curves or synthetic session demand, it belongs in `day_cycle_simulation`.
- If logic orchestrates one full synthetic day run across packages, it belongs in `day_run`.
- If logic only formats or configures shared run logging, it belongs in `run_reporting.py`.
- If logic reads completed thesis result artifacts and derives analysis tables, summaries, or figures, it belongs in `thesis_analysis`.
- If logic defines or runs scheduler-comparison campaign chunks, it belongs in the future experiment-runner module, not `thesis_analysis`.
- Shared config values and PA behavior belong in `configs`; shared models and derived radio physics belong in `models`.
