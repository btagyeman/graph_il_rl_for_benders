# Modernization Plan

## Goal
Bring the research codebase to production-grade software engineering quality while preserving algorithmic behavior.

## Phase 1: Foundation (Completed)
- Add `src/` package layer.
- Add standardized CLI workflow runner.
- Add packaging (`pyproject.toml`), linting, formatting, tests, and CI.
- Add project-level README and contributing guide.

## Phase 2: Safe Refactor
- [Done] Convert main training/data scripts to import-safe `main()` entrypoints.
- [Done] Replace script-wide globals with structured configuration objects.
- [Done] Externalize tunable settings via JSON config files (`configs/` + `--config` support).
- [Done] Add strict schema validation for config payloads.

## Phase 3: Deduplication
- [Done] Replaced repeated year-specific entrypoint scripts in `case_study_2/data_generation_il_stage/<year>/generate_graph_data.py` with compatibility wrappers.
- [Done] Centralized shared generation logic in `case_study_2/data_generation_il_stage/shared/generate_graph_data_shared.py`.
- [Done] Deduplicated repeated `helper_functions` and `common` source trees across years using shared canonical modules with forwarding wrappers.

## Phase 4: Verification
- Add deterministic smoke/regression tests.
- Add baseline metric checks for key paper experiments.
- Add reproducibility metadata (seeds, versions, runtime environment).

## Phase 5: Standardized Asset Layout
- [Done] Add top-level `data/` and `artifacts/` compatibility layout.
- [Done] Add idempotent setup script (`scripts/setup_standard_layout.py`).
- [Done] Add regression test coverage for layout links and targets.

## Phase 6: Canonical Folder Organization (Completed)
- [Done] Added coherent canonical views for both case studies:
  - `pipelines/`, `shared/`, `data/`, and `models/` alias trees.
- [Done] Updated workflow resolution to prefer canonical paths with legacy-path fallback.
- [Done] Added/updated automated checks to validate alias links and targets.
- [Done] Removed redundant root-level aliases in `case_study_2` to reduce clutter.
- [Done] Mirrored the same de-cluttering approach for `case_study_1`.

Canonical layout reference (legacy folders remain supported):

```text
case_study_1/
|-- pipelines/{il,rl,comparative}
|-- shared/{optimization,agent}
|-- data/{datasets,scaling_data,test_parameters}
`-- models/{il_agent,comparative_gbd_agents}

case_study_2/
|-- pipelines/{il,rl,comparative,il_data_generation}
|-- shared/{environments,estimators,simulation,optimization}
|-- data/{graph_datasets/{il,comparative},weather,initial_states}
`-- models/{il_agent,lstm_weights}
```

## Phase 7: Naming And Cleanup (Completed)
- [Done] Renamed `case_study_2/rl_stage/train_agent.py` to `train_rl_agent.py` with a backward-compatible wrapper.
- [Done] Renamed typo modules `minimize_lagragrian.py` -> `minimize_lagrangian.py` and updated imports/forwarders.
- [Done] Removed high-confidence unused data files and unused Python modules identified via conservative static reference checks.

## Next Priorities
- Add deterministic regression tests for key paper workflows (beyond smoke tests).
- Add provenance metadata logging (config hash, git commit, environment snapshot) to training outputs.
- Add optional end-to-end workflow validation profiles for heavier experiments.
