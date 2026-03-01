# Data Layout

This directory provides a stable top-level entrypoint to key data assets.

Current implementation is symlink-based to avoid breaking legacy paths in `case_study_1/` and `case_study_2/`.

## Case Study 1
- `datasets` -> `case_study_1/data_generation_and_il_stage/datasets`
- `scaling_data` -> `case_study_1/comparative_study/scaling_data`
- `test_parameters` -> `case_study_1/comparative_study/test_parameters`

## Case Study 2
- `il_stage_gbd_datasets` -> `case_study_2/il_stage/gbd_datasets`
- `comparative_gbd_datasets` -> `case_study_2/comparative_study/gbd/gbd_datasets`
- `weather_generation` -> `case_study_2/weather_data_generation`
- `rl_common_weather_data` -> `case_study_2/rl_stage/common/weather_data`
- `rl_common_initial_states` -> `case_study_2/rl_stage/common/initial_states`

To (re)create this layout:

```bash
python scripts/setup_standard_layout.py
```
