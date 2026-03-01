# Artifacts Layout

This directory provides a stable top-level entrypoint to trained models/agents and reusable model weights.

Current implementation is symlink-based to preserve backward compatibility with existing scripts.

## Trained Agents
- `trained_agents/case_study_1/comparative_gbd_agents` -> `case_study_1/comparative_study/gbd_agents`
- `trained_agents/case_study_1/rl_il_agent` -> `case_study_1/rl_stage/proposed_approach/il_agent`
- `trained_agents/case_study_2/rl_il_agent` -> `case_study_2/rl_stage/il_agent`
- `trained_agents/case_study_2/comparative_proposed_trained_agent` -> `case_study_2/comparative_study/proposed/trained_agent`

## Model Weights
- `model_weights/case_study_2_lstm` -> `case_study_2/trained_lstm_models/model_weights`

To (re)create this layout:

```bash
python scripts/setup_standard_layout.py
```
