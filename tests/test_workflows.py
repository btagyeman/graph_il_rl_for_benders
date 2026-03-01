from graph_il_rl_for_benders.workflows import (
    SUPPORTED_CASE_STUDY_2_YEARS,
    get_case_study_2_yearly_script,
    get_workflow,
    list_workflows,
)


def test_known_workflows_present() -> None:
    names = {workflow.name for workflow in list_workflows()}
    assert "cs1_train_il" in names
    assert "cs1_train_rl" in names
    assert "cs2_train_il" in names
    assert "cs2_train_rl" in names
    assert "cs1_evaluate_comparative" in names
    assert "cs2_generate_evaluation_data" in names


def test_workflow_script_exists() -> None:
    workflow = get_workflow("cs1_train_il")
    assert workflow.script.exists()


def test_case_study_2_year_script_exists() -> None:
    year = SUPPORTED_CASE_STUDY_2_YEARS[0]
    script = get_case_study_2_yearly_script(year)
    assert script.exists()


def test_legacy_workflow_aliases_resolve() -> None:
    assert get_workflow("cs1_comparative_study").name == "cs1_evaluate_comparative"
    assert get_workflow("cs2_prepare_evaluation_data").name == "cs2_generate_evaluation_data"
    assert get_workflow("cs2_comparative_generate_graph_data").name == "cs2_generate_evaluation_data"
