from graph_il_rl_for_benders.cli import main


def test_cli_list_runs(capsys) -> None:
    exit_code = main(["list"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "cs1_train_il" in output
