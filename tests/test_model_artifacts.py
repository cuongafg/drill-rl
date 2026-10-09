import re

import pytest

from model_artifacts import (
    create_model_run_id,
    model_checkpoint_filename,
    model_run_directory,
)


def test_model_run_id_is_timestamped_and_unique():
    first = create_model_run_id()
    second = create_model_run_id()

    assert re.fullmatch(r"\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{8}", first)
    assert first != second


def test_model_run_directory_is_algorithm_scoped(tmp_path):
    path = model_run_directory(str(tmp_path), "PPO", "run-123")

    assert path == tmp_path / "ppo" / "run-123"


def test_checkpoint_filename_preserves_configured_name():
    assert model_checkpoint_filename("ppo_drill_baseline_200k") == (
        "ppo_drill_baseline_200k.zip"
    )


@pytest.mark.parametrize("model_name", ["", "../model", "a/model", "model.zip"])
def test_checkpoint_filename_rejects_unsafe_names(model_name):
    with pytest.raises(ValueError, match="Invalid model name"):
        model_checkpoint_filename(model_name)


@pytest.mark.parametrize("run_id", ["", ".", "..", "../outside", "run/name"])
def test_model_run_directory_rejects_path_like_ids(tmp_path, run_id):
    with pytest.raises(ValueError, match="Invalid model run ID"):
        model_run_directory(str(tmp_path), "ppo", run_id)
