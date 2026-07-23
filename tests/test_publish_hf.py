import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentforge.publish_hf import HF_OWNER, publish_to_hub


class TestPublishToHub:
    def test_creates_repo_under_auro_rirum_namespace(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "adapter"
        local_dir.mkdir()
        (local_dir / "adapter_model.safetensors").write_text("fake")

        with patch("agentforge.publish_hf.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            repo_id = publish_to_hub(
                local_dir=str(local_dir), repo_name="test-model", base_model="google/gemma-4-12B-it"
            )

        assert repo_id == f"{HF_OWNER}/test-model"
        mock_api.create_repo.assert_called_once()
        assert mock_api.create_repo.call_args.kwargs["repo_id"] == f"{HF_OWNER}/test-model"

    def test_uploads_folder_with_expected_repo_id(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "adapter"
        local_dir.mkdir()

        with patch("agentforge.publish_hf.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            publish_to_hub(local_dir=str(local_dir), repo_name="my-adapter", base_model="google/gemma-4-12B-it")

        mock_api.upload_folder.assert_called_once()
        kwargs = mock_api.upload_folder.call_args.kwargs
        assert kwargs["repo_id"] == "auro-rirum/my-adapter"
        assert kwargs["folder_path"] == str(local_dir)

    def test_writes_model_card_before_upload(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "adapter"
        local_dir.mkdir()

        with patch("agentforge.publish_hf.HfApi") as mock_api_cls:
            mock_api_cls.return_value = MagicMock()
            publish_to_hub(local_dir=str(local_dir), repo_name="my-adapter", base_model="google/gemma-4-12B-it")

        card = (local_dir / "README.md").read_text()
        assert "auro-rirum/my-adapter" in card
        assert "google/gemma-4-12B-it" in card
        assert "license: apache-2.0" in card

    def test_metrics_embedded_when_provided(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "adapter"
        local_dir.mkdir()
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps({"multi_turn_base": 0.61}))

        with patch("agentforge.publish_hf.HfApi") as mock_api_cls:
            mock_api_cls.return_value = MagicMock()
            publish_to_hub(
                local_dir=str(local_dir),
                repo_name="my-adapter",
                base_model="google/gemma-4-12B-it",
                metrics_path=str(metrics_path),
            )

        card = (local_dir / "README.md").read_text()
        assert "multi_turn_base" in card
        assert "0.61" in card

    def test_no_metrics_path_omits_metrics_section(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "adapter"
        local_dir.mkdir()

        with patch("agentforge.publish_hf.HfApi") as mock_api_cls:
            mock_api_cls.return_value = MagicMock()
            publish_to_hub(local_dir=str(local_dir), repo_name="my-adapter", base_model="google/gemma-4-12B-it")

        card = (local_dir / "README.md").read_text()
        assert "## Metrics" not in card

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "adapter"
        local_dir.mkdir()
        with pytest.raises(ValueError, match="mode must be"):
            publish_to_hub(
                local_dir=str(local_dir),
                repo_name="my-adapter",
                base_model="google/gemma-4-12B-it",
                mode="not_a_real_mode",
            )

    def test_exist_ok_true_so_republishing_does_not_fail(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "adapter"
        local_dir.mkdir()
        with patch("agentforge.publish_hf.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            publish_to_hub(local_dir=str(local_dir), repo_name="my-adapter", base_model="google/gemma-4-12B-it")
        assert mock_api.create_repo.call_args.kwargs["exist_ok"] is True

    def test_private_flag_propagated(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "adapter"
        local_dir.mkdir()
        with patch("agentforge.publish_hf.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            publish_to_hub(
                local_dir=str(local_dir),
                repo_name="my-adapter",
                base_model="google/gemma-4-12B-it",
                private=True,
            )
        assert mock_api.create_repo.call_args.kwargs["private"] is True
