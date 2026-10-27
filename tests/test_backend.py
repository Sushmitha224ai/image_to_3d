import asyncio
import io

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from backend import app as backend


@pytest.mark.parametrize("startup_error", [
    FileNotFoundError("trained checkpoints are missing"),
    ValueError("PCA-Ridge checkpoint is not fitted"),
])
def test_lifespan_keeps_saved_demo_available_when_inference_is_unavailable(
    monkeypatch, startup_error
):
    def fail_pipeline_initialization(config_path):
        raise startup_error

    monkeypatch.setattr(backend, "ForensicInferencePipeline", fail_pipeline_initialization)

    async def check_health():
        async with backend.lifespan(backend.app):
            return backend.health()

    health = asyncio.run(check_health())

    assert health["status"] == "ok"
    assert health["pipeline_ready"] is False
    assert health["inference_error"] == str(startup_error)
    assert health["device"] is None


def test_invalid_image_upload_returns_400_and_removes_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "project_dir", str(tmp_path))
    monkeypatch.setattr(backend, "pipeline", object())
    upload = UploadFile(filename="sketch.png", file=io.BytesIO(b"not an image"))

    with pytest.raises(HTTPException) as error:
        asyncio.run(backend.process_sketch(upload))

    assert error.value.status_code == 400
    temp_upload_dir = tmp_path / "data" / "temp_uploads"
    assert list(temp_upload_dir.iterdir()) == []
