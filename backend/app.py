import os
import sys
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from inference.pipeline import ForensicInferencePipeline

pipeline = None
inference_error = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, inference_error
    print("[*] Starting FastAPI Server & Loading Forensic Pipeline...")
    config_path = os.path.join(project_dir, "configs", "config.yaml")
    try:
        pipeline = ForensicInferencePipeline(config_path=config_path)
        inference_error = None
        print("[OK] Forensic Pipeline initialized successfully.")
    except (FileNotFoundError, ValueError) as e:
        pipeline = None
        inference_error = str(e)
        print(f"[!] Live inference unavailable: {inference_error}")
    try:
        yield
    finally:
        pipeline = None

app = FastAPI(
    title="Adaptive Multi-Model Forensic Sketch-to-2D and 3D Face Reconstruction System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Frontend Files
frontend_dir = os.path.join(project_dir, "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

output_dir = os.path.join(project_dir, "output")
results_dir = os.path.join(project_dir, "results")
app.mount("/output", StaticFiles(directory=output_dir), name="output")
app.mount("/results", StaticFiles(directory=results_dir), name="results")

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Frontend index.html not found.</h2>", status_code=404)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "inference_error": inference_error,
        "device": pipeline.device if pipeline is not None else None
    }

@app.post("/api/process_sketch")
async def process_sketch(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=inference_error or "Live inference is unavailable because the trained pipeline is not initialized."
        )

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="Invalid image file format. Supported: PNG, JPG, JPEG, BMP, WEBP.")

    temp_path = None
    try:
        temp_dir = os.path.join(project_dir, "data", "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="upload_",
            suffix=suffix,
            dir=temp_dir,
            delete=False
        ) as buffer:
            temp_path = buffer.name
            shutil.copyfileobj(file.file, buffer)

        try:
            with Image.open(temp_path) as image:
                image.verify()
            with Image.open(temp_path) as image:
                sketch_pil = image.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from e

        res = pipeline.process_sketch(sketch_pil, output_basename="sketch_result")
        return JSONResponse(content=res)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Sketch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/download_3d/{file_type}")
def download_3d(file_type: str):
    file_type = file_type.lower()
    if file_type not in ["obj", "ply", "glb"]:
        raise HTTPException(status_code=400, detail="Invalid 3D format requested. Use obj, ply, or glb.")

    target_file = os.path.join(output_dir, "3d", f"sketch_result.{file_type}")
    if os.path.exists(target_file):
        return FileResponse(path=target_file, filename=f"forensic_3d_face.{file_type}", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Requested 3D face file not found. Process a sketch first.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("backend.app:app", host="127.0.0.1", port=port, reload=False)
