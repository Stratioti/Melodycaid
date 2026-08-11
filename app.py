import os
import re
import uuid
import threading
import traceback
import urllib.request

from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import load_audio, save_audio

DATA_DIR = os.environ.get("DATA_DIR", "/data")
os.makedirs(DATA_DIR, exist_ok=True)

AUTH_TOKEN = os.environ.get("DEMUCS_AUTH_TOKEN", "")

app = FastAPI(title="Demucs Stem Separator")
app.mount("/files", StaticFiles(directory=DATA_DIR), name="files")

jobs = {}  # job_id -> {status, progress, error?, stems?}
lock = threading.Lock()

_model = None


def get_model_cached():
    global _model
    if _model is None:
        _model = get_model("htdemucs")
        _model.to("cpu").eval()
    return _model


class SplitReq(BaseModel):
    audio_url: str


def require_auth(authorization: str = ""):
    if AUTH_TOKEN and authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


def run_job(job_id, audio_url):
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 5
        job_dir = os.path.join(DATA_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        ext = ".wav"
        m = re.search(r"\.(mp3|wav|m4a|flac|ogg|aac)", audio_url, re.I)
        if m:
            ext = m.group(0)
        in_path = os.path.join(job_dir, "input" + ext)
        urllib.request.urlretrieve(audio_url, in_path)
        jobs[job_id]["progress"] = 15

        model = get_model_cached()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            model.to(device)
        jobs[job_id]["progress"] = 25

        wav, sr = load_audio(in_path)  # (channels, samples)
        wav_in = wav.unsqueeze(0).to(device)  # (1, channels, samples)
        ref = wav_in.mean(0)
        wav_in = (wav_in - ref.mean()) / ref.std()
        jobs[job_id]["progress"] = 40

        out = apply_model(model, wav_in, device=device, split=True, shifts=1, overlap=0.25)
        out = out * ref.std() + ref.mean()  # (1, stems, channels, samples)
        jobs[job_id]["progress"] = 85

        sources = model.sources  # ['drums','bass','other','vocals']
        vocals_idx = sources.index("vocals")
        accomp_idx = [i for i in range(len(sources)) if i != vocals_idx]

        vocals = out[0, vocals_idx].cpu()
        accomp = out[0, accomp_idx].sum(0).cpu()

        save_audio(vocals, os.path.join(job_dir, "vocals.wav"), sr)
        save_audio(accomp, os.path.join(job_dir, "accompaniment.wav"), sr)

        jobs[job_id]["stems"] = {
            "vocals": {"url": f"/files/{job_id}/vocals.wav"},
            "accompaniment": {"url": f"/files/{job_id}/accompaniment.wav"},
        }
        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = f"{e}\n{traceback.format_exc()}"


@app.post("/split")
async def split(req: SplitReq, _=Depends(require_auth)):
    job_id = uuid.uuid4().hex
    with lock:
        jobs[job_id] = {"status": "queued", "progress": 0}
    threading.Thread(target=run_job, args=(job_id, req.audio_url), daemon=True).start()
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/jobs/{job_id}")
async def job_status(job_id: str, _=Depends(require_auth)):
    with lock:
        j = jobs.get(job_id)
    if not j:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse(j)


@app.get("/health")
async def health():
    return {"status": "ok"}
