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

jobs = {}
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

        wav, sr = load_audio(in_path)
        wav_in = wav.unsqueeze(0).to(device)
        ref = wav_in.mean(0)
        wav_in = (wav_in - ref.mean()) / ref.std()
        jobs[job_id]["progress"] = 40

        out = apply_model(model, wav_in, device=device, split=True, shifts=1, overlap=0.25)
        out = out * ref.std() + ref.mean()
        jobs[job_id]["progress"] = 85

        sources = model.sources
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
requirements.txt

fastapi
uvicorn[standard]
demucs
pydantic
Dockerfile

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
README.md — just notes for yourself; not required for deployment.

What to do:

Go to GitHub → create a new repo (e.g. melodycaid-demucs).
Create these 3 files in that repo with the exact content above (app.py, requirements.txt, Dockerfile), then commit.
Go to render.com → New + → Web Service → connect that repo → Runtime: Docker → add env var DEMUCS_AUTH_TOKEN = any random string → Create Web Service.
When live, copy Render's URL and set DEMUCS_SERVICE_URL + DEMUCS_AUTH_TOKEN in your app secrets to match.
Want me to walk through any of these steps in more detail?

16 hours ago


we will do this another time during the day... i am very interested but i need to sleep. not a machine here... we will get it done.... the part of the melody helper is not in the place i want it yet... i need a more coherent with the song possibilities... thanks..

Show more
16 hours ago


Revert
Totally understood — rest well. We'll pick up the Demucs deployment whenever you're ready.

When you're back, tell me more about what "more coherent with the song" means for the melody helper — e.g. should it follow the song's actual phrasing/breath points, loop cleanly with the accompaniment, or align tighter to the score notes? — and I'll rework it.

16 hours ago


d0778099aed196decef77dea2771604e

4 hours ago


Revert

Thought for less than a second
That looks like a job ID or token, but I'm not sure what you'd like me to do with it — could you clarify?

4 hours ago


I did what you asked me... github and then render and then

4 hours ago


Revert

Thought for 2s
Nice work! Last step — wire the service into your app:

In your app's Secrets settings, set:

DEMUCS_SERVICE_URL = your Render URL (e.g. https://melodycaid-demucs.onrender.com, no trailing slash)
DEMUCS_AUTH_TOKEN = the random string you put in Render's DEMUCS_AUTH_TOKEN env var (leave blank if you didn't set one on Render)
Quick health check — paste your Render URL into a browser as https://<your-url>/health; you should see {"status":"ok"}.

Then in the studio, upload a song and hit Separate stems — first run takes a minute or two on CPU (Render free tier may need ~30–60s to wake if it slept).

Want to paste your Render URL here and I'll verify it's reachable before you try a real song?

4 hours ago


it is asking me to pay to deploy the webservice

4 hours ago


Revert
That's Render's new policy — free web services now require a payment method on file (they won't charge for the free tier, but a card is required). You have a few options:

Option A — Add a card to Render (still free) Render's "Free" instance type stays $0; the card is just to verify identity. If you're okay with that, add it and proceed — no charge unless you upgrade.

Option B — Use a fully-free alternative

Fly.io — genuinely free tier, no card needed for the trial:

Install the flyctl CLI: brew install flyctl (Mac) or see fly.io/docs for Windows.
In a terminal, in the demucs-service folder:
fly launch        # accept defaults, it detects the Dockerfile
fly deploy
Set the auth token: fly secrets set DEMUCS_AUTH_TOKEN=<your random string>
Copy the resulting URL (e.g. https://melodycaid-demucs.fly.dev).
Hugging Face Spaces (free, no card, simplest) — HF Spaces supports Docker and is great for ML services:

Create a free account at huggingface.co.
New Space → SDK: Docker → name it melodycaid-demucs.
Upload the 3 files (app.py, requirements.txt, Dockerfile) to the Space's repo (Files tab → Add file).
In Space Settings → Repository secrets, add DEMUCS_AUTH_TOKEN.
The Space URL (e.g. https://<your-name>-melodycaid-demucs.hf.space) is your DEMUCS_SERVICE_URL.
Hugging Face Spaces is probably the least friction if you want zero card involvement. Want me to tailor the steps for whichever you pick?

4 hours ago


check if it is working

4 hours ago


Revert

Thought for less than a second
I'll check if your Demucs service is reachable.

Fetched
melodycaid.onrender.com
Format: markdown

Thought for 2s
Your service URL (https://melodycaid.onrender.com) is returning a 502 — it's not live yet. Common causes:

Still building/deploying — Render's first build downloads torch (~5–8 min). Check your Render dashboard → the service's Logs and Events tabs; look for "Deploy live" or any error.
Crashed on startup — if logs show an import error (e.g. demucs or torchaudio missing), paste the error here and I'll fix the Dockerfile.
Sleeping (free tier) — if it deployed successfully but has been idle 15+ min, the first request wakes it (~30–60 s). Try hitting https://melodycaid.onrender.com/health again in your browser after a minute.
Can you open the Render dashboard → your service → Logs, and tell me what the last lines say (or paste them)? That'll tell us whether it's still building, crashed, or just sleeping.

4 hours ago


image.png
esto dice

4 hours ago


Revert

Thought for 6s
The build failed (not a runtime crash) — I need the actual error to fix it. On Render, open that same service → Logs tab (top, next to Events) → scroll to the red error lines near the bottom, and paste the last ~15 lines here.

Most likely it's running out of memory while installing torch (Render's free build tier is tight). Once I see the log I'll give you an exact fix — usually it's a one-line Dockerfile tweak.

4 hours ago


image.png
Aqui

4 hours ago


Revert
Found it — the Dockerfile on GitHub is empty (38 bytes, "file with no instructions"). When you created the file on GitHub it didn't paste correctly.

Fix: Go to your GitHub repo → open Dockerfile → click the ✏️ pencil icon → delete everything, paste this exact content, then Commit changes:

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
Also double-check app.py and requirements.txt aren't empty either (open each on GitHub and verify they have the content I sent earlier). Once all three are filled in and committed, Render will auto-redeploy — watch the Logs tab for "Deploy live".

4 hours ago


try again

4 hours ago


Revert
Fetched
melodycaid.onrender.com
Format: markdown
Still 502 — not live yet. Check the Render Logs tab again: if the build is still running (you'll see pip installing torch), just wait ~5–8 min; if it failed again, paste the new error lines and I'll fix it.

4 hours ago


image.png
This is how it looks in github

4 hours ago


Revert

Thought for less than a second
The Dockerfile looks correct now — click the green "Commit changes..." button (bottom-right), then Render will auto-redeploy. Watch the Logs tab; first build takes ~5–8 min for torch to install. Paste any new error if it fails.

4 hours ago


image.png
?

4 hours ago


Revert

Thought for 1s
Still failing. I need the build logs to see why — go to the Logs tab (top nav), and paste the last ~15 red error lines. The most common cause on Render free tier is running out of memory while installing torch.

If it's an OOM/memory error, the quick fix is to add this near the top of the Dockerfile (right after FROM):

ENV PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu --no-deps
But paste the actual error first so I give you the right fix.

4 hours ago


image.png
aqui

4 hours ago


Revert

Thought for 1s
The files live in the demucs-service/ subfolder, but Render builds from the repo root — so it can't find requirements.txt.

Fix: On Render → your service → Settings → scroll to Build → set Docker Build Context Directory to demucs-service → Save → then Manual Deploy → Deploy latest commit.

4 hours ago


hay un error en

4 hours ago


Revert
¿Qué error y en dónde (Render logs, la app, o GitHub)? Pega el mensaje o sube la captura y te lo resuelvo.

4 hours ago


Github pero no lo veo... voy a estudiar tal vez es que no están todos los archivos de github llenos

4 hours ago


Revert

Thought for 2s
Exacto. Verifica en GitHub que estos 3 archivos dentro de la carpeta demucs-service/ tengan contenido (no estén vacíos):

Dockerfile ✅ (ya está bien)
app.py — debe tener el código FastAPI (~120 líneas)
requirements.txt — debe tener: fastapi, uvicorn, demucs, pydantic
Abre cada uno en GitHub (Stratioti/Melodycaid → demucs-service/ → clic en el archivo). Si alguno está vacío, dímelo y te paso el contenido exacto para pegarlo.

4 hours ago


app.py... me das el codigo?

a few seconds ago


Revert

Thought for less than a second
Read
demucs service/app

Aquí está el código completo de app.py. Ábrelo en GitHub (demucs-service/app.py → ✏️), borra todo y pega esto:

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
