# Deploy Karst Intelligence Agent on Render

One **Web Service** runs the FastAPI app (API + static frontend). No separate frontend service.

---

## 1. Push your code

Ensure the repo is on **GitHub** or **GitLab** and that Render can access it (public repo or connected account).

---

## 2. Create a Web Service on Render

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**.
2. Connect your repo and select the **sinkhole-scanner** (or your repo name).
3. Use these settings:

| Field | Value |
|--------|--------|
| **Name** | `karst-intelligence-agent` (or any name) |
| **Region** | Choose closest to you (e.g. Oregon) |
| **Branch** | `main` (or your default branch) |
| **Runtime** | **Python 3** |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |

Render sets `PORT` automatically; the app binds to it.

4. **Advanced** (optional):
   - **Python Version**: set to `3.10` or `3.11` if you add a `runtime.txt` (see below).

---

## 3. Environment variables

In the service → **Environment** tab, add:

| Key | Required | Notes |
|-----|----------|--------|
| `DEBUG` | Yes | Set to `false` (disables reload and uses production mode). |
| `GEMINI_API_KEY` | For AI | Your Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey). Use this for the simplest setup. |
| `USE_VERTEX_AI` | Optional | Set to `false` when using `GEMINI_API_KEY`. Omit or `true` only if you configure GCP (see below). |
| `GOOGLE_CLOUD_PROJECT` | If Vertex | Your GCP project ID (only if `USE_VERTEX_AI=true`). |
| `GOOGLE_CLOUD_REGION` | Optional | e.g. `us-central1` (default in code). |
| `PLANETARY_COMPUTER_API_KEY` | Optional | For Sentinel/satellite imagery. |
| `OPENTOPOGRAPHY_API_KEY` | Optional | For 3D terrain. |

**Recommended for Render:** Use **Gemini API key** (no GCP setup):

- `DEBUG=false`
- `GEMINI_API_KEY=<your-key>`
- `USE_VERTEX_AI=false` (or leave unset; default in code may be false depending on your config).

If you use **Vertex AI** on Render, you must provide GCP credentials (e.g. a **service account JSON key** in an env var and code that loads it); Application Default Credentials are not available in Render’s environment.

---

## 4. Deploy

Click **Create Web Service**. Render will:

1. Clone the repo  
2. Run `pip install -r requirements.txt`  
3. Start `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`  

After the build finishes, the app URL will be like:

`https://karst-intelligence-agent.onrender.com`

- **Frontend:** open that URL in a browser.  
- **API docs:** `https://<your-service>.onrender.com/docs`  
- **Health:** `https://<your-service>.onrender.com/health`  

---

## 5. Optional: Pin Python version

In the repo root, add **runtime.txt**:

```
python-3.10.11
```

Render will use that version. Adjust the minor version as needed (e.g. `3.11.9`).

---

## 6. Optional: Deploy from Blueprint (render.yaml)

If you added **render.yaml** in the repo root, you can use **Blueprint**:

1. **New** → **Blueprint**.
2. Connect the repo; Render will read **render.yaml** and create the Web Service with the same build/start commands and env var placeholders.

You still need to set **secret** env vars (e.g. `GEMINI_API_KEY`) in the Render dashboard.

---

## Troubleshooting

- **Build fails on a dependency**  
  - Geospatial stack (rasterio, GDAL, etc.) is supported on Render’s Linux image. If a version fails, try a slightly older version in **requirements.txt** or pin the Python version with **runtime.txt**.

- **App crashes or “Application failed to respond”**  
  - Ensure **Start Command** is exactly:  
    `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`  
  - Ensure **Root Directory** is left blank (so the repo root, where `backend/` lives, is the working directory).

- **Cold starts**  
  - Free-tier instances spin down after inactivity; the first request after idle can take 30–60 seconds. Paid instances stay warm.

- **Missing env vars**  
  - If AI or terrain features don’t work, check that the required env vars are set in the **Environment** tab and that no typos (e.g. `GEMINI_API_KEY` not `GEMINI_KEY`).
