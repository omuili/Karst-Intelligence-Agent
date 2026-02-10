# Karst Intelligence Agent

An autonomous AI‑powered web application for **sinkhole susceptibility mapping and early warning**, focused on the Winter Park, Florida karst district.  
The system combines an XGBoost susceptibility model with **Google Gemini 3** for validation, explanation, continuous monitoring, and alert drafting.

**Live demo:** [https://karst-intelligence-agent.onrender.com/](https://karst-intelligence-agent.onrender.com/)  
> Hosted on a free Render instance – after 15+ minutes of inactivity, the service hibernates and may take **30–60 seconds** to wake up before responding to the first request.

---

## 🌍 What this app does

- predicts **sinkhole susceptibility** over Winter Park, Florida – a known sinkhole hotspot in the Central Florida karst district.
- Shows a **heatmap overlay** of predicted susceptibility with known sinkholes plotted as red points.
- Runs a **Gemini‑powered agentic analysis** that:
  - Validates ML scan results,
  - Explains the risk in natural language,
  - Generates prioritized recommendations.
- Continuously **monitors ground movement** (e.g., GPS‑like vertical velocity) and keeps a **combined risk score** up to date.
- When thresholds are crossed, **Gemini drafts early‑warning alerts** suitable for emergency managers and local authorities.
- The monitoring loop is designed to run **for days, weeks, or months** unless the operator explicitly stops it.

---

## 📊 Data sources (including ground movement)

- **Florida Geological Survey** – historical sinkhole inventory and karst geology.  
- **USGS 3DEP DEM** – 10 m elevation model for terrain and depressions.  
- **National Hydrography Dataset (NHD)** – lakes and surface water features.  
- **Karst / limestone units** – proximity to karst geology in the Central Florida karst district.  
- **Ground movement (GPS‑style)** – vertical velocity and distance for a nearby GPS station (e.g. FLOL, Orlando), used in the combined risk score and early‑warning triggers.

---

## 🖼 Screenshots (from `images/`)

These images correspond to what you see in the live app:

- ![Dashboard](images/dashboard.png)  
  Full‑screen **dashboard**: Florida basemap, Winter Park AOI, susceptibility legend, scan controls, and status.

- ![Susceptibility heatmap](images/sinkhole_susceptibility_winter_park.png)  
  **Susceptibility heatmap**: XGBoost probabilities rendered as XYZ tiles, warmer colors = higher risk, with historical sinkholes as red points.

- ![Gemini risk assessment](images/gemini_risk_assessment.png)  
  **Gemini AI risk panel**: risk category (e.g. VERY HIGH), confidence, narrative reasoning, and 3–5 prioritized recommendations.

- ![Model explainability](images/model_explanability.png)  
  **Model explainability** view: feature importance and metrics for the XGBoost model.

- ![3D DEM visualization](images/3d_visualization.png)  
  **3D DEM visualization**: terrain around Winter Park to inspect depressions and drainage patterns.

- ![Early Warning Agent](images/early_warning_agent.png)  
  **Early Warning Agent**: continuous monitoring status, GPS‑style station data, combined risk score, and a Gemini‑drafted alert.

These screenshots reflect the same behavior as the hosted app and a local run.

---

## 🧠 How Gemini 3 is used

The Karst Intelligence Agent is deliberately **hybrid**: the ML model provides a fast risk surface; Gemini makes sense of it, checks it, and talks to humans.

### 1. Scan validation and self‑correction

After a scan, the backend aggregates ML metrics:

- Average susceptibility over the AOI,
- Counts of high‑risk tiles,
- Data coverage per source (DEM, imagery, sinkholes),
- Historical sinkhole statistics.

`GeminiMLValidator` sends this summary (and optional displacement context) to Gemini, which returns:

- A **risk category** (e.g. LOW / MEDIUM / HIGH / VERY HIGH),
- A **confidence level**,
- A **data‑quality assessment**,
- A list of **warnings and recommendations**.

The UI exposes this as **AI Risk Assessment**. Where Gemini flags data gaps or inconsistencies, the app surfaces that so the ML output is not blindly trusted – Gemini is effectively **validating and re‑interpreting** the model’s result.

### 2. Agentic analysis over imagery and context

The `SinkholeAnalysisAgent` runs a multi‑step Gemini 3 pipeline:

1. Fetch geologic and hydrologic context for the AOI.  
2. Fetch satellite/aerial imagery tiles for Winter Park.  
3. Ask Gemini to detect karst‑related features (depressions, drainage anomalies, vegetation stress, lineaments) and output **structured risk factors**.  
4. Ask Gemini to integrate that with other data and produce a **final risk assessment**.  
5. Ask Gemini to generate **actionable recommendations** (e.g., where to prioritize geotechnical surveys or instrumentation).

This agent is what powers the **“Run AI Analysis”** experience in the UI.

### 3. Continuous monitoring and alert drafting

The Early Warning Agent keeps running after the scan:

- It combines:
  - **Static susceptibility** from the XGBoost model,  
  - **Movement risk** from ground‑displacement signals (GPS‑style metrics),  
  - Optional additional risk components.
- Computes a **combined risk score** at each monitoring step.
- When thresholds are crossed (Watch / Warning / Critical), it calls Gemini to:
  - Draft a concise alert message for authorities,  
  - Explain *why* the alert fired (evidence and context),  
  - Recommend prioritized actions.

Alerts appear in the UI and could be wired to email/SMS in a production deployment. The monitoring loop is intended to run **indefinitely** until explicitly stopped.

### 4. Model choice for the hosted demo

- In internal development (with Vertex AI), the project uses **`gemini-3-pro-preview`** for the deepest reasoning.  
- The **public Render demo** uses **`gemini-3-flash-preview`** via the Gemini API key path.  
  The AI Studio project behind the public key has **0 free tokens for `gemini-3-pro`**, so we switched to **Flash Preview** for the hosted app to stay within quota.  
  Flash still supports the agentic flows but with a more generous free‑tier profile.

The monitoring and early‑warning loop is designed to run **for days or weeks** – it keeps checking new data and generating alerts until the operator explicitly stops monitoring.

---

## 🚀 How to run the app locally (judges)

You can run the full system on a single machine.

### 1. Prerequisites

- **Python**: 3.10 (recommended)  
- **Git**  
- A **Gemini credential**, either:
  - A **Gemini API key (AI Studio)** – easiest, uses `gemini-3-flash-preview`, or  
  - A **Vertex AI** project + service‑account JSON – if you already use Vertex and want `gemini-3-pro-preview`.

### 2. Clone and create a virtual environment

```bash
git clone https://github.com/omuili/Karst-Intelligence-Agent.git
cd Karst-Intelligence-Agent

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure `.env`

Create a `.env` file in the repo root.

#### Option A – AI Studio key (Flash Preview, simplest)

```env
DEBUG=true
USE_VERTEX_AI=false
GEMINI_API_KEY=your_gemini_api_key_here
```

This path uses **`gemini-3-flash-preview`** for validation, analysis, and alerts – matching the hosted demo.

#### Option B – Vertex AI (Pro Preview)

If you have a GCP project with Vertex AI and a service‑account key:

```env
DEBUG=true
USE_VERTEX_AI=true
GOOGLE_CLOUD_PROJECT=your-gcp-project-id

# Either point to a JSON key file:
GOOGLE_APPLICATION_CREDENTIALS=/full/path/to/your-service-account.json

# Or use an inline JSON env:
# GOOGLE_APPLICATION_CREDENTIALS_JSON={ ...full JSON content... }
```

This path enables **`gemini-3-pro-preview`** locally (the model ID is set in `backend/config.py` under `GeminiConfig.MODEL_NAME`).

### 4. (Optional) Download real AOI data

To fully reproduce the Winter Park pipeline:

```bash
python -m backend.data.download_data
```

This populates `data/` with DEM and supporting datasets.

### 5. Run the backend + frontend

```bash
python -m backend.main
```

Then open:

```text
http://localhost:8000
```

### 6. Suggested walkthrough

1. Click **START AGENT** (Train + Agent).  
   - Watch the scanner sweep the Winter Park AOI and fill in the susceptibility overlay.  
2. After “Scan complete”, look at **Gemini 3 AI Analysis**:  
   - Risk category and confidence,  
   - Reasoning text,  
   - Recommendations.  
3. Click **Start Monitoring** in the Early Warning panel.  
   - The app computes a combined risk score on a schedule.  
   - When a trigger fires, Gemini drafts the full alert message, including key evidence and recommended next steps.  
4. Let monitoring run – it will continue autonomously until you stop it.

---

## 📜 License

MIT License – see `LICENSE`.

