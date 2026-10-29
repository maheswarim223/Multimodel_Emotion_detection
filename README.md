# EmotiSense — Unified Emotion Detection Dashboard

A production-ready, single-entry-point web application that integrates three independent emotion detection models into one beautiful, real-time UI.

---

## 🧠 Architecture

```
Mental Health/
├── backend/
│   ├── app.py              ← Flask API + SSE streaming
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx             ← Main application shell
│   │   ├── config.js           ← Modality config & API base
│   │   ├── index.css           ← Design system
│   │   └── components/
│   │       ├── ModalitySelector.jsx   ← 3-card modality picker
│   │       ├── PipelineProgress.jsx   ← Real-time log + progress
│   │       └── ResultsPanel.jsx       ← Metrics, charts, table
│   └── vite.config.js
├── start.bat               ← One-click Windows launcher
├── train.py                ← Text: original training script
├── train_text_emotion.py
├── test_text_emotion.py
├── voicedata/
│   ├── trainvoice.py
│   └── testvoice.py
└── image/
    ├── train.py
    └── test.py
```

---

## 🚀 Quick Start

### Option A — Double-click launcher
```
Double-click  start.bat
```
It will:
1. Install backend Python packages
2. Open Flask backend on **http://localhost:5050**
3. Open React frontend on **http://localhost:3000**

---

### Option B — Manual

**Backend**
```powershell
cd "Mental Health\backend"
pip install flask flask-cors pandas numpy scikit-learn nltk joblib librosa tensorflow
python app.py
```

**Frontend** (separate terminal)
```powershell
cd "Mental Health\frontend"
npm install
npm run dev
```

Open browser → **http://localhost:3000**

---

## 🎯 How to Use

1. **Select a modality** — click one of the three cards:
   - 💬 **Text** — NLP with TF-IDF + Logistic Regression
   - 🎙️ **Voice** — MFCC features + Random Forest
   - 😊 **Image** — CNN (TensorFlow/Keras)

2. **Click Run** — the full training + evaluation pipeline executes on the backend

3. **Watch real-time progress** — log feed + animated progress bar stream updates via SSE

4. **View results** — accuracy, precision, recall, F1 and per-class breakdown shown in the results panel

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/run/{modality}` | Start pipeline job, returns `job_id` |
| `GET`  | `/api/stream/{job_id}` | SSE stream of progress events |
| `GET`  | `/api/health` | Backend health check |

### SSE Event Types
```json
{ "type": "progress", "stage": "training", "message": "...", "percent": 65 }
{ "type": "result",   "accuracy": 87.4, "class_metrics": [...], ... }
{ "type": "error",    "message": "...", "traceback": "..." }
```

---

## 📦 Requirements

| Layer | Requirement |
|-------|------------|
| Python | 3.9+ |
| Node.js | 18+ |
| pip packages | flask, flask-cors, pandas, scikit-learn, nltk, librosa, tensorflow |
| npm | React 18 + Vite 5 |

---

## ⚡ Notes

- The **image (CNN) pipeline** can take 10–30 minutes depending on your GPU/CPU.  
  Early stopping is active (patience=5) so it won't unnecessarily run all 30 epochs.
- The **voice pipeline** extracts MFCC features from ~5,600 audio files — expect 5–15 minutes.
- The **text pipeline** is the fastest — typically 2–5 minutes.
- All models are saved back to their original paths after training.
