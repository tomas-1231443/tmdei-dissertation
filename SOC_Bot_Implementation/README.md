# 🛡️ SOC Bot – Security Alert Triage Automation

## Overview

SOC Bot is an intelligent, ML-driven triage assistant for Security Operations Centers (SOCs). It combines traditional machine learning (Random Forest) with reinforcement learning (Stable Baselines3 PPO) to classify, prioritize, and flag incoming security alerts. It is designed to:

* Automate alert classification (priority + taxonomy)
* Predict if an alert is a false positive
* Learn continuously from analyst feedback
* Integrate seamlessly with QRadar, SOAR, and similar platforms
* Run with or without GPU acceleration (via CUDA)

This project supports both batch learning from historical data (Excel) and real-time inference through a REST API.

---

## 📁 Project Structure

```
my_soc_bot/
├── data/                      # Sample historical alert data
├── docs/                      # Architecture diagrams, WBS
├── logs/                      # Output logs for debug/audit
├── src/                       # All source code modules
│   ├── main.py                # CLI + FastAPI app
│   ├── logger.py              # Centralized logging
│   ├── config.py              # Global configuration and device selection
│   ├── preprocessing/         # Preprocessing logic for alerts
│   ├── models/                # ML + RL model loading/training
│   └── realtime/              # Real-time ingestion and reinforcement feedback
├── tests/                     # Unit/integration tests
├── requirements.txt           # Python dependencies
├── Dockerfile                 # CUDA-compatible container image
├── docker-compose.yml         # Multi-container setup with profiles
├── README.md                  # You're here.
├── DEPLOY.md                  # Deployment instructions (Docker, GPU)
└── API.md                     # REST API endpoint documentation
```

---

## ⚙️ Key Features

* 📊 Hybrid ML architecture: Random Forest + RL (PPO)
* 🧠 Learns from analyst feedback via Celery background training tasks
* 🌐 Real-time inference via FastAPI
* 🛠️ Fully containerized: `bot`, `celery`, `redis`, `flower`
* 🔥 GPU acceleration via CUDA (optional)
* 📈 Task queue monitoring via Flower
* 📦 Redis-backed statistics tracking for model accuracy
* 📤 API-based export of training statistics to CSV
* 🧪 Testable, modular, production-ready architecture

---

## 🚀 Getting Started

### 🐍 1. Install Python dependencies (local dev)

```bash
pip install -r requirements.txt
```

### ▶️ 2. Run the bot locally (CPU mode)

```bash
python -m src.main --excel-path data/sample_alerts.xlsx --verbose
```

Optional arguments:

* `--retrain`: Retrain the RF model
* `--cuda`: Enable GPU support (if available)
* `--model-version N`: Load a specific model version
* `--port 8000`: Set custom API port

---

## 🧠 How It Works

1. Loads a trained Random Forest model from disk (`/src/models/V*/`).
2. Loads a PPO RL agent (`rl_agent.zip`) to refine predictions.
3. Receives alerts via the `/alerts/final` API or Excel batch.
4. Normalizes and vectorizes alerts (SBERT + critical asset flag).
5. Combines ML and RL outputs to classify:

   * Priority (P1–P4)
   * Taxonomy (e.g., malware, C2, phishing)
   * Is false positive?
6. Analyst feedback is sent to `/alerts/feedback`, which enqueues a Celery task that:

   * Re-evaluates the decision
   * Trains the RL agent for a few steps
   * Logs accuracy stats to Redis

---

## 🔗 REST API

For complete details on endpoints, parameters, payloads, and responses, see:

📄 [`API.md`](API.md)

Key endpoints include:

* `POST /alerts/final`: Final prediction using RF + RL
* `POST /alerts/feedback`: Submit analyst feedback
* `GET /health`: Health check
* `GET /queue_length`: Celery queue size
* `GET /stats/export`: Export model accuracy stats as CSV

---

## ⚙️ GPU & CUDA Support

This bot can optionally use GPU acceleration for:

* SBERT vectorization (via `sentence-transformers`)
* PPO reinforcement learning (via PyTorch)

To enable:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

   ```bash
   sudo apt install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
2. Start the container with `--cuda` flag and `runtime: nvidia` (see below).

---

## 🐳 Docker Deployment

Containerized via `docker-compose.yml` with optional CUDA support.

📄 See [`DEPLOY.md`](DEPLOY.md) for detailed steps.

### Run in CPU mode:

```bash
docker compose up --build
```

### Run in GPU mode (if available):

```bash
docker compose --profile gpu up --build
```

Services:

* `bot` / `bot-gpu`: API with/without CUDA
* `celery`: Handles feedback training
* `flower`: Dashboard on port `5555`
* `redis`: Message broker and stats store

---

## 📊 Monitoring & Metrics

* Flower UI available at: [http://localhost:5555](http://localhost:5555)
* Daily Redis stats by:

  * `taxonomy` and `priority`
  * correct/incorrect predictions
  * exportable as CSV

---

## 📦 Dependencies

* Python 3.10
* PyTorch (with optional CUDA)
* FastAPI, Uvicorn
* scikit-learn
* stable-baselines3
* sentence-transformers
* Celery + Redis
* Flower

See `requirements.txt` for full list.

---

## ✅ Feedback Loop

The RL agent is updated incrementally:

* Alerts classified with RF + RL
* Feedback is submitted via API
* A Celery task evaluates correctness and trains the PPO model
* Accuracy stats are logged by `priority` and `taxonomy`
* Can be analyzed daily for improvement tracking

---

## 🧪 Testing

Basic tests live under `tests/`. Run via:

```bash
pytest
```

---

## 📌 Notes

* All model versions are stored under `src/models/V*/`
* Latest RL agent is always saved as `rl_agent.zip`
* Volumes ensure shared model state between containers
* Logs written to `logs/` folder
* Supports graceful fallback from CUDA → CPU if GPU unavailable

---

## 📚 Related Docs

* 📄 [`API.md`](API.md) – REST API spec
* 📄 [`DEPLOY.md`](DEPLOY.md) – Docker + GPU deployment
---

Let me know if you'd like this in Markdown, PDF, or embedded into your repository automatically.
