# 📦 SOC Bot – Docker Deployment Instructions

## 1. 📁 Prerequisites

Ensure the following are installed on the deployment machine:

* Docker Engine ≥ 20.10
* Docker Compose v2 (use `docker compose` not `docker-compose`)
* (Optional, for GPU support) NVIDIA GPU with drivers installed
* (Optional, for CUDA) NVIDIA Container Toolkit

---

## 2. 📂 Project Structure

The deployment is based on the following core files:

* `Dockerfile` – Builds the application image (CUDA-compatible).
* `docker-compose.yml` – Defines services: bot, celery, flower, redis.
* `requirements.txt` – Python dependencies.

---

## 3. 🚀 Building the Docker Image

From the root of the project:

```bash
docker compose build
```

This builds the image named `soc-bot-image` used by all containers.

---

## 4. ⚙️ Running in CPU-Only Mode (Default)

To start the application using CPU-only:

```bash
docker compose up --build
```

This launches the following services:

* `redis` – Redis broker
* `bot` – SOC bot API (`python -m src.main`)
* `celery` – Celery worker
* `flower` – Task dashboard on port `5555`

Accessible at:

* API: [http://localhost:8000/health](http://localhost:8000/health)
* Flower: [http://localhost:5555/](http://localhost:5555/)

---

## 5. ⚡ Running in GPU Mode (CUDA)

### 5.1. Install NVIDIA Container Toolkit

If GPU support is desired, install the NVIDIA runtime:

```bash
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify that the NVIDIA runtime is available:

```bash
docker info | grep -i runtime
```

Should include: `nvidia`

---

### 5.2. Launch GPU-Compatible Bot

Start with the `"gpu"` profile to run the bot in CUDA mode:

```bash
docker compose --profile gpu up --build
```

This uses:

* `bot-gpu` container: `python -m src.main --cuda`
* `runtime: nvidia`

ℹ️ The container will fall back to CPU if `torch.cuda.is_available()` is false.

---

## 6. 🗃️ Volumes

A named volume `shared-rl-agent` is used to share the RL model (`rl_agent.zip`) between:

* API (`bot` / `bot-gpu`)
* Celery worker (`celery`)

---

## 7. 📊 Monitoring (Flower)

Access task queue monitoring via:

```bash
http://localhost:5555/
```

Authentication is defined in Compose (e.g. `Tomas:2002`).

---

## 8. 📌 Notes

* **Only one** of `bot` or `bot-gpu` runs at a time (both expose port 8000).
* Ensure `--cuda` is only passed when `runtime: nvidia` is used.
* If CUDA is enabled but no GPU is found, the application logs a warning and uses CPU automatically.