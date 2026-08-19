# AI-Trace 🔍

> **High-Throughput Asynchronous Multimodal Digital Forensics & Content Authenticity Engine**

AI-Trace is a distributed, event-driven backend pipeline and real-time dashboard designed to detect AI-generated media and synthetic manipulations across image and text modalities. Rather than running heavy signal transformations and neural inference inside synchronous HTTP cycles, AI-Trace decouples ingestion, background task processing, and evidential decision fusion, visualized through a dynamic Next.js interface.

---

## 🏗️ System Architecture

```text
[ Next.js React Dashboard ]
│  • Async Polling Mechanism
│  • Dynamic Metric Rendering
│  • FastAPI Artifact Proxy
▼
1. HTTP POST Payload (Image/Text)
│
▼
┌──────────────────────────────────────┐
│       FastAPI API Orchestrator       │
│  • Schema Validation (Pydantic)      │
│  • pHash / Cryptographic Caching     │
│  • Static File Mounting Volume       │
└──────────────────┬───────────────────┘
│
2. Check / Queue Task
│
▼
┌──────────────────────────────────────┐
│     Redis Task Broker & Cache        │
└──────────┬────────────────┬──────────┘
│                │
3a. Dispatch     3b. Dispatch
│                │
▼                ▼
┌────────────────────┐   ┌────────────────────────┐
│ Image Worker Pool  │   │   Text Worker Pool     │
│ (Celery + OpenCV)  │   │  (Celery + HuggingFace)│
├────────────────────┤   ├────────────────────────┤
│ • 2D FFT Spectral  │   │ • Stylometric Variance │
│ • ELA Differential │   │ • Perplexity (PPL)     │
│ • Spatial Artifacts│   │ • DeBERTa Classifier   │
└──────────┬─────────┘   └──────────┬─────────────┘
│                        │
└───────────┬────────────┘
│ 4. Heterogeneous Sub-Scores
▼
┌──────────────────────────────────────┐
│       Evidential Fusion Engine       │
│  • Calibrated Logit Ensembles        │
│  • Dempster-Shafer Belief Mapping    │
│  • Dynamic Risk Tiering              │
└──────────────────┬───────────────────┘
│
5. Persist Verdict & Artifacts to Docker Volume

```

---

## ⚡ Key Features

* **Real-Time Polling Dashboard:** Next.js frontend with dynamic metric rendering that automatically adapts to image-specific visual heatmaps or text-specific NLP cards based on the backend response.
* **Non-Blocking Ingestion:** Instant `202 Accepted` responses with microsecond turnaround via FastAPI and Celery.
* **$O(1)$ Fast-Path Perceptual Caching:** Computes Perceptual Hash (`pHash`) for images and SHA-256 for text, bypassing worker computation on duplicate/near-identical inputs.
* **Frequency-Domain Signal Analysis:** 2D Fast Fourier Transform (FFT) and azimuthal radial power spectrum integration to detect high-frequency grid artifacts left by generative upsampling/transposed convolutions.
* **Spatial Compression Analysis:** Error Level Analysis (ELA) with scaled differential matrices ($15\times$) to highlight non-uniform local JPEG re-compression.
* **Statistical NLP Forensics:** Sentence-level burstiness calculation ($\frac{\sigma_L - \mu_L}{\sigma_L + \mu_L}$) and autoregressive log-likelihood perplexity mapping.
* **Mathematical Evidential Decision Fusion:** Fuses conflicting and heterogeneous signals into calibrated probabilities and Dempster-Shafer belief distributions ($\text{Belief}_{\text{AI}}$, $\text{Belief}_{\text{Authentic}}$, $\text{Uncertainty}$).
* **Hardware-Agnostic:** Optimized for CPU execution out-of-the-box with full multi-container Docker Compose support and shared volume mounting for static assets.

---

## 📁 Repository Structure

```text
AI-Trace/
├── frontend/                # Next.js React Dashboard
│   ├── src/components/      # Dropzone, ResultsDashboard, dynamic metric cards
│   ├── src/app/page.tsx     # Main application view & state management
│   └── next.config.mjs      # API & Artifact proxy routing to FastAPI
├── app/
│   ├── config.py            # System configuration & environment settings
│   ├── main.py              # FastAPI application gateway & endpoints
│   ├── schemas.py           # Pydantic data validation schemas
│   ├── fusion/
│   │   └── engine.py        # Dempster-Shafer & Logit Sigmoid Fusion Engine
│   ├── services/
│   │   ├── cache.py         # Redis hash caching & lookup logic
│   │   └── hasher.py        # pHash / dHash and text SHA-256 utilities
│   └── tasks/
│       ├── celery_app.py    # Celery broker & queue routing configuration
│       ├── vision_tasks.py  # 2D FFT, ELA, and image forensic workers
│       └── text_tasks.py    # Stylometrics, Burstiness, and PPL workers
├── test_pipeline.py         # End-to-end integration test & polling script
├── docker-compose.yml       # Multi-container orchestration (API, Redis, Worker)
└── README.md

```

---

## 🚀 Quick Start

### Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* Node.js & npm

### 1. Start the Backend Infrastructure

```bash
# Clone the repository
git clone https://github.com/your-username/AI-Trace.git
cd AI-Trace

# Build and start all services (API, Redis, Celery Workers)
docker compose up --build -d

```

### 2. Start the Next.js Dashboard

```bash
# Open a new terminal and navigate to the frontend directory
cd frontend

# Install dependencies and start the development server
npm install
npm run dev

```

Access the UI at `http://localhost:3000`.

---

## 📡 API Reference

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### 1. Submit Analysis Job

`POST /api/v1/forensics/analyze`

**Request (multipart/form-data):**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `modality` | string | Yes | `IMAGE` or `TEXT` |
| `file` | binary | Conditional | Image file (required if modality=IMAGE) |
| `text_content` | string | Conditional | Raw text payload (required if modality=TEXT) |

**Response (`202 Accepted`):**

```json
{
  "task_id": "8f3b2a19-9c32-4d21-a4ef-1200fa882a1b",
  "status": "QUEUED",
  "estimated_ms": 450,
  "created_at": "2026-08-20T00:00:00Z"
}

```

### 2. Poll Task Execution Status & Result

`GET /api/v1/forensics/tasks/{task_id}`

**Image Response (`200 OK`):**

```json
{
  "task_id": "8f3b2a19-...",
  "status": "COMPLETED",
  "verdict": {
    "is_ai_generated": true,
    "confidence_score": 0.932,
    "classification": "HIGH_CONFIDENCE_AI_GENERATED",
    "belief_distribution": {
      "belief_ai": 0.884,
      "belief_authentic": 0.041,
      "uncertainty": 0.075
    }
  },
  "sub_metrics": {
    "frequency_anomaly_score": 0.89,
    "ela_discrepancy_score": 0.94,
    "spatial_anomaly_score": 0.96
  },
  "artifacts": {
    "fft_spectrum_url": "/api/v1/artifacts/fft_8f3b2a19.png",
    "ela_heatmap_url": "/api/v1/artifacts/ela_8f3b2a19.png"
  },
  "execution_time_ms": 185
}

```

**Text Response (`200 OK`):**

```json
{
  "task_id": "a1b2c3d4-...",
  "status": "COMPLETED",
  "verdict": {
    "is_ai_generated": false,
    "confidence_score": 0.12,
    "classification": "AUTHENTIC_HUMAN_TEXT"
  },
  "sub_metrics": {
    "perplexity_score": 0.15,
    "burstiness_variance": 0.82,
    "linguistic_anomaly": 0.09
  },
  "artifacts": null,
  "execution_time_ms": 110
}

```

---

## 🧮 Mathematical & Algorithmic Core

### 1. 2D FFT Spectral Integration

Generative models introduce periodic grid artifacts during transposed convolution operations. We calculate the azimuthal average of the centered 2D Fourier power spectrum:

$$P(r) = \frac{1}{N_\theta} \sum_\theta \vert{}F(r, \theta)\vert{}^2$$

High-frequency outer bands are statistically evaluated for variance spikes.

### 2. Error Level Analysis (ELA)

Calculates localized pixel discrepancies after re-saving at a fixed 90% JPEG quality scale:

$$D(x, y) = \vert{}I_{\text{orig}}(x, y) - I_{\text{resaved}}(x, y)\vert{} \times 15.0$$

### 3. Evidential Logit Fusion

Combines heterogeneous component probabilities $p_i$ with reliability weights $w_i$ using calibrated Platt log-odds scaling:

$$\text{Logit}_{\text{fused}} = \sum_{i=1}^{k} w_i \cdot \ln \left( \frac{p_i}{1 - p_i} \right) + b$$

$$P(\text{AI Generated}) = \frac{1}{1 + e^{-\text{Logit}_{\text{fused}}}}$$
