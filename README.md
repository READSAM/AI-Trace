# AI-Trace

> **High-Throughput Asynchronous Multimodal Digital Forensics & Content Authenticity Engine**

AI-Trace is a distributed, event-driven backend pipeline and real-time dashboard designed to detect AI-generated media and synthetic manipulations across image and text modalities. Rather than running heavy signal transformations and neural inference inside synchronous HTTP cycles, AI-Trace decouples ingestion, background task processing, and evidential decision fusion, visualized through a dynamic Next.js interface.

---

## System Architecture

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
│ • Patch Autoencoder│   │ • DeBERTa Classifier   │
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

## Key Features

* **Real-Time Polling Dashboard:** Next.js frontend with dynamic metric rendering that automatically adapts to image-specific visual heatmaps or text-specific NLP cards.
* **Non-Blocking Ingestion:** Instant `202 Accepted` responses with microsecond turnaround via FastAPI and Celery.
* **$O(1)$ Fast-Path Caching:** Computes Perceptual Hash (`pHash`) for images and SHA-256 for text, bypassing worker computation on duplicate inputs.
* **Frequency-Domain Signal Analysis:** 2D Fast Fourier Transform (FFT) and azimuthal radial power spectrum integration.
* **Spatial Compression Analysis:** Error Level Analysis (ELA) with scaled differential matrices to highlight non-uniform local JPEG re-compression.
* **Latent Noise Reconstruction:** Evaluates diffusion-generated structural anomalies by calculating the reconstruction loss of local image patches through a custom noise autoencoder.
* **Statistical NLP Forensics:** Autoregressive log-likelihood perplexity mapping and sentence-level stylometric burstiness profiling.
* **Transformer-Based Classification:** Context-aware semantic analysis using fine-tuned sliding-window sequence classifiers (DeBERTa-v3).
* **Mathematical Evidential Decision Fusion:** Fuses conflicting signals into calibrated probabilities and Dempster-Shafer belief distributions.

---

## Celery Worker Pipelines

AI-Trace offloads all intensive computations to specialized asynchronous Celery workers running isolated environments.

### Vision Pipeline (`run_image_pipeline`)

1. **Hex Byte Ingestion:** Receives image data as hex strings over Redis to bypass JSON serialization limits, decoded directly into NumPy `uint8` arrays via OpenCV.
2. **Multi-Algorithm Extraction:**
* **FFT Spectral Analyzer:** Computes the `frequency_anomaly_score`.
* **ELA Differential Engine:** Computes the `ela_discrepancy_score`.
* **Complex Scene Forensics (Autoencoder):** Extracts the `spatial_anomaly_score` and computes the `synthetic_noise_score` via patch-based latent reconstruction loss.


3. **Artifact Persistence:** Visually generated 2D FFT and ELA heatmaps are saved directly to the `/tmp/aitrace_artifacts` shared Docker volume.
4. **Evidential Fusion:** Sub-scores are routed through the `EvidentialFusionEngine` using strict modality weights and bias parameters.

### NLP Pipeline (`run_text_pipeline`)

1. **Stylometric Analyzer:** Utilizes NLTK sentence tokenization to calculate variance against the mean length distribution (`burstiness`).
2. **Perplexity Calculator:** Loads GPT-2 via HuggingFace `transformers` to compute sliding-window autoregressive cross-entropy loss and raw perplexity mapping.
3. **DeBERTa Sequence Classifier:** Chunks input context into 512-token windows, passing them through `microsoft/deberta-v3-small` for semantic classification.
4. **Evidential Fusion & Metadata:** Analyzes the three sub-scores through the fusion engine and dynamically calculates document structural metadata (total word and sentence counts).

---

## Repository Structure

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
│       ├── vision_tasks.py  # 2D FFT, ELA, and Autoencoder workers
│       └── text_tasks.py    # Stylometrics, Burstiness, and PPL workers
├── test_pipeline.py         # End-to-end integration test & polling script
├── docker-compose.yml       # Multi-container orchestration (API, Redis, Worker)
└── README.md

```

---

## Quick Start

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

## API Reference

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### Submit Analysis Job

`POST /api/v1/forensics/analyze`

**Request (multipart/form-data):**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `modality` | string | Yes | `IMAGE` or `TEXT` |
| `file` | binary | Conditional | Image file (required if modality=IMAGE) |
| `text_content` | string | Conditional | Raw text payload (required if modality=TEXT) |

---

## Mathematical & Algorithmic Core

### 1. Image: 2D FFT Spectral Integration

Generative models introduce periodic grid artifacts during transposed convolution operations. We calculate the azimuthal average of the centered 2D Fourier power spectrum:

$$P(r) = \frac{1}{N_\theta} \sum_\theta \vert{}F(r, \theta)\vert{}^2$$

High-frequency outer bands are statistically evaluated for variance spikes.

### 2. Image: Error Level Analysis (ELA)

Calculates localized pixel discrepancies after re-saving at a fixed 90% JPEG quality scale:

$$D(x, y) = \vert{}I_{\text{orig}}(x, y) - I_{\text{resaved}}(x, y)\vert{} \times 15.0$$

### 3. Image: Latent Noise Reconstruction (Autoencoder)

Diffusion models (like Midjourney or Stable Diffusion) produce synthetic noise patterns that mathematically differ from natural camera sensor noise. The image is segmented into $64 \times 64$ local patches and passed through a specialized convolutional autoencoder. The system flags regions exhibiting high Mean Squared Error (MSE) reconstruction loss:

$$L_{\text{recon}} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2$$

### 4. Text: Stylometric Variance & Burstiness

Human writers naturally vary their sentence lengths, creating "bursts" of long and short structures, while LLMs tend to regress to a uniform mean length. Burstiness is calculated over the sentence length vector $L$:

$$\text{Burstiness} = \frac{\sigma_L - \mu_L}{\sigma_L + \mu_L}$$

A lower or flat burstiness score is linearly mapped to a higher AI-generation probability.

### 5. Text: Autoregressive Log-Likelihood (Perplexity)

Evaluates token predictability using a causal language model (e.g., GPT-2) via a sliding-window cross-entropy loss. Synthetic text typically clusters in low-perplexity regimes (PPL < 20):

$$\text{Perplexity}(X) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(x_i \vert{} x_1, \dots, x_{i-1})\right)$$

Raw perplexity is mapped to an AI probability space using a sigmoid decay curve:

$$P(\text{AI}) = \frac{1}{1 + \exp\left(\frac{PPL - 25.0}{5.0}\right)}$$

### 6. Text: Transformer-Based Sequence Classification

Leverages fine-tuned bidirectional transformers (`microsoft/deberta-v3-small`) to evaluate semantic context. The text is chunked into 512-token windows, and sliding-window attention is applied to extract the underlying softmax logic mapping AI versus human semantics.

### 7. Evidential Logit Fusion

Combines heterogeneous component probabilities $p_i$ across all worker modalities with reliability weights $w_i$ using calibrated Platt log-odds scaling:

$$\text{Logit}_{\text{fused}} = \sum_{i=1}^{k} w_i \cdot \ln \left( \frac{p_i}{1 - p_i} \right) + b$$

$$P(\text{AI Generated}) = \frac{1}{1 + e^{-\text{Logit}_{\text{fused}}}}$$

---

## License

**Copyright © 2026 Samriddhi. All Rights Reserved.**

This software and associated documentation files (the "Software") are proprietary and confidential. No part of this Software may be reproduced, copied, modified, distributed, or transmitted in any form or by any means without prior written permission from the author.

For academic inquiries, architectural discussions, or authorized usage requests, please contact the repository owner directly.
