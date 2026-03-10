# Content Guide — Harshini Murugadoss Portfolio

> Single source of truth for portfolio content. Updated: March 2026.

---

## 1. Profile Information

| Field | Value |
|---|---|
| Full Name | Harshini Murugadoss |
| Headline | AI & Machine Learning Engineer · Architecting scalable AI solutions with a focus on GenAI, Computer Vision, and MLOps |
| Email | harshini.murugadoss@outlook.com |
| Phone | +1 (951) 558-0221 |
| LinkedIn | https://linkedin.com/in/harshinimurugadoss |
| GitHub | https://github.com/Harshini1331 |
| Kaggle | https://www.kaggle.com/harshinimurugadoss13 |
| Medium | https://medium.com/@harshini13 |

---

## 2. About Me

### Heading
The Brain Behind the Machine

### Body
I hold a Master's in Robotics from UCR and a B.Tech in AI and Data Science, but my true passion isn't just the hardware — it's the intelligence inside it. I view AI as the "brain" of any system, and I've spent my academic and professional career learning exactly how to make that brain smarter, faster, and more efficient.

Recruiters often see models that live and die in a Jupyter Notebook. That's not me. I'm an engineer who builds for the real world.

### Why Me?

**I Deliver End-to-End:** I don't just train models; I wrap them in full-stack applications and deploy them. I've built everything from Tamil handwritten character recognition engines using CNNs to production RAG pipelines with streaming APIs and agentic LLM routing.

**I Love Optimization:** I find satisfaction in making systems run smoother and cheaper. Whether it's benchmarking 10+ LLMs for production or implementing Redis caching for a 150–400x response speedup, I focus on efficiency.

**I Figure Things Out:** My engineering philosophy is simple: there is always a solution if you are willing to look for it. From navigating autonomous robots in Gazebo to building agentic LangGraph workflows, I don't shy away from complex problems — I solve them.

**The Goal:** I am currently seeking an AI/MLE role where I can apply this technical rigor and "figure it out" attitude to build scalable, high-impact AI solutions.

### Stats (header counters)
- **7+** End-to-End Projects
- **98%** Model Accuracy Achieved
- **40%** Optimization Gain

---

## 3. Projects

---

### Project 1: IMS Bearing Fault Detection System

**Tag:** MLOps · Predictive Maintenance  
**One-liner:** End-to-end predictive maintenance pipeline detecting machinery anomalies in real-time, deployed via NVIDIA Triton Inference Server.  
**GitHub:** https://github.com/Harshini1331/ims-fault-diagnosis.git  
**Images:** `images/IMS Fault`

#### The Challenge
Bearings are the "wheels" of industrial machinery — when they fail, production lines stop, costing manufacturers millions in unplanned downtime. Using the NASA IMS Bearing Dataset (University of Cincinnati), the goal was to detect early fault signatures in high-frequency vibration data to enable scheduled maintenance before catastrophic failure.

#### The Solution
A lightweight, interpretable ML pipeline processing 8-channel vibration data. Instead of a heavy "black-box" deep learning model, I focused on manual feature engineering to extract statistical signals (Kurtosis, RMS) that mechanical engineers trust. The final model is containerized and served via NVIDIA Triton for scalable, production-ready inference.

#### Technical Highlights
- **Pragmatic MLOps (Triton Python Backend):** ONNX export proved brittle for complex Scikit-Learn pre-processors, so I pivoted to the Triton Python Backend — serving the native joblib model while gaining dynamic batching, concurrent execution, and standardized APIs.
- **Feature Engineering over Deep Learning:** Deliberately chose Random Forest over CNN. With scarce failure events, DL would overfit. 40 statistical features (RMS, Skewness, Kurtosis across 8 channels) achieved 98% accuracy with microsecond inference suitable for edge CPUs.
- **Containerized Scalability:** Full inference runtime packaged in Docker (`tritonserver:23.12-py3`) — write once, run anywhere across local servers or cloud clusters.

#### Tech Stack
`Scikit-learn` `Pandas` `NumPy` `NVIDIA Triton Inference Server` `Docker` `HTTP/gRPC (tritonclient)` `NASA IMS Dataset`

#### Key Metrics
- **98%** Model accuracy on validation set
- **<50ms** End-to-end latency (client → feature extraction → inference → response)
- **40** Statistical features extracted per sample

#### Architecture
```
Raw Vibration Sensor Data
  → Preprocessing Client (Feature Extraction)
  → HTTP Request (Tensor Payload)
  → Docker Container (Triton Server)
  → Python Backend (Random Forest)
  → Failure Probability
```

---

### Project 2: Medical AI — Pulmonary Health Diagnosis

**Tag:** Computer Vision · Healthcare AI  
**One-liner:** Comparative deep learning study benchmarking 5 CNN architectures for multi-class diagnosis of COVID-19, Pneumonia, and Healthy lungs.  
**GitHub:** https://github.com/Harshini1331/DECODING-PULMONARY-HEALTH.git  
**Images:** `images/Medical AI`

#### The Challenge
Automated medical diagnosis faces two major hurdles: class imbalance (far more "Normal" X-rays than "COVID-19" ones) and visual ambiguity (Viral and Bacterial Pneumonia look nearly identical on X-rays). The goal was to determine which architecture could best handle these constraints and reliably detect critical pathologies without majority-class bias.

#### The Solution
A robust training pipeline evaluating 5 architectures (VGG16, ResNet50, DenseNet121, InceptionV3, Custom CNN) on a 4-class dataset. To combat overfitting and bias, I implemented Generative Data Augmentation (rotation, shear, zoom) and Inverse Class Weights to penalize the model heavily for missing minority classes like COVID-19.

#### Technical Highlights
- **Battle of Architectures:** Contrary to "deeper is better," VGG16 (78.56%) outperformed ResNet50 (60.6%) and DenseNet121 (69%). For lower-resolution medical imaging (180×180), simpler linear architectures generalize better — deeper networks suffered from overfitting and vanishing gradients.
- **Handling Class Imbalance:** Dynamic weighting strategy (`weight = total / (4 × count)`) balanced the loss function, achieving Recall of 0.83 for critical COVID-19 and NORMAL classes — prioritizing sensitivity over raw accuracy.
- **Honest Failure Analysis:** Model successfully separated "Healthy" from "Sick" but struggled to distinguish Viral from Bacterial Pneumonia (F1 ~0.00 for Viral in Custom CNN) — highlighting the need for higher-resolution inputs (>224px) or clinical metadata.

#### Tech Stack
`TensorFlow` `Keras` `VGG16` `ResNet50` `DenseNet121` `InceptionV3` `Transfer Learning` `Data Augmentation` `Class Weighting`

#### Key Metrics
- **78.56%** Top test accuracy (VGG16)
- **0.83** Recall (sensitivity) for COVID-19 cases
- **5** Architectures benchmarked end-to-end

---

### Project 3: SmartEyes — Real-Time Obstacle Detection

**Tag:** Computer Vision · Real-Time AI  
**One-liner:** Lightweight multi-mode object detection system identifying 80+ classes in real-time for autonomous navigation and surveillance.  
**GitHub:** https://github.com/Harshini1331/SmartEyes-Obstacle-Detection-System.git  
**Images:** `images/SmartEyes`

#### The Challenge
In dynamic environments — autonomous vehicles or security surveillance — latency is fatal. A one-second delay in obstacle detection can cause a collision. The challenge was to build a vision system with high confidence detection without the massive computational lag of older architectures like Faster R-CNN, running on consumer-grade hardware.

#### The Solution
SmartEyes uses the YOLOv7 architecture (SOTA for speed-accuracy tradeoff at time of build). Unlike hard-coded single-input scripts, I built a flexible multimode pipeline capable of switching between Live Webcam Streams, Pre-recorded Video, and Static Images via CLI arguments.

#### Technical Highlights
- **SOTA Model Utilization:** YOLOv7's "Bag of Freebies" architectural optimizations achieved significantly higher mAP than YOLOv3 at a fraction of the model size (71MB vs 240MB).
- **Multimode Input Pipeline:** Three distinct data streams via `--mode camera`, `--mode video`, `--mode image` — same codebase for real-time robotic navigation tests and post-process surveillance analysis.
- **Performance Optimization:** Tuned NMS thresholds to hardware capabilities, maintaining 30 FPS on GPU-enabled hardware and preventing the "stutter" common in unoptimized Python CV applications.

#### Tech Stack
`YOLOv7` `PyTorch` `OpenCV` `Python` `COCO Dataset (80 classes)`

#### Key Metrics
- **30 FPS** Real-time inference on GPU-enabled consumer hardware
- **71 MB** Lightweight model footprint suitable for embedded deployment
- **80+** Object classes detected out-of-the-box

---

### Project 4: Uncertainty-Aware Cooperative Occupancy Prediction

**Tag:** Robotics · Deep Learning · Research  
**One-liner:** Deep learning framework enabling robot teams to share LiDAR data and "see through walls" using Dual Attention mechanisms for occlusion resolution.

#### The Research Gap
In crowded spaces like hospitals or warehouses, a single robot is limited by its line-of-sight. Blind spots caused by corners or obstacles create safety hazards. The goal was to solve this "Perceptual Horizon" problem using Cooperative Perception (V2V), treating a fleet of robots as a distributed sensor network.

#### The Solution
An Early Fusion framework fusing raw LiDAR point clouds from multiple agents into a unified spatiotemporal prediction. Integrates a novel Dual Attention Mechanism into a Recurrent Variational Autoencoder (RVAEP) backbone — selectively attending to dynamic actors while suppressing sensor noise, generating future-aware occupancy maps.

#### Technical Highlights
- **Dual Attention Innovation:**
  - *Spatial Attention (SAM):* Acts as a "visual spotlight," assigning high weights to dynamic regions (pedestrians) and ignoring static background.
  - *Channel Attention (CAM):* Addresses "ghosting" from sensor misalignment by dynamically re-weighting feature channels to suppress noise.
- **Uncertainty-Aware Navigation:** Outputs Entropy Maps quantifying confidence — allowing navigation planners to distinguish "definitely empty" space from "uncertain" zones (erratic human motion), enabling cautious behavior in high-entropy areas.
- **High-Fidelity Simulation:** Built a custom Unity3D environment using NVIDIA PhysX and NavMesh agents to collect 360,000 synchronized frames across diverse layouts (T-intersections, narrow corridors).

#### Tech Stack
`PyTorch` `RVAEP` `ConvLSTM` `Unity3D` `C#` `PhysX` `NavMesh` `Early Fusion` `HPC Cluster (RTX 3090-tier)`

#### Key Metrics
- **11.8%** Reduction in Binary Cross-Entropy loss vs. standard fusion baselines
- **Ghosting elimination** in dynamic object prediction (qualitative)
- **360,000** Proprietary frames generated for multi-agent social navigation

---

### Project 5: Intelligent Crop Recommendation System

**Tag:** Data Science · Precision Agriculture  
**One-liner:** Precision agriculture tool recommending optimal crops from soil and climatic data, achieving 99.54% accuracy via Naive Bayes.  
**GitHub:** https://github.com/Harshini1331/Crop-Rotation-Planner.git

#### The Challenge
Planting the wrong crop for specific soil conditions leads to poor yield and financial loss. Traditional farming relies on heuristics or expensive, slow soil testing labs. The goal was to democratize Precision Agriculture by instantly analyzing soil metrics (N, P, K, pH) to recommend the scientifically optimal crop.

#### The Solution
A comparative ML framework training and benchmarking 6 algorithms (Decision Tree, Random Forest, SVM, Logistic Regression, KNN, Naive Bayes) on N-P-K levels, temperature, humidity, and rainfall. The final system is deployed via Streamlit for intuitive farmer-facing use.

#### Technical Highlights
- **The Naive Bayes Surprise:** While Random Forest (99.3%) is usually the default winner for tabular data, Gaussian Naive Bayes performed best at 99.54%. Crop features likely follow independent Gaussian distributions per crop type, making Naive Bayes the theoretically perfect estimator for this domain.
- **Rigorous EDA:** Correlation heatmaps and histograms verified feature independence and distribution shapes before model selection — confirmed distinct correlations for features like Phosphorous and Potassium per crop type.
- **Full-Stack ML Deployment:** Wrapped the winning model in a Streamlit web app, bridging complex Scikit-Learn logic and end-users (farmers) who need a simple GUI.

#### Tech Stack
`Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Streamlit` `Kaggle Crop Recommendation Dataset`

#### Key Metrics
- **99.54%** Test accuracy (Naive Bayes winner)
- **6** Distinct ML algorithms benchmarked
- **<1 second** Inference time for real-time recommendations

---

### Project 6: Autonomous Warehouse Pick-and-Deliver Robot

**Tag:** Robotics · ROS · Path Planning  
**One-liner:** ROS-based autonomous mobile manipulator with SLAM, RRT* path planning, and dynamic obstacle avoidance for flexible warehouse logistics.

#### The Challenge
Current warehouse automation often relies on rigid infrastructure like magnetic tracks — lacking flexibility and affordability. The goal was a versatile, low-cost autonomous robot capable of navigating dynamic environments and handling objects without pre-defined paths.

#### The Solution
A unified robotic system using a TurtleBot3 Waffle Pi mobile base integrated with an OpenManipulator arm, implemented in ROS and simulated in Gazebo. Features a custom Navigation Node using GMapping for SLAM and RRT* for optimal path planning, plus a safety layer using LiDAR-based Time-to-Collision (TTC) estimation for real-time re-planning around moving obstacles.

#### Technical Highlights
- **RRT* Path Planning:** Unlike deterministic planners, RRT* generates asymptotically optimal, collision-free paths in complex, clutter-filled environments.
- **Closed-Loop Navigation:** PID Controller precisely tracks waypoints from the global planner, ensuring smooth trajectory execution and minimal tracking error.
- **Integrated Manipulation:** Solves Inverse Kinematics (IK) for the 4-DOF OpenManipulator arm — calculating geometric joint angles to autonomously grasp and transport objects after navigation goal is reached.

#### Tech Stack
`ROS` `Gazebo` `Python` `GMapping (SLAM)` `RRT*` `PID Control` `Inverse Kinematics` `TurtleBot3` `OpenManipulator-X` `LiDAR` `IMU`

#### Key Metrics
- **Collision-free navigation** around dynamic obstacles (simulated humans) via TTC estimation
- **High-fidelity 2D occupancy grid maps** (.pgm) of warehouse layout generated
- **Full autonomy validated:** Mapping → Planning → Navigation → Manipulation

---

### Project 7: ScholarStream — Production RAG Pipeline

**Tag:** GenAI · RAG · Production MLOps  
**One-liner:** Production RAG pipeline: arXiv ingestion → GPU PDF parsing → hybrid vector search → agentic LLM routing → real-time streaming answers.  
**GitHub:** https://github.com/harshini1331/ScholarStream  
**Images:** `images/ScholarStream`

#### The Challenge
Building a RAG system that actually works in production — not just a notebook demo — requires solving multiple hard problems at once: structured ingestion of academic PDFs at scale, efficient hybrid retrieval that beats pure vector search, intelligent query routing to avoid unnecessary LLM calls, and real-time streaming responses, all while running entirely locally with zero API costs.

#### The Solution
ScholarStream is a fully dockerized, 7-service RAG pipeline. Papers are fetched from arXiv, parsed using GPU-accelerated Docling (RTX 5070) into structured Markdown, split via two-stage semantic chunking, embedded with `nomic-embed-text` (768d), and indexed in OpenSearch. A FastAPI serving layer handles hybrid BM25 + KNN search via manual Reciprocal Rank Fusion, with Redis caching (150–400x speedup on cache hits), SSE streaming, and a LangGraph agentic mode that decides whether retrieval is even needed before calling the LLM. The corpus includes 3,700+ chunks from foundational papers: DDPM, Attention Is All You Need, GPT-3, LLaMA, CLIP, SAM, RAG (Lewis et al.), and more.

#### Technical Highlights
- **Hybrid BM25 + KNN Search (Manual RRF):** OpenSearch 2.11 doesn't natively support hybrid queries with RRF, so I implemented Reciprocal Rank Fusion manually in Python — running BM25 and KNN independently then merging via `1/(k+rank_A) + 1/(k+rank_B)` (k=60). Consistently outperforms either method alone.
- **Agentic RAG (LangGraph):** A deterministic state machine (decide → retrieve → grade → rewrite → generate) routes simple queries to direct answers without retrieval, grades each retrieved chunk for relevance, and rewrites the query if grading fails (max 2 retries). Used LangGraph over tool-calling agents because llama3 doesn't reliably support tool-calling — deterministic graph makes routing auditable via `reasoning_steps`.
- **Redis Response Cache:** SHA-256 keyed cache with 24h TTL produces ~100ms responses on cache hits vs. 15–20s on misses — a 150–400x speedup. Gracefully self-disables if Redis is unavailable.
- **GPU-Accelerated PDF Parsing:** Docling runs on the RTX 5070 via NVIDIA CUDA 12.4.1 for layout analysis — preserving headings, tables, and section structure as Markdown, which directly improves chunking quality over raw text extraction.
- **Full Observability:** Langfuse v3 span tracing instruments every request — separate spans for embed, retrieve, and generate phases with timing data visible in Langfuse Cloud.

#### Tech Stack
`Python 3.12` `FastAPI` `LangChain` `LangGraph` `OpenSearch 2.11` `PostgreSQL 16` `Ollama` `Llama 3` `nomic-embed-text` `Apache Airflow 3.1.7` `Redis 7` `Langfuse v3` `Docling` `Docker Compose` `NVIDIA CUDA 12.4` `uv`

#### Key Metrics
- **3,700+** Semantic chunks indexed across foundational ML papers
- **150–400×** Response speedup via Redis cache (100ms vs. 15–20s)
- **7** Microservices orchestrated via Docker Compose
- **9** REST API endpoints (ask, stream, agentic, search, hybrid-search, health, stats, cache, UI)
- **~100ms** Cache-hit response time

#### Architecture
```
arXiv API
  → Docling PDF Parser (GPU · RTX 5070)
  → Two-Stage Semantic Chunker
  → nomic-embed-text (768d embeddings · Ollama)
  → OpenSearch (BM25 + KNN · HNSW index)
  → PostgreSQL (full text + metadata)

FastAPI Serving Layer
  → Redis Cache (SHA-256 key · 24h TTL)
  → Hybrid Search (BM25 + KNN · RRF fusion)
  → Llama 3 (Ollama · local LLM)
  → SSE Streaming / Standard / Agentic (LangGraph)
  → Langfuse Observability
```

---

## 4. Skills

### AI & Machine Learning
`TensorFlow` `PyTorch` `Keras` `Scikit-learn` `OpenAI API` `Google Gemini API` `LangChain` `LangGraph` `Hugging Face` `YOLOv7` `ONNX`

### GenAI & RAG
`LangChain` `LangGraph` `Ollama` `nomic-embed-text` `OpenSearch` `pgvector` `Hybrid Search (BM25 + KNN)` `Langfuse` `Prompt Engineering` `Agentic RAG`

### Robotics & Simulation
`ROS` `Gazebo` `RViz` `SLAM (GMapping)` `RRT*` `Inverse Kinematics` `Unity3D`

### Full Stack & Cloud
`Python` `C++` `JavaScript` `React.js` `TypeScript` `Node.js` `Flask` `FastAPI` `PostgreSQL` `Supabase` `GCP` `Vercel`

### MLOps & Infrastructure
`Docker` `Docker Compose` `NVIDIA Triton Inference Server` `Apache Airflow` `Redis` `CI/CD` `Git` `uv`

---

## 5. Experience & Education

### Education

| Degree | Institution | Score | Dates |
|---|---|---|---|
| MS in Robotics | University of California, Riverside | GPA: 3.97/4.0 | Sep 2024 – Dec 2025 |
| B.Tech in AI & Data Science | Saveetha Engineering College, Chennai | CGPA: 9.13/10.0 | Aug 2020 – May 2024 |

---

### Work Experience

#### AI Engineer Intern — India Literacy Project (ILP), Remote
**Dates:** Jun 2025 – Nov 2025  
**Domain:** GenAI · EdTech · Speech Processing

**One-liner:** GenAI-powered career guidance platform for rural India, featuring custom speech-to-text for local dialects and teacher-in-the-loop verification.

**Key Work:**
- Architected a full-stack "Career Journey" platform allowing rural students to speak in Kannada/Tamil and receive AI-generated career guidance — routed through a Teacher Verification Dashboard before delivery.
- Built a custom Phonetic Restoration Engine (`speechToTextService.ts`) with a 3-layer STT fallback (Google STT → Azure → Gemini Flash) to handle rural Indian accents and code-switching.
- Leveraged Gemini's 1M+ token context window for long-context personalization — injecting each student's full longitudinal history (assessments, dreams, hobbies) directly into the prompt, eliminating the latency of vector retrieval.
- Implemented Hybrid Search RAG via Supabase pgvector — combining semantic similarity (Gemini text-embedding-004) with SQL metadata filters (student class/location) for culturally relevant advice.
- Optimized API calls by 40% through aggressive caching strategies.

**Tech Stack:** `Google Gemini API` `Supabase (pgvector)` `React` `TypeScript` `TanStack Query` `Prompt Engineering`

---

#### AI Research Intern — PURPLESPOT
**Dates:** Jan 2024 – May 2024  
**Domain:** NLP · Logistics Automation

**Key Work:**
- Engineered an automated NLP Lead Extraction Pipeline converting unstructured WhatsApp logistics messages into structured SQL database entries — replacing manual data entry.
- Conducted cost-benefit analysis of Regex vs. LLMs; pivoted to cost-efficient LLM (Gemini/GPT) after Regex failed on noisy, human-generated text (0% success on edge cases).
- Designed prompt engineering strategies to accurately extract 4 key entities (Origin, Destination, Truck Size, Date) in the presence of typos, slang, and mixed languages.

**Tech Stack:** `Python` `LLMs` `Prompt Engineering` `NLP` `WhatsApp Automation`

---

#### Software Developer Intern — DGTEL Rehumanice
**Dates:** Jan 2023 – Dec 2023  
**Domain:** EdTech · On-Device ML

**Key Work:**
- Engineered an end-to-end On-Device Tamil Handwritten Character Recognition system for a children's language learning app.
- Curated a proprietary dataset of 247 Tamil characters written by children to handle erratic handwriting — no public dataset existed for this demographic.
- Designed and trained a lightweight Custom CNN optimized for Tamil script structure, then converted to TensorFlow Lite for offline, real-time feedback (<50ms latency) without internet dependency.

**Tech Stack:** `TensorFlow` `TFLite` `Custom CNN` `Python` `Mobile Deployment`

---

## 6. Certifications

**Location:** `images/certificate`

---

## 7. Resume

PDF resume link — TBD.