# Content Required for Your Portfolio Website

To build a high-impact portfolio that highlights your skills as a Machine Learning Engineer, please provide the following details:

## 1. Profile Information
- **Full Name**: Harshini Murugadoss
- **Professional Headline**: AI & Machine Learning Engineer | Architecting scalable AI solutions with a focus on GenAI, Computer Vision, and MLOps
- **Social Links**:
  - LinkedIn URL - https://linkedin.com/in/harshinimurugadoss
  - GitHub URL - https://github.com/Harshini1331
  - Kaggle - https://www.kaggle.com/harshinimurugadoss13
  - Medium - https://medium.com/@harshini13
- **Contact Email**: harshini.murugadoss@outlook.com
- **Phone Number**: +1 (951) 558-0221

## 2. About Me
The Brain Behind the Machine
I hold a Master’s in Robotics from UCR and a B.Tech in AI and Data Science, but my true passion isn’t just the hardware, it’s the intelligence inside it. I view AI as the "brain" of any system, and I’ve spent my academic and professional career learning exactly how to make that brain smarter, faster, and more efficient.

Recruiters often see models that live and die in a Jupyter Notebook. That’s not me. I’m an engineer who builds for the real world.

Why Me?
I Deliver End-to-End: I don’t just train models; I wrap them in full-stack applications and deploy them. I’ve built everything from Tamil handwritten character recognition engines using CNNs to GenAI-powered career platforms served via React and Supabase.

I Love Optimization: I find satisfaction in making systems run smoother and cheaper. Whether it is benchmarking 10+ LLMs (including GPT-4 and Llama 2) to find the perfect fit for production or optimizing state management to reduce API calls by 40%, I focus on efficiency.

I Figure Things Out: My engineering philosophy is simple: there is always a solution if you are willing to look for it. From navigating autonomous robots in Gazebo to architecting RAG pipelines, I don't shy away from complex problems,I solve them.

The Goal: I am currently seeking an AI/MLE role where I can apply this technical rigor and "figure it out" attitude to build scalable, high-impact AI solutions.

## 3. Projects (The most important section!)
### Project 1: IMS Bearing Fault Detection System
One-Line Summary An end-to-end predictive maintenance pipeline capable of detecting machinery anomalies in real-time, deployed via NVIDIA Triton Inference Server.

The Challenge Bearings are the "wheels" of industrial machinery; when they fail, production lines stop, costing manufacturers millions in unplanned downtime. The goal was to build a system that could predict these failures before they became catastrophic. Using the standard NASA IMS Bearing Dataset (University of Cincinnati), I needed to detect early fault signatures in high-frequency vibration data to enable scheduled maintenance.

The Solution I engineered a lightweight, interpretable Machine Learning pipeline that processes 8-channel vibration data. Instead of using a heavy, "black-box" Deep Learning model, I focused on manual feature engineering to extract statistical signals (like Kurtosis and RMS) that mechanical engineers actually trust. The final model is containerized and served via NVIDIA Triton, ensuring the system is scalable, standardized, and production-ready.

Technical Highlights (The "Portfolio Gold")

Pragmatic MLOps (Triton Python Backend): I initially attempted to export the Scikit-Learn pipeline to ONNX, but the conversion for complex pre-processors proved brittle. Instead of getting stuck, I pivoted to the Triton Python Backend. This allowed me to serve the native joblib model while still gaining Triton's benefits—dynamic batching, concurrent execution, and standardized APIs—without fighting the graph conversion tools.

Feature Engineering vs. Deep Learning: I deliberately chose a Random Forest over a CNN. Given the dataset's scarcity of "failure events," a Deep Learning model would likely overfit. By extracting 40 statistical features (RMS, Skewness, Kurtosis across 8 channels), I achieved 98% accuracy with a model that is computationally cheap enough to run on edge CPUs (microseconds vs. milliseconds).

Containerized Scalability: The entire inference runtime is packaged in Docker (tritonserver:23.12-py3). This means the system is "write once, run anywhere"—whether deployed on a local server or a cloud cluster, the environment is identical.

Tech Stack

ML Frameworks: Scikit-learn, Pandas, NumPy

Deployment: NVIDIA Triton Inference Server, Docker

Protocol: HTTP (via tritonclient), architected to support gRPC

Data: NASA IMS Bearing Dataset

Key Metrics

98% Model Accuracy on validation set.

<50ms End-to-End Latency (Client Request → Feature Extraction → Inference → Response).

40 Statistical Features extracted per sample.

Architecture Diagram

[Raw Vibration Sensor Data] --> [Preprocessing Client (Feature Extraction)] --> [HTTP Request (Tensor Payload)] --> [Docker Container (Triton Server)] --> [Python Backend (Random Forest)] --> [Failure Probability]

Links: https://github.com/Harshini1331/ims-fault-diagnosis.git

Images: inside images/IMS Fault

### Project 2: Medical AI: Pulmonary Health Diagnosis

One-Line Summary A comparative deep learning study benchmarking 5 CNN architectures for the multi-class diagnosis of COVID-19, Pneumonia, and Healthy lungs.

The Challenge Automated medical diagnosis faces two major hurdles: Class Imbalance (far more "Normal" X-rays than "COVID-19" ones) and Visual Ambiguity (Viral and Bacterial Pneumonia look incredibly similar on X-rays). The goal was not just to "fit a model," but to determine which architecture could best handle these constraints and reliably detect critical pathologies without being biased toward the majority class.

The Solution I engineered a robust training pipeline to evaluate 5 distinct architectures (VGG16, ResNet50, DenseNet121, InceptionV3, and a Custom CNN) on a 4-class dataset. To combat overfitting and bias, I implemented Generative Data Augmentation (rotation, shear, zoom) and calculated Inverse Class Weights to penalize the model heavily for missing minority classes like COVID-19.

Technical Highlights (The "Portfolio Gold")

The "Battle of Architectures" (VGG16 vs. The Rest): Contrary to the popular belief that "deeper is better," my benchmarking revealed that VGG16 (78.56% Accuracy) outperformed complex models like ResNet50 (60.6%) and DenseNet121 (69%). This demonstrated that for lower-resolution medical imaging (180x180), simpler, linear architectures generalize better than deeper networks which suffered from overfitting and vanishing gradients.

Handling Class Imbalance: I implemented a dynamic weighting strategy (weight = total / (4 * count)) to balance the loss function. This ensured that despite the dataset skew, the model achieved a high Recall of 0.83 for the critical COVID-19 and NORMAL classes, prioritizing "Sensitivity" (catching the disease) over raw accuracy.

Honest Failure Analysis: My evaluation uncovered a critical limitation in current CNNs: while the model successfully separated "Healthy" from "Sick," it struggled to distinguish Viral Pneumonia from Bacterial Pneumonia (F1-score ~0.00 for Viral in the Custom CNN). This insight highlights the specific radiological difficulty of separating pneumonia subtypes without higher-resolution inputs (>224px) or clinical metadata.

Tech Stack

Models: VGG16, ResNet50, DenseNet121, InceptionV3, Custom CNN

Frameworks: TensorFlow, Keras

Techniques: Transfer Learning, Data Augmentation, Class Weighting

Data: Curated Chest X-Ray Dataset (Kermany + COVID-19 Collection)

Key Metrics

78.56% Top Test Accuracy (VGG16).

0.83 Recall (Sensitivity) for COVID-19 cases.

5 Architectures Benchmarked end-to-end.

Links:https://github.com/Harshini1331/DECODING-PULMONARY-HEALTH.git

Images: inside images/Medical AI

### Project 3: SmartEyes - Real-Time Obstacle Detection
One-Line Summary A lightweight, multi-mode object detection system capable of identifying 80+ classes in real-time for autonomous navigation and surveillance applications.

The Challenge In dynamic environments—whether for autonomous vehicles or security surveillance—latency is fatal. A delay of even one second in detecting an obstacle can result in a collision. The challenge was to build a vision system that could detect objects with high confidence without the massive computational lag associated with older architectures like Faster R-CNN, running effectively on consumer-grade hardware.

The Solution I engineered SmartEyes using the YOLOv7 architecture, which was State-of-the-Art at the time for the speed-to-accuracy trade-off. Unlike standard scripts that are hard-coded for one input, I built a flexible "Multimode" pipeline capable of switching seamlessly between Live Webcam Streams, Pre-recorded Video, and Static Images. The system leverages the efficient ELAN architecture of YOLOv7 to deliver high-performance inference (30+ FPS) without needing aggressive quantization.

Technical Highlights (The "Portfolio Gold")

SOTA Model Utilization: Leveraged the 71MB Standard YOLOv7 model. By utilizing its "Bag of Freebies" (architectural optimizations), I achieved significantly higher mean Average Precision (mAP) than YOLOv3 at a fraction of the model size (71MB vs 240MB), proving that bigger isn't always better.

Multimode Input Pipeline: I architected the application to handle three distinct data streams via CLI arguments (--mode camera, --mode video, --mode image). This modular design allows the same codebase to be used for real-time robotic navigation tests and post-process video analysis for surveillance logs.

Performance Optimization: The hardest technical hurdle was balancing FPS vs. Confidence. By implementing non-max suppression (NMS) thresholds tailored to the hardware capabilities, I optimized the inference loop to maintain real-time fluidity (30 FPS) on GPU-enabled hardware, preventing the "stutter" common in unoptimized Python CV applications.

Tech Stack

Model: YOLOv7 (PyTorch)

Computer Vision: OpenCV (cv2)

Language: Python

Classes: COCO Dataset (80 Classes including Persons, Vehicles, Animals)

Key Metrics

30 FPS Real-time inference speed on GPU-enabled consumer hardware.

71 MB Lightweight model footprint suitable for embedded deployment.

80+ Object classes detected out-of-the-box.

Links: https://github.com/Harshini1331/SmartEyes-Obstacle-Detection-System.git

Images: inside images/SmartEyes

### Project 4: Uncertainty-Aware Cooperative Occupancy Prediction with Attention Mechanisms

One-Line Summary A Deep Learning framework enabling robot teams to share LiDAR data and "see through walls," utilizing Dual Attention mechanisms to resolve occlusions in dynamic social environments.

The Research Gap In crowded spaces like hospitals or warehouses, a single robot is limited by its line-of-sight. "Blind spots" caused by corners or obstacles create safety hazards—a robot cannot plan for a pedestrian walking out from behind a wall. The goal was to solve this "Perceptual Horizon" problem using Cooperative Perception (V2V), treating a fleet of robots as a distributed sensor network that shares data to eliminate blind spots.

The Solution I developed an Early Fusion framework that fuses raw LiDAR point clouds from multiple agents into a unified spatiotemporal prediction. Unlike standard fusion methods that suffer from noise and alignment artifacts ("ghosting"), my approach integrates a novel Dual Attention Mechanism into a Recurrent Variational Autoencoder (RVAEP) backbone. This allows the system to selectively "attend" to dynamic actors while suppressing sensor noise, generating sharp, future-aware occupancy maps.

Technical Highlights (The "Research Gold")

Dual Attention Innovation: Standard fusion treats all data equally, often leading to blurry predictions. I engineered two custom modules:

Spatial Attention (SAM): Acts as a "visual spotlight," assigning high weights to dynamic regions (pedestrians) and ignoring static background (walls), effectively filtering out irrelevant data.

Channel Attention (CAM): Addresses the "Ghosting" problem caused by sensor misalignment. It dynamically re-weights feature channels to suppress noise, ensuring that a single object isn't duplicated in the final map.

Uncertainty-Aware Navigation: The model doesn't just predict where objects are; it outputs Entropy Maps quantifying confidence. This allows downstream navigation planners to distinguish between "definitely empty" space and "uncertain" zones (e.g., erratic human motion), enabling robots to slow down cautiously in high-entropy areas rather than blindly planning a path.

High-Fidelity Simulation: To train this data-hungry model, I built a custom Unity3D environment utilizing NVIDIA PhysX and NavMesh agents. I collected a massive dataset of 360,000 synchronized frames across diverse layouts (T-intersections, narrow corridors) to bridge the gap between abstract grid worlds and realistic physics.

Tech Stack

Core Logic: Deep Learning (PyTorch), RVAEP, ConvLSTM

Simulation: Unity3D (C# scripting, PhysX, NavMesh)

Fusion: Early Fusion (Raw LiDAR), Homogeneous Coordinate Transformation

Compute: High-Performance Computing Cluster (RTX 3090-tier)

Key Metrics

11.8% Reduction in Binary Cross-Entropy (BCE) Loss compared to standard fusion baselines.

Ghosting Reduction: Qualitative elimination of "smearing" artifacts in dynamic object prediction.

360k Frame proprietary dataset generated for multi-agent social navigation.

### Project 5: Intelligent Crop Recommendation System

One-Line Summary A precision agriculture tool that utilizes Machine Learning to recommend optimal crops based on soil composition and climatic data, achieving 99.54% accuracy via Naive Bayes.

The Challenge Farming is inherently risky; planting the wrong crop for specific soil conditions can lead to poor yield and financial loss. Traditional farming relies on heuristics or expensive soil testing labs with slow turnaround. The goal was to democratize "Precision Agriculture" by building a digital tool that instantly analyzes soil metrics (Nitrogen, Phosphorous, pH) to recommend the scientifically optimal crop.

The Solution I developed a comparative Machine Learning framework to determine the best predictive model for crop suitability. Using a dataset of N-P-K levels, temperature, humidity, and rainfall, I trained and benchmarked 6 different algorithms (Decision Tree, Random Forest, SVM, Logistic Regression, KNN, and Naive Bayes). The final system is deployed via Streamlit, offering an intuitive interface where farmers can input their soil data and receive an instant, high-confidence crop recommendation.

Technical Highlights (The "Data Science" Flex)

The "Naive Bayes" Surprise: While Random Forest (99.3%) is usually the default winner for tabular data, my benchmarking revealed that Gaussian Naive Bayes performed best with 99.54% accuracy. This insight demonstrates deep understanding of the data: crop features (like pH and Temperature) likely follow independent Gaussian distributions for each specific crop type, making Naive Bayes the theoretically perfect estimator for this domain.

Exploratory Data Analysis (EDA): I didn't just feed data blindly. I utilized correlation heatmaps and histograms to verify feature independence and distribution shapes. This rigorous pre-analysis confirmed that features like "Phosphorous" and "Potassium" had distinct correlations for specific crops, validating the feature selection process.

Full-Stack ML Deployment: Instead of leaving the model in a notebook, I wrapped the winning model in a Streamlit web application. This bridges the gap between complex Scikit-Learn logic and end-users (farmers) who need a simple GUI.

Tech Stack

ML Algorithms: Naive Bayes (Winner), Random Forest, SVM, KNN

Libraries: Scikit-Learn, Pandas, NumPy, Matplotlib/Seaborn

Deployment: Streamlit

Data: Kaggle Crop Recommendation Dataset (N, P, K, pH, Rainfall)

Key Metrics

99.54% Test Accuracy achieved by the winning model (Naive Bayes).

6 Distinct ML architectures benchmarked.

<1 Second Inference time for real-time recommendations.

Links: https://github.com/Harshini1331/Crop-Rotation-Planner.git

### Project 6: Autonomous Warehouse Pick-and-Deliver Robot

One-Line Summary: A ROS-based autonomous mobile manipulator capable of SLAM, RRT* path planning, and dynamic obstacle avoidance for flexible warehouse logistics.

The Challenge: Current warehouse automation systems often lack flexibility and affordability, relying on rigid infrastructure like magnetic tracks.
The goal was to develop a versatile, low-cost autonomous robot capable of navigating dynamic environments and handling objects without pre-defined paths

The Solution: I engineered a unified robotic system using a TurtleBot3 Waffle Pi mobile base integrated with an OpenManipulator arm3. Implemented in ROS and simulated in Gazebo, the system features a custom Navigation Node that utilizes GMapping for SLAM and the RRT (Rapidly-exploring Random Tree Star)* algorithm for optimal path planning

To handle dynamic warehouse environments, I implemented a safety layer that uses LiDAR data to calculate Time-to-Collision (TTC), triggering real-time re-planning when moving obstacles (simulated humans) are detected

Technical Highlights (The "Robotics" Flex)

RRT Path Planning:
Unlike standard deterministic planners, I implemented RRT* to generate asymptotically optimal, collision-free paths in complex, clutter-filled environments

Closed-Loop Navigation:
I designed a PID Controller to precisely track the waypoints generated by the global planner, ensuring smooth trajectory execution and reducing tracking errors

Integrated Manipulation:
The system solves Inverse Kinematics (IK) for the 4-DOF OpenManipulator arm, calculating geometric joint angles to autonomously grasp and transport objects once the navigation goal is reached

Tech Stack

Framework: ROS (Robot Operating System)

Simulation: Gazebo (Physics engine) 

Algorithms: SLAM (GMapping), RRT* (Path Planning), PID Control, Inverse Kinematics

Hardware (Simulated): TurtleBot3, OpenManipulator-X, LiDAR, IMU

Key Metrics:

Collision-Free Navigation:
Successfully detected and avoided dynamic obstacles (simulated humans) using TTC estimation.

Mapping Accuracy:
Generated high-fidelity 2D occupancy grid maps (.pgm) of the warehouse layout.

Full Autonomy:
Validated end-to-end workflow: Mapping $\to$ Planning $\to$ Navigation $\to$ Manipulation

## 4. Skills
Technical Skills
Languages

Python: (NumPy, Pandas, Scikit-Learn)

C++: (Robotics/ROS, Low-level Control)

JavaScript: (React.js, Node.js)

SQL: (PostgreSQL, Supabase)

AI, Machine Learning & GenAI

Frameworks: TensorFlow, PyTorch, Keras, Scikit-learn.

Generative AI: OpenAI API, Google Gemini API, Hugging Face, LangChain.

Computer Vision: OpenCV, YOLOv7, ONNX.

Edge AI: TensorFlow Lite (Mobile Deployment).

Data Science: Matplotlib, Seaborn, Pandas, NumPy.

Robotics & Simulation

Core: ROS (Robot Operating System), Gazebo, RViz.

Algorithms: SLAM (GMapping), Path Planning (RRT*), Inverse Kinematics.

Simulation: Unity3D (PhysX, NavMesh).

Web Development & Cloud

Frontend: React.js, Tailwind CSS.

Backend: Node.js, Express.js, Flask, REST APIs, gRPC.

Cloud: Google Cloud Platform (GCP), Vercel, Supabase.

Database: PostgreSQL, pgvector.

MLOps & Tools

Deployment: Docker, NVIDIA Triton Inference Server.

DevOps: CI/CD, Git, GitHub.

Tools: VS Code, Postman.

## 5. Timeline / Experience
- **Education**: Degree, Institution, Graduation Year.
University of California, Riverside Riverside, CA
Master of Science in Robotics – GPA: 3.97/4.0  Sep 2024 - Dec 2025
Saveetha Engineering College Chennai, India
Bachelor of Technology in AI & Data Science – CGPA: 9.13/10.0 Aug 2020 – May 2024

- **Work Experience**: Role, Company, Duration, Key Responsibilities.
- AI Engineer Intern – India Literacy Project (ILP), Remote Jun 2025 – Nov 2025

One-Line Summary A GenAI-powered career guidance platform designed for rural India, featuring custom speech-to-text engines for local dialects and teacher-in-the-loop verification.

The Challenge Rural students in India often lack access to professional career counseling. Existing tools are English-centric and fail to understand local accents, leaving students disconnected. The goal was to build a system that allows students to reflect on their dreams in their native languages (Kannada/Tamil) while giving teachers oversight tools to validate the AI’s advice.

The Solution I architected a full-stack "Career Journey" ecosystem. The platform allows students to speak naturally in their native tongue. It uses a custom Phonetic Restoration Engine to correct accent errors before processing the input via Google Gemini. The system builds a structured, longitudinal profile of the student's aspirations, which is then summarized by AI using context-aware prompt engineering and routed to a Teacher Dashboard for final human validation before being shown to the student.

Technical Highlights (The "Portfolio Gold")

Long-Context AI Personalization: Instead of a complex RAG pipeline, I leveraged Gemini's 1M+ token window to inject the student's entire longitudinal history (Assessments, Dreams, Hobbies) directly into the prompt context. This ensures the AI has "perfect memory" of the student's journey without the data loss or latency often associated with vector retrieval.

Hybrid Search RAG Pipeline: Instead of a generic vector DB, I utilized Supabase pgvector to perform Hybrid Search. This combines semantic vector similarity (via Gemini text-embedding-004) with SQL metadata filters (Student Class/Location), ensuring advice is not just "smart" but culturally and logistically relevant.

Custom Phonetic Restoration Engine: Standard STT APIs failed on rural Indian accents (e.g., transcribing 'vhat' instead of 'what'). I built a post-processing layer in speechToTextService.ts with a curated mapping dictionary and a 3-layer fallback system (Google STT -> Azure -> Gemini Flash) to handle code-switching and heavy accents.

Human-in-the-Loop Workflow: To solve the hallucination/safety risk, I designed a "Teacher Verification Layer." AI summaries are flagged for review and must be explicitly "Approved" or "Rejected" by a teacher in the dashboard, ensuring 100% safety for the K-12 demographic.

Tech Stack

GenAI: Google Gemini API (Long Context Window), Prompt Engineering (Context Injection)

Database: Supabase (PostgreSQL)

Frontend: React, TypeScript, TanStack Query (State Management)

Performance: Optimized API calls by 40% using aggressive caching strategies.

Architecture Diagram

[Student Voice Input] --> [Phonetic Restoration Engine] --> [Text Correction] --> [Supabase (Structured History)] --> [Gemini LLM (Full Context Injection)] --> [Teacher Verification Dashboard] --> [Final Career Path]

Experience Entry: AI Research Intern @ PURPLESPOT
Role: AI Research Intern Dates: Jan 2024 – May 2024 Domain: Logistics & Natural Language Processing (NLP)

The Challenge PurpleSpot relied on dozens of high-activity WhatsApp groups to find logistics leads. Agents had to manually read thousands of messages to spot truck availability (e.g., "Need 10-ton truck Delhi to Mumbai tmrw"). The data was highly unstructured, filled with spelling errors ("Dlhi", "tmrw"), and lacked any standard format, making traditional Regex parsing impossible (0% success rate on edge cases).

The Solution I engineered an automated NLP Lead Extraction Pipeline that monitors social channels and converts unstructured text into structured SQL database entries.

Technology Shift: Conducted a cost-benefit analysis between Regex-based parsing and LLMs. Determined that rigid rules failed on human-generated text, leading to the implementation of a Cost-Efficient LLM (e.g., Gemini/GPT) optimized for entity extraction.

Robust Entity Recognition: Designed prompt engineering strategies to accurately identify 4 key entities—Origin, Destination, Truck Size, Date—even in the presence of typos, slang, and mixed languages (Code-mixing).

Impact: Automated the scraping and structuring of leads from multiple WhatsApp groups, transforming a manual, error-prone workflow into a real-time lead generation engine.

Resume Bullet Points (Copy-Paste)
Automated Lead Generation: Engineered an NLP pipeline to scrape and structure logistics data (Origin, Destination, Truck Size) from high-volume WhatsApp groups, replacing manual data entry.

Unstructured Data Processing: Overcame the limitations of Regex on noisy, human-generated text (typos, slang) by implementing a cost-effective Large Language Model (LLM) solution for robust Entity Extraction (NER).

Pipeline Optimization: Benchmarked various LLMs for cost-vs-accuracy, delivering a solution that handled non-standard formats with high precision.

Portfolio "Mini-Project" Card (Optional)
If you want to feature this on your "Projects" grid as well:

WhatsApp Logistics Parser

Problem: Converting messy WhatsApp chat logs into structured sales leads.

Solution: An LLM-based extraction engine that handles spelling mistakes and slang.

Tech: Python, LLMs, Prompt Engineering, WhatsApp Automation.

Software Developer Intern @ DGTEL Rehumanice
Role: Software Developer Intern Dates: Jan 2023 – Dec 2023 Domain: EdTech & Computer Vision (On-Device ML)

The Challenge The company was building a language learning app for children to practice writing Tamil. Unlike English (26 characters), Tamil has 247 characters with complex compound letters, making recognition difficult. Furthermore, standard OCR models failed on the erratic, unrefined handwriting of children, and there were no public datasets available for this specific demographic.

The Solution I engineered an end-to-end On-Device Handwritten Character Recognition System tailored for children.

Data Engineering: I took ownership of the data scarcity problem by designing a collection protocol and manually curating a proprietary dataset of Tamil characters written by children.

Model Architecture: Designed and trained a Custom CNN (Convolutional Neural Network) specifically optimized for the structural nuances of Tamil script, balancing high accuracy with low computational complexity.

Edge Deployment: Instead of a slow cloud API, I converted the model to TensorFlow Lite (TFLite) and integrated it directly into the mobile app. This enabled offline, real-time feedback (<50ms latency), allowing children to practice writing without an internet connection.

Resume Bullet Points (Copy-Paste)
End-to-End Edge AI System: Built and deployed a real-time Tamil Handwritten Character Recognition model using TensorFlow Lite, enabling offline usage for a children's educational app.

Custom Dataset Curation: Overcame data scarcity by collecting and preprocessing a custom dataset of Tamil characters specifically from children to handle erratic handwriting styles.

Mobile Model Optimization: Architected a lightweight Custom CNN achieving high accuracy on 247 distinct Tamil characters while maintaining low latency on mobile devices.

Portfolio "Mini-Project" Card (Optional)
Tamil EdTech OCR Engine

Problem: Helping children learn the complex 247-character Tamil script.

Solution: An offline AI model that recognizes messy handwriting instantly.

Tech: Python, TensorFlow, TFLite, Custom CNN.

- **Certifications**: 
inside images/certificate

## 6. Resume
- Do you have a PDF resume you want to link to?
