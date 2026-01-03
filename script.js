// Project Data
const projects = [
    {
        id: "ims-fault",
        title: "IMS Bearing Fault Detection System",
        // One-Line Summary
        summary: "An end-to-end predictive maintenance pipeline capable of detecting machinery anomalies in real-time, deployed via NVIDIA Triton Inference Server.",
        // Image path (URL encoded for safety)
        image: "images/IMS%20Fault/sys%20architecture.png",
        gallery: [
            "images/IMS%20Fault/sys%20architecture.png",
            "images/IMS%20Fault/confusion_matrix.png",
            "images/IMS%20Fault/feature_importance.png",
            "images/IMS%20Fault/signal_plot.png"
        ],
        // Full Details
        challenge: "Bearings are the \"wheels\" of industrial machinery; when they fail, production lines stop. The goal was to build a system that could predict these failures using the NASA IMS Bearing Dataset to detect early fault signatures in high-frequency vibration data.",
        solution: "Engineered a lightweight Machine Learning pipeline processing 8-channel vibration data. Used manual feature engineering (Kurtosis, RMS) with a Random Forest model instead of a \"black-box\" Deep Learning model. Served via NVIDIA Triton for standardized, scalable production deployment.",
        highlights: [
            "<strong>Pragmatic MLOps:</strong> Pivoted to Triton Python Backend to serve native joblib models, gaining dynamic batching and standardized APIs without brittle ONNX conversion.",
            "<strong>Feature Engineering vs. Deep Learning:</strong> Choose Random Forest over CNN to avoid overfitting on scarce failure data. Extracted 40 statistical features for 98% accuracy.",
            "<strong>Containerized Scalability:</strong> Packaged entire runtime in Docker (tritonserver:23.12-py3) for \"write once, run anywhere\" deployment."
        ],
        tech: ["Scikit-learn", "Pandas", "NVIDIA Triton", "Docker", "HTTP/gRPC"],
        metrics: [
            { value: "98%", label: "Validation Accuracy" },
            { value: "<50ms", label: "End-to-End Latency" },
            { value: "40", label: "Features per Sample" }
        ],
        link: "https://github.com/Harshini1331/ims-fault-diagnosis.git"
    },
    {
        id: "medical-ai",
        title: "Medical AI: Pulmonary Health Diagnosis",
        summary: "A comparative deep learning study benchmarking 5 CNN architectures for the multi-class diagnosis of COVID-19, Pneumonia, and Healthy lungs.",
        image: "images/Medical%20AI/sys%20architecture.png",
        gallery: [
            "images/Medical%20AI/sys%20architecture.png",
            "images/Medical%20AI/1.png",
            "images/Medical%20AI/2.png",
            "images/Medical%20AI/3.png",
            "images/Medical%20AI/4.png",
            { type: 'video', src: "images/Medical%20AI/project-II.mp4" }
        ],
        challenge: "Automated diagnosis faces Class Imbalance and Visual Ambiguity (Viral vs. Bacterial Pneumonia). The goal was to determine which architecture could best handle these constraints without bias toward the majority class.",
        solution: "Engineered a training pipeline to evaluate 5 architectures (VGG16, ResNet50, DenseNet121, InceptionV3, Custom CNN). Implemented Generative Data Augmentation and Inverse Class Weights to combat overfitting and bias.",
        highlights: [
            "<strong>Battle of Architectures:</strong> VGG16 (78.56%) outperformed complex models like ResNet50 (60.6%), proving simpler architectures generalize better for low-res medical imaging.",
            "<strong>Handling Class Imbalance:</strong> Dynamic weighting strategy ensured high Recall (0.83) for critical COVID-19 cases.",
            "<strong>Honest Failure Analysis:</strong> Identified limitations in distinguishing Viral vs. Bacterial Pneumonia without higher resolution data."
        ],
        tech: ["TensorFlow", "Keras", "VGG16", "ResNet50", "Transfer Learning"],
        metrics: [
            { value: "78.56%", label: "Top Test Accuracy (VGG16)" },
            { value: "0.83", label: "Recall (Sensitivity) for COVID" },
            { value: "5", label: "Architectures Benchmarked" }
        ],
        link: "https://github.com/Harshini1331/DECODING-PULMONARY-HEALTH.git"
    },
    {
        id: "smarteyes",
        title: "SmartEyes - Real-Time Obstacle Detection",
        summary: "A lightweight, multi-mode object detection system identifying 80+ classes in real-time for autonomous navigation and surveillance.",
        image: "images/smarteyes/sys%20architecture.png",
        gallery: [
            "images/smarteyes/sys%20architecture.png",
            "images/smarteyes/Example2_detectedobstacle.jpg",
            "images/smarteyes/Example2_image.jpg",
            "images/smarteyes/Example3_detectedobstacle.jpg",
            "images/smarteyes/Example3_image.jpg"
        ],
        challenge: "In dynamic environments, latency is fatal. The challenge was to build a vision system with high confidence but without the massive lag of older architectures, running on consumer hardware.",
        solution: "Engineered SmartEyes using YOLOv7. Built a flexible \"Multimode\" pipeline (Live Stream, Video, Static Image) leveraging ELAN architecture for high performance (30+ FPS) without aggressive quantization.",
        highlights: [
            "<strong>SOTA Model Utilization:</strong> leveraged YOLOv7's \"Bag of Freebies\" to achieve higher mAP than YOLOv3 at a fraction of the size (71MB vs 240MB).",
            "<strong>Multimode Input Pipeline:</strong> Modular design handles Camera, Video, and Image modes via CLI arguments.",
            "<strong>Performance Optimization:</strong> Optimized inference loop with custom NMS thresholds to maintain 30 FPS fluidity on GPU."
        ],
        tech: ["YOLOv7", "PyTorch", "OpenCV", "Python"],
        metrics: [
            { value: "30 FPS", label: "Real-time Inference" },
            { value: "71 MB", label: "Model Footprint" },
            { value: "80+", label: "Classes Detected" }
        ],
        link: "https://github.com/Harshini1331/SmartEyes-Obstacle-Detection-System.git"
    },
    {
        id: "cooperative-occupancy",
        title: "Cooperative Occupancy Prediction",
        summary: "A Deep Learning framework enabling robot teams to share LiDAR data and \"see through walls\" using Dual Attention mechanisms.",
        image: "images/Cooperative_Occupancy_Prediction/Untitled%20diagram-2025-12-27-031832.png",
        gallery: [
            "images/Cooperative_Occupancy_Prediction/Untitled%20diagram-2025-12-27-031832.png",
            { type: 'pdf', src: "images/Cooperative_Occupancy_Prediction/Capstone_project.pdf" }
        ],
        challenge: "Single robots have \"blind spots\" in crowded spaces. The goal was to solve this using Cooperative Perception (V2V), treating a fleet as a distributed sensor network.",
        solution: "Developed an Early Fusion framework fusing raw LiDAR from multiple agents. Integrated Dual Attention (Spatial + Channel) into a Recurrent VAE backbone to suppress noise and generate sharp occupancy maps.",
        highlights: [
            "<strong>Dual Attention Innovation:</strong> Spatial Attention (SAM) focuses on dynamic actors; Channel Attention (CAM) suppresses sensor noise/ghosting.",
            "<strong>Uncertainty-Aware:</strong> Outputs Entropy Maps to quantify confidence, allowing safer navigation planning.",
            "<strong>High-Fidelity Simulation:</strong> Built a custom Unity3D environment to collect 360k synchronized frames."
        ],
        tech: ["PyTorch", "Unity3D", "RVAEP", "ConvLSTM"],
        metrics: [
            { value: "11.8%", label: "Reduction in BCE Loss" },
            { value: "360k", label: "Frames Generated" },
            { value: "High", label: "Ghosting Reduction" }
        ],
        link: "#" // No link provided in guide, placeholder
    },
    {
        id: "crop-planner",
        title: "Intelligent Crop Recommendation System",
        summary: "Precision agriculture tool recommending optimal crops based on soil composition with 99.54% accuracy via Naive Bayes.",
        image: "images/crop_rotation_planner/Untitled%20diagram-2025-12-27-031506.png",
        gallery: [
            "images/crop_rotation_planner/Untitled%20diagram-2025-12-27-031506.png",
            "images/crop_rotation_planner/285666004-ab2c3cc2-8cfc-43a1-b092-4c8f94b92d91.png",
            "images/crop_rotation_planner/285666017-d572d5e9-1fe6-4532-81ba-733fd0edaae1.png",
            "images/crop_rotation_planner/285666034-97cb2746-ce83-4da1-bb8c-656e8ff4205e.png",
            "images/crop_rotation_planner/285666089-d31cfdca-d352-44f8-a5e7-e2fd27c2f093.png",
            "images/crop_rotation_planner/285666103-732d85a6-d1ea-4eeb-b577-a9a2f65d2eb2.png"
        ],
        challenge: "Farming carries risk; planting the wrong crop leads to loss. The goal was to democratize Precision Agriculture with an instant, scientific recommendation tool.",
        solution: "Benchmarked 6 ML algorithms (Random Forest, SVM, etc.) on soil data. Gaussian Naive Bayes emerged as the winner. Deployed via Streamlit for an intuitive farmer-friendly interface.",
        highlights: [
            "<strong>The \"Naive Bayes\" Surprise:</strong> NB (99.54%) outperformed Random Forest, proving crop features likely follow Gaussian distributions.",
            "<strong>Exploratory Data Analysis:</strong> Rigorous EDA confirmed feature independence, validating the model choice.",
            "<strong>Full-Stack ML Deployment:</strong> Wrapped the model in Streamlit to bridge the gap between code and end-users."
        ],
        tech: ["Scikit-Learn", "Naive Bayes", "Streamlit", "Pandas"],
        metrics: [
            { value: "99.54%", label: "Test Accuracy" },
            { value: "6", label: "Algorithms Benchmarked" },
            { value: "<1s", label: "Inference Time" }
        ],
        link: "https://github.com/Harshini1331/Crop-Rotation-Planner.git"
    },
    {
        id: "warehouse-robot",
        title: "Autonomous Warehouse Robot",
        summary: "ROS-based autonomous mobile manipulator capable of SLAM, RRT* path planning, and dynamic obstacle avoidance.",
        image: "images/autonomous_warehouse/Untitled%20diagram-2025-12-27-032900.png",
        gallery: [
            "images/autonomous_warehouse/Untitled%20diagram-2025-12-27-032900.png",
            { type: 'pdf', src: "images/autonomous_warehouse/autonomous_robot_nav.pdf" }
        ],
        challenge: "Warehouse automation often lacks flexibility. The goal was a low-cost robot capable of navigating dynamic environments without pre-defined tracks.",
        solution: "Engineered a system using TurtleBot3 + OpenManipulator in ROS/Gazebo. Implemented GMapping for SLAM and RRT* for optimal path planning, with TTC-based safety replanning.",
        highlights: [
            "<strong>RRT* Path Planning:</strong> Implemented asymptotically optimal planning for clutter-filled environments.",
            "<strong>Closed-Loop Navigation:</strong> Designed PID controller for precise waypoint tracking.",
            "<strong>Integrated Manipulation:</strong> Solved Inverse Kinematics for autonomous grasping."
        ],
        tech: ["ROS", "Gazebo", "SLAM", "RRT*", "C++"],
        metrics: [
            { value: "100%", label: "Collision-Free Nav" },
            { value: "High", label: "Mapping Fidelity" },
            { value: "Full", label: "Autonomy Achieved" }
        ],
        link: "#" // No specific link provided
    }
];

// Certifications Data
const certifications = [
    { name: "Oracle Generative AI Professional", file: "certificate/Oracle-Generative AI Professional.pdf" },
    { name: "Oracle Foundations Associate", file: "certificate/Oracle-FoundationsAssociate.pdf" },
    { name: "Data Science - Coursera", file: "certificate/Data Science - Coursera.pdf" },
    { name: "Python & Statistics - Coursera", file: "certificate/Python & Statistics - Coursera.pdf" },
];

// DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    // 1. Render Projects
    const projectsContainer = document.getElementById('projects-container');

    projects.forEach(project => {
        const card = document.createElement('div');
        card.className = 'project-card';
        card.setAttribute('data-id', project.id);

        card.innerHTML = `
            <div class="project-image" style="background-image: url('${project.image}');">
                <div class="project-img-overlay">
                    <span class="btn btn-outline" style="background:rgba(0,0,0,0.8); border:none;">View Details</span>
                </div>
            </div>
            <div class="project-content">
                <h3 class="project-title">${project.title}</h3>
                <p class="project-summary">${project.summary}</p>
                <div class="tech-stack-mini">
                    ${project.tech.slice(0, 3).map(t => `<span class="skill-tag">${t}</span>`).join('')}
                    ${project.tech.length > 3 ? `<span class="skill-tag">+${project.tech.length - 3}</span>` : ''}
                </div>
            </div>
        `;

        card.addEventListener('click', () => openModal(project));
        projectsContainer.appendChild(card);
    });

    // 2. Modal Gallery & Lightbox Logic
    const modal = document.getElementById('project-modal');
    const closeBtn = document.querySelector('.close-modal');

    // Create lightbox elements dynamically
    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox';
    lightbox.innerHTML = `
        <span class="close-lightbox">&times;</span>
        <img class="lightbox-content" id="lightbox-img">
        <div class="lightbox-caption" id="lightbox-caption"></div>
    `;
    document.body.appendChild(lightbox);

    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxCaption = document.getElementById('lightbox-caption');
    const closeLightboxBtn = document.querySelector('.close-lightbox');

    closeLightboxBtn.onclick = () => {
        lightbox.style.display = "none";
    }

    lightbox.onclick = (e) => {
        if (e.target === lightbox) {
            lightbox.style.display = "none";
        }
    }

    function openModal(project) {
        document.getElementById('modal-title').innerText = project.title;
        document.getElementById('modal-desc').innerHTML = `
            <div style="margin-bottom: 2rem;">
                <h4 style="color:var(--text-primary); margin-bottom:0.5rem;">The Challenge</h4>
                <p style="color:var(--text-secondary);">${project.challenge}</p>
            </div>
            <div style="margin-bottom: 2rem;">
                <h4 style="color:var(--text-primary); margin-bottom:0.5rem;">The Solution</h4>
                <p style="color:var(--text-secondary);">${project.solution}</p>
            </div>
            <div class="highlight-box">
                <h4 style="color:var(--accent-primary); margin-bottom:1rem;">Technical Highlights</h4>
                <ul>
                    ${project.highlights.map(h => `<li>${h}</li>`).join('')}
                </ul>
            </div>
            
            <!-- Gallery Rendering -->
            ${project.gallery && project.gallery.length > 0 ? `
            <div class="gallery-container">
                <div class="gallery-title"><i class="fas fa-images"></i> Project Gallery</div>
                <div class="gallery-grid">
                    ${project.gallery.map(item => {
            // Check if item is object (video/pdf) or string (image)
            if (typeof item === 'object') {
                if (item.type === 'video') {
                    return `
                                    <div class="gallery-item video-item" onclick="window.open('${item.src}', '_blank')">
                                        <div class="gallery-video-container">
                                            <video muted loop><source src="${item.src}" type="video/mp4"></video>
                                            <i class="fas fa-play-circle"></i>
                                        </div>
                                    </div>
                                `;
                } else if (item.type === 'pdf') {
                    return `
                                    <div class="gallery-item pdf-item" onclick="window.open('${item.src}', '_blank')">
                                        <div class="gallery-video-container" style="background:#1a1a1a;">
                                            <i class="fas fa-file-pdf"></i>
                                            <span style="position:absolute; bottom:10px; color:#aaa; font-size:0.8rem;">View PDF</span>
                                        </div>
                                    </div>
                                `;
                }
            } else {
                // Regular Image
                return `<div class="gallery-item"><img src="${item}" alt="Gallery Image" class="gallery-img-trigger"></div>`;
            }
        }).join('')}
                </div>
            </div>
            ` : ''}
        `;

        // Header Image
        document.getElementById('modal-header-img').style.backgroundImage = `url('${project.image}')`;

        // Tech Stack
        document.getElementById('modal-tech').innerHTML = project.tech.map(t => `<span class="skill-tag" style="background:rgba(0,240,255,0.1); color:var(--accent-primary);">${t}</span>`).join('');

        // Metrics
        document.getElementById('modal-metrics').innerHTML = project.metrics.map(m => `
            <div class="metric-item">
                <strong>${m.value}</strong>
                <span>${m.label}</span>
            </div>
        `).join('');

        // Link
        const linkBtn = document.getElementById('modal-link');
        if (project.link && project.link !== '#') {
            linkBtn.href = project.link;
            linkBtn.style.display = 'inline-flex';
        } else {
            linkBtn.style.display = 'none';
        }

        // Attach Lightbox Events
        setTimeout(() => {
            const triggers = document.querySelectorAll('.gallery-img-trigger');
            triggers.forEach(img => {
                img.onclick = () => {
                    lightbox.style.display = "block";
                    lightboxImg.src = img.src;
                    lightboxCaption.innerHTML = project.title;
                }
            });
        }, 100);

        modal.style.display = 'block';
        document.body.style.overflow = 'hidden'; // Disable scroll
    }

    closeBtn.onclick = () => {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto'; // Enable scroll
    };

    window.onclick = (event) => {
        if (event.target == modal) {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    };

    // 3. Render Certifications
    const certsContainer = document.getElementById('certs-container');
    if (certsContainer && certifications.length > 0) {
        certifications.forEach(cert => {
            const certItem = document.createElement('div');
            certItem.className = 'cert-item';
            certItem.innerHTML = `
                <div style="font-size: 3rem; color: var(--accent-secondary); margin-bottom: 1rem;">
                    <i class="fas fa-file-pdf"></i>
                </div>
                <h4 style="margin-bottom: 0.5rem;">${cert.name}</h4>
                <a href="${cert.file}" target="_blank" class="btn btn-outline" style="font-size: 0.8rem; padding: 0.5rem 1rem;">View Certificate</a>
            `;
            certsContainer.appendChild(certItem);
        });
    }
});
