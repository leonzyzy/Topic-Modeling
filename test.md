## **Sequence Assist – CM Spend Forecasting & Acquisition Risk Modeling**

*Explore the potential of transformer-based architectures to model structured and sequential data for outstanding balance prediction and acquisition risk scoring, potentially supporting credit line increase decisions. Below are my core contributions and outcomes:*

---

### **Key Results & Business Impact**

* Successfully scaled the **LitData + PyTorchJob training pipeline** I previously developed to both **CM Spend** and **Acquisition Risk** tracks. All model results in this project were produced using this infrastructure—training at this scale would not have been possible without it.

* Led a targeted experiment in CM Spend to show that a **1-layer or multi-layer FFN model with non-linear activation function** can **reproduce champion model performance using non-interactive static features only** This deepened our understanding of the role of non-linear models in capturing complex relationships and informs future model simplification strategies.

* Replaced inefficient **DataParallel** logic in CM Spend with **Distributed Data Parallel (DDP)** and enabled multi-GPU training via PyTorchJob, significantly improving training speed and iteration time.

* Resolved a critical infrastructure bug by identifying a **LitData AWS credential expiration issue** during long training runs. Collaborated with the team to **monkey-patch the issue**, leading to LitData being internalized as a Capital One package—ensuring stable usage for larger-scale experiments.

* Contacted the author of **Fieldy Transformer** and successfully got an **Apache license** added to the GitHub repository, allowing legal code adoption. This enabled us to incorporate Fieldy into our modeling pipeline without legal concerns.

* Shared the LitData pipeline and experimental framework with **business stakeholders** to support the Sequence Assist roadmap. Presented how the infrastructure works and brainstormed future applications of LitData for sequence modeling.

---

### **Competency Strengths Demonstrated**

* **Model Architecture Expertise**: Contributed to model architecture design by sharing and implementing ideas including **raw encoder**, **skip connections**, **sequence outputs**, **iTransformer**, and **Fieldy Transformer**. 
* **Infrastructure & Debugging Leadership**: Acted as a technical consultant for both the **training pipeline (LitData + PyTorchJob)** and **data pipeline (Spark ETL + KFP)**, assisting colleagues with implementation, debugging, and scalability concerns.

* **Collaboration & Communication**: Worked closely with a large cross-functional team, many for the first time as a new hire. Actively participated in **weekly meetings**, **slide deck preparation**, and **brainstorming sessions** with both technical and business stakeholders—demonstrating collaborative spirit and effective communication.

* **Enablement & Impact Beyond Myself**: Facilitated broader team adoption of new tools and methods—whether by helping debug complex issues or enabling access to external codebases through license resolution.

---

### **Competency Growth & Future Opportunities**

* This project provided a rare opportunity to work across both **CM Spend** and **Acquisition Risk** tracks, accelerating my learning of Capital One’s credit risk business and product space.

* By collaborating with many colleagues for the first time, I’ve built new relationships and established a reputation for being reliable, knowledgeable, and collaborative.

* I'm currently working on **integrating Weights & Biases and Git tagging** within the CM Spend KFP repo to improve experiment tracking and reproducibility—an initiative that will benefit future modeling efforts across teams.

---

Let me know if you'd like to include metrics, figures, or a summary version for slides.
