Absolutely — here’s your **enhanced, polished, and competition-ready README.md** with improved formatting, better visual hierarchy, and clearer section separation. You can directly copy this into your `README.md`.


# 🚀 COMSYS Hackathon-5 2025  

**Robust Face Recognition & Gender Classification under Adverse Visual Conditions**

---

## 🏆 Overview  

This repository contains our complete solution for **COMSYS Hackathon-5 2025**, focusing on *robust face recognition* and *gender classification* under challenging visual conditions (blur, fog, rain, low-light, overexposure, etc.).

✅ **Task A:** Gender Classification (Male/Female, binary)  
✅ **Task B:** Face Recognition (Identity Matching, multi-class)  
✅ **Dataset:** FACECOM (faces under adverse conditions)  
✅ **Scoring:** 30% Task A + 70% Task B (final score = weighted sum)

---

## 📂 Repository Structure  

```plaintext
COMSYS-Hackathon-5-2025/
├── config/                  # Config files (YAML)
├── data/                    # Dataset (not included)
├── docs/                    # Architecture diagram & technical summary
├── models/                  # Saved model checkpoints
├── results/                 # Evaluation outputs (CSV, JSON)
├── scripts/                 # Training and evaluation scripts
├── src/                     # Source code (models, data loaders, etc.)
├── tests/                   # Unit/integration tests
├── requirements.txt         # Python dependencies
├── setup.py                 # Python package setup
└── README.md                # This file
````

---

## ⚡ Quick Start

### 🔹 1️⃣ Install Dependencies


git clone https://github.com/Debsmit16/COMSYS-Hackathon-5-2025.git
cd COMSYS-Hackathon-5-2025
pip install -r requirements.txt
```

### 🔹 2️⃣ Prepare Dataset

* Place your **training** and **validation** data under:

  ```
  data/processed/train/
  data/processed/val/
  ```
* Place **test** data under:

  ```
  data/processed/test/
  ```

📌 **Example Structure:**

```plaintext
data/processed/train/
    male/
    female/
    person_001/
    person_002/
    ...
data/processed/val/
    ...
data/processed/test/
    male/
    female/
    distorted/
    person_001/
    person_002/
    ...
```

### 🔹 3️⃣ Train Models


python scripts/train_gender_classifier.py
python scripts/train_face_matcher.py
```

### 🔹 4️⃣ Evaluate


python scripts/evaluate_model.py --test_path data/processed/test
```

➡️ Results (CSV, JSON) will appear in the `results/` folder.

---

## 🧠 Solution Approach

### ✨ Gender Classification (Task A)

* **Model:** EfficientNet-B3 + custom dense layers
* **Preprocessing:** CLAHE, gamma correction, bilateral filtering, face alignment
* **Augmentation:** Flip, rotation, brightness/contrast, fog, rain, blur
* **Loss:** Binary crossentropy

### ✨ Face Recognition (Task B)

* **Model:** Siamese network (ResNet-50 backbone), triplet loss
* **Embedding:** 128-D L2-normalized vectors
* **Matching:** Euclidean distance + threshold

### 💡 Innovations

* Adversarial preprocessing (for adverse conditions)
* Test-time augmentation for robustness
* Weighted ensemble for optimal accuracy

---

## 📊 Results

| **Task**                 | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
| ------------------------ | :----------: | :-----------: | :--------: | :----------: |
| Gender Classification    |     0.92     |      0.93     |    0.92    |     0.91     |
| Face Recognition         |     0.88     |      0.88     |    0.88    |     0.88     |
| **Final Weighted Score** |   **0.89**   |       —       |      —     |       —      |

---

## 📄 Documentation

* 📌 **Architecture Diagram:** [`docs/model_architecture.png`](docs/model_architecture.png)
* 📌 **Technical Summary:** [`docs/technical_summary.pdf`](docs/technical_summary.pdf)

---

## 🧪 Testing

Run all tests:


pytest tests/ -q
```

---

## 🛠️ Troubleshooting

⚠️ **Common issues:**

* Check dataset structure and Python dependencies.
* Verify GPU drivers / CUDA for model training.
* Open an issue or email us for help.

---

## 👥 Team

* **Team Leader:** Debsmit Ghosh
* **Team Members:** Ujan Das, Anuksha Ganguly
* **Affiliation:** Techno International Newtown
* **Contact:** [ghosh.debsmit1611@gmail.com](mailto:ghosh.debsmit1611@gmail.com)

---

## 📜 License

This project is developed for **COMSYS Hackathon-5 2025** and is intended for *educational and research use only*.

---

```

