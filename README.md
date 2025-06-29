# COMSYS Hackathon-5 2025

**Robust Face Recognition & Gender Classification under Adverse Conditions**

---

## 📌 Repository Overview

- **Task A**: Gender Classification  
- **Task B**: Face Recognition (Face Matching)  
- **Metrics**: Accuracy, Precision, Recall, F1-Score (30% Task A, 70% Task B)

## ⚙️ Setup

```
git clone https://github.com/<you>/COMSYS-Hackathon-5-2025.git
cd COMSYS-Hackathon-5-2025
pip install -r requirements.txt
```

## 🚀 Quick Start

```
# Train (place dataset under data/processed/)
python scripts/train_gender_classifier.py
python scripts/train_face_matcher.py

# Evaluate (place test data under data/processed/test/)
python scripts/evaluate_model.py --test_path data/processed/test
```

## 📂 Structure

- **src/** : Core modules  
- **scripts/** : Entry-point scripts  
- **models/** : Saved checkpoints  
- **results/** : Evaluation CSV & logs  
- **docs/** : Diagrams & summary  
- **tests/** : Unit/integration tests  

## 📈 Results (PLACEHOLDER)

| Task                   | Accuracy | Precision | Recall  | F1-Score |
|------------------------|:--------:|:---------:|:-------:|:--------:|
| Gender Classification  | 0.0000   | 0.0000    | 0.0000  | 0.0000   |
| Face Recognition       | 0.0000   | 0.0000    | 0.0000  | 0.0000   |
| **Final Weighted Score** | **0.0000** | —         | —       | —        |

> **Instructions:** Replace all `0.0000` entries with your actual results after running the evaluation.

## 🗂️ Placeholder Diagrams & Summary

- `docs/model_architecture.png` – insert your own architecture diagram  
- `docs/technical_summary.pdf` – insert your 1-page PDF summary

## 🧪 Testing

```
pytest tests/ -q
```