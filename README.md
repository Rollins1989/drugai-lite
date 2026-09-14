# DrugAI Lite

### Explainable Computational Drug-Discovery Workspace

DrugAI Lite is a machine-learning based computational drug-discovery workspace for analyzing molecules, predicting molecular properties, evaluating toxicity, and estimating target-specific biological activity.

The project combines **RDKit**, **scikit-learn**, **FastAPI**, and a lightweight web interface into a single workflow.

---

## 🚀 Features

### Molecular Analysis
- SMILES-based molecular analysis
- Molecular descriptors calculated with RDKit
- Molecular fingerprints
- Basic drug-likeness and property analysis

### Solubility Prediction
- Delaney ESOL dataset
- Random Forest and Gradient Boosting models
- Regression evaluation using:
  - R²
  - RMSE
  - MAE
- Random split vs scaffold split comparison

### Toxicity Prediction
- Tox21 NR-AR dataset
- Random Forest and Gradient Boosting classifiers
- Evaluation using:
  - ROC-AUC
  - PR-AUC
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Scaffold-based evaluation to reduce molecular similarity leakage

### Target-Specific Activity Prediction
- Human EGFR activity prediction
- Target: **CHEMBL203**
- IC50 data sourced from ChEMBL
- Activity represented as pIC50
- Random Forest + Gradient Boosting ensemble
- Scaffold-split evaluation
- Predicted IC50 reported in nM

### Web Application
- Interactive browser-based interface
- Molecule analysis
- Molecule screening
- Target-specific activity prediction
- Model evaluation information
- FastAPI backend

---

## 🧠 Machine Learning Approach

DrugAI Lite uses molecular representations generated from SMILES structures.

The workflow is:

```text
SMILES
  ↓
RDKit Molecular Representation
  ↓
Descriptors / Morgan Fingerprints
  ↓
Train / Test Split
  ↓
Random Forest + Gradient Boosting
  ↓
Model Evaluation
  ↓
Prediction
````

For activity and property evaluation, the project uses **Bemis-Murcko scaffold splitting** in addition to conventional random splitting.

Scaffold splitting helps evaluate whether models generalize to chemically different molecular scaffolds rather than simply memorizing highly similar molecules.

---

## 📊 Model Evaluation

### Solubility

| Split    | Ensemble R² |   RMSE |    MAE |
| -------- | ----------: | -----: | -----: |
| Random   |      0.8781 | 0.7589 | 0.5193 |
| Scaffold |      0.8667 | 0.8360 | 0.6053 |

### Tox21 NR-AR

| Split    | Ensemble ROC-AUC | PR-AUC |     F1 |
| -------- | ---------------: | -----: | -----: |
| Random   |           0.7692 | 0.4544 | 0.5106 |
| Scaffold |           0.7758 | 0.4626 | 0.5135 |

> Accuracy is not used as the primary toxicity metric because the dataset contains substantial class imbalance.

### EGFR Activity

| Metric           | Result |
| ---------------- | -----: |
| Molecules        |  3,970 |
| Test molecules   |    799 |
| Scaffold overlap |      0 |
| Ensemble R²      | 0.6733 |
| Ensemble RMSE    | 0.7535 |
| Ensemble MAE     | 0.5884 |

---

## 🧪 Example

Input:

```text
COC1=C(OCCCN2CCCCC2)C=CC(=C1)NC3=NC=CC(=C3)C#N
```

Example EGFR prediction:

```text
Target: EGFR (CHEMBL203)

Predicted pIC50: 5.668
Predicted IC50: 2148.69 nM

Random Forest: 5.519
Gradient Boosting: 5.817
```

These predictions are computational estimates and should not be interpreted as experimental measurements.

---

## 🛠️ Tech Stack

* **Python**
* **RDKit**
* **scikit-learn**
* **pandas**
* **NumPy**
* **joblib**
* **FastAPI**
* **Uvicorn**
* **HTML / CSS / JavaScript**
* **ChEMBL**
* **Git / GitHub**

---

## 📁 Project Structure

```text
drugai/
│
├── backend/
│   ├── main.py
│   ├── train_models.py
│   ├── train_target_activity.py
│   ├── requirements.txt
│   ├── Dockerfile
│   │
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
│
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Rollins1989/drugai-lite.git
cd drugai-lite
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

---

## ▶️ Run the API

Navigate to the backend:

```bash
cd backend
```

Start FastAPI:

```bash
uvicorn main:app
```

The application can then be accessed through the local web server.

Interactive API documentation is available through FastAPI's Swagger interface.

---

## 🔬 Training

### Train property and toxicity models

```bash
python train_models.py
```

### Train the EGFR activity model

```bash
python train_target_activity.py --target CHEMBL203 --name egfr --max-records 5000
```

The EGFR model uses target-specific IC50 measurements and evaluates performance using a scaffold-based train/test split.

---

## ⚠️ Limitations

DrugAI Lite is a **research and learning project**, not a validated clinical or drug-development system.

Important limitations include:

* Predictions are computational estimates.
* Model performance depends on the training data and molecular representation.
* The EGFR model is trained for a specific target rather than being a universal target-activity predictor.
* Predictions should not replace experimental validation.
* Dataset bias and chemical-space limitations can affect generalization.
* Confidence estimates are heuristic and are not experimentally calibrated.

---

## 🎯 Project Goals

DrugAI Lite is designed to explore practical applications of machine learning in computational drug discovery, including:

* Molecular representation
* Chemical property prediction
* Toxicity classification
* Scaffold-aware model evaluation
* Target-specific activity prediction
* ML model serving through APIs
* Building usable scientific software

---

## 📌 Current Version

**v3**

Current focus:

* Scaffold-aware evaluation
* Target-specific activity prediction
* EGFR activity modeling
* FastAPI prediction endpoints
* Interactive prediction interface

---

## 👨‍💻 Author

**Kuldeep**

GitHub:
[https://github.com/Rollins1989/drugai-lite](https://github.com/Rollins1989/drugai-lite)

```