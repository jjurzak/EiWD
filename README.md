# 🧠 IRwZM — Inteligentne rozwiązania w zagadnieniach medycznych (Smart solutions for medical challenges)

> Lab solutions and academic reports for the **IRwZM** course (*Smart solutions for medical challenges*) — applied AI for medical decision support 🩺

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.51-7F52FF)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/LIME-0.2-1F77B4)](https://github.com/marcotcr/lime)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LaTeX](https://img.shields.io/badge/LaTeX-PDF-008080?logo=latex&logoColor=white)](https://www.latex-project.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)](https://github.com/astral-sh/uv)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red)](#)

---

## 📚 About

This repository contains **10 laboratory solutions** for the IRwZM university course, covering the full pipeline of **AI-driven medical decision support** — from raw data ingestion, through ML/RL modelling, to expert systems and clinical case studies.

Each lab includes:

- 🐍 a **Python solution script** (`labN_solution.py`) — fully reproducible
- 📊 **figures, tables, and intermediate artifacts** in `labN/output/`
- 📄 a **professional academic report** in Polish (`labN_report.pdf` + `labN_report.tex`)
- 📋 the original **assignment PDF** and reference notebook from the lecturer

---

## 🗂️ Repository structure

```
IRWZM/
├── lab1/   — 📥 Medical data import, preprocessing, statistical analysis
├── lab2/   — 📈 Logistic regression on clinical data
├── lab3/   — 📝 NLP I (PL): Medical NER + document classification
├── lab4/   — 🌐 NLP II (EN): NER + classifier comparison (LR vs SVC vs NB)
├── lab5/   — 🔍 XAI: Permutation Importance, PDP/ICE, SHAP, LIME
├── lab6/   — 🔐 Medical data security: encryption, hashing, RBAC, integrity
├── lab7/   — ⚖️ Rule-based expert system (crisp + fuzzy + explainable)
├── lab8/   — 📱 Streamlit health-monitoring app (anonymous vs personalized)
├── lab9/   — 🤖 Reinforcement Learning: MDP + Q-learning + Fitted Q (DQN-style)
└── lab10/  — 🩺 Clinical case studies: Random Forest + probability calibration
```

Each folder shares the same layout:

```
labN/
├── labN.pdf              ← assignment instructions
├── labN.ipynb / .py      ← lecturer's reference code
├── labN_solution.py      ← solution script
└── output/
    ├── labN_report.pdf   ← final academic report
    ├── labN_report.tex   ← LaTeX source
    ├── *.png             ← generated figures
    └── *.csv (as it was) ← intermediate data
```

---

## 🧪 Lab catalogue

| # | Topic | Highlights |
|---|-------|------------|
| 1️⃣ | Medical data preprocessing | Synthetic prostate-cancer cohort, EDA, baseline LR |
| 2️⃣ | Classification on clinical data | LogReg + ROC, PR, confusion matrix, coefficient analysis |
| 3️⃣ | NLP — Polish corpus | TF-IDF + dictionary NER + LogReg/SVC |
| 4️⃣ | NLP — English corpus | TF-IDF + entity highlighting + 3-way classifier comparison |
| 5️⃣ | Explainable AI | PI, PDP, ICE, SHAP TreeExplainer, LIME tabular |
| 6️⃣ | Data security | Fernet/AES, SHA family benchmark, RBAC + audit, anonymization impact |
| 7️⃣ | Expert systems | Crisp + fuzzy (Mamdani) rules, geriatric patient analysis, XAI |
| 8️⃣ | Mobile health monitor | Streamlit app (`app_v8.py`), global vs per-user models |
| 9️⃣ | Reinforcement Learning | Infection-treatment MDP, Value Iteration, Q-learning, FQI |
| 🔟 | Clinical case studies | RF + `CalibratedClassifierCV`, threshold sweep, FP/FN case analysis |

---

## 🚀 Quick start

### 1. Clone

```bash
git clone https://github.com/jjurzak/EiWD.git
cd EiWD
```

### 2. Set up the environment (with [uv](https://github.com/astral-sh/uv))

```bash
uv venv
uv pip install numpy pandas matplotlib scipy scikit-learn sympy networkx \
               jupyter pylatex shap lime cryptography joblib streamlit
```

### 3. Install LaTeX (for report generation)

> 💡 Recommended: **TinyTeX** (lightweight, cross-platform).

```bash
# Required packages once TeX is installed
tlmgr install mathtools polski hyphen-polish lm latex-bin amsmath \
              booktabs hyperref fancyhdr microtype caption setspace \
              parskip listings xcolor float multirow geometry babel-polish bm
```

### 4. Run any lab

```bash
# Example: Lab 7 — fuzzy expert system for geriatric patients
python lab7/lab7_solution.py
```

The script will:
- run the analysis,
- save figures and CSVs to `lab7/output/`,
- compile a polished PDF report at `lab7/output/lab7_report.pdf` 📄

### 5. Launch the Streamlit app (Lab 8)

```bash
python -m streamlit run lab8/app_v8.py
```

Then open the URL printed in the terminal (typically <http://localhost:8501>) on your laptop or phone 📲.

---

## 📊 Tech stack

| Layer | Tools |
|-------|-------|
| 🧮 Numerics | `numpy`, `pandas`, `scipy` |
| 🤖 ML | `scikit-learn`, `joblib` |
| 🧠 XAI | `shap`, `lime`, `sklearn.inspection` |
| 🔐 Security | `cryptography` (Fernet/AES), `hashlib` (SHA-256/3, BLAKE2) |
| 📊 Visualization | `matplotlib` |
| 📱 UI | `streamlit` |
| 📝 Reports | LaTeX (`pdflatex`) + custom Python templating engine in `common/` |
| 📦 Env | [`uv`](https://github.com/astral-sh/uv) for fast dependency management |

---

## 🎓 Report structure

Every report follows the same academic skeleton:

1. 🎯 **Cel** (Goal)
2. 🩺 **Opis problemu** (Problem statement)
3. 📦 **Opis danych** (Data description)
4. 🛠️ **Opis metod** (Methods)
5. 🧱 **Opis implementacji** (Implementation)
6. 🧮 **Obliczenia** (Computations)
7. 📈 **Wyniki** (Results)
8. 🖼️ **Wykresy** (Figures)
9. 🔎 **Interpretacja** (Interpretation)
10. ✅ **Wnioski** (Conclusions)

> Built from a single shared LaTeX template + placeholder engine, so all 10 reports look consistent.

---

## ⚠️ Disclaimer

> 🚨 All lab solutions are for **educational purposes only** and do **not** constitute medical advice, diagnostic tools, or clinical decision-support software. Synthetic data is used wherever real datasets weren't available.

---

## 📜 License

Released under the **MIT License** — feel free to learn from, adapt, and extend.

---

## 👤 Author

**Jakub Jurzak** ·  
🎓 IRWZM course assignments, 2025/2026

[![GitHub](https://img.shields.io/badge/GitHub-jjurzak-181717?logo=github)](https://github.com/jjurzak)

---

<p align="center">
  Made with ☕, 🐍, and a lot of <code>pdflatex</code>.
</p>
