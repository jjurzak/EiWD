"""Streamlit app for Lab 8 — wariant 8 (anon vs spersonalizowany).

Run with:
    python -m streamlit run lab8/app_v8.py

Functional changes vs `lab8.py`:
- mode toggle: anon | personalized,
- per-user storage when personalized (column `user_id`),
- side-by-side comparison: global model vs per-user (local) model.

Diagnostic CSV is at `lab8/output/health_measurements.csv`.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
DATA_PATH = OUT / "health_measurements.csv"
GLOBAL_MODEL_PATH = OUT / "risk_model_global.joblib"
LOCAL_MODEL_DIR = OUT / "local_models"
LOCAL_MODEL_DIR.mkdir(exist_ok=True)

NUM_COLS = ["age", "bmi", "glucose", "systolic_bp", "diastolic_bp"]


def ensure_data_file() -> None:
    if not DATA_PATH.exists():
        cols = ["timestamp", "user_id", "mode"] + NUM_COLS
        pd.DataFrame(columns=cols).to_csv(DATA_PATH, index=False)


def load_data() -> pd.DataFrame:
    ensure_data_file()
    return pd.read_csv(DATA_PATH)


def append_measurement(row: dict) -> None:
    df = load_data()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)


def make_demo_label(df: pd.DataFrame) -> pd.Series:
    return ((df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)).astype(int)


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                NUM_COLS,
            )
        ]
    )
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def train_global(df: pd.DataFrame) -> tuple[Pipeline | None, dict]:
    if len(df) < 20:
        return None, {"reason": f"za mało danych globalnie (N={len(df)})"}
    y = make_demo_label(df)
    X = df[NUM_COLS]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = _build_pipeline().fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "n_train": len(ytr),
        "n_test": len(yte),
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)) if len(set(yte)) > 1 else None,
        "report": classification_report(yte, pred, zero_division=0),
    }
    joblib.dump({"model": clf, "metrics": metrics}, GLOBAL_MODEL_PATH)
    return clf, metrics


def train_local(df: pd.DataFrame, user_id: str) -> tuple[Pipeline | None, dict]:
    df_u = df[df["user_id"] == user_id]
    if len(df_u) < 20:
        return None, {"reason": f"za mało danych użytkownika (N={len(df_u)})"}
    y = make_demo_label(df_u)
    if len(set(y)) < 2:
        return None, {"reason": "tylko 1 klasa w danych użytkownika"}
    X = df_u[NUM_COLS]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = _build_pipeline().fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "n_train": len(ytr),
        "n_test": len(yte),
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)) if len(set(yte)) > 1 else None,
    }
    path = LOCAL_MODEL_DIR / f"local_{user_id}.joblib"
    joblib.dump({"model": clf, "metrics": metrics}, path)
    return clf, metrics


# --- Streamlit UI -------------------------------------------------------------
st.set_page_config(page_title="Monitor zdrowia (wariant 8)", layout="centered")
st.title("Monitor zdrowia — wariant 8 (anonimowy vs spersonalizowany)")

mode = st.radio("Tryb", ["anonimowy", "spersonalizowany"], horizontal=True)

with st.form("measure"):
    user_id = st.text_input(
        "Identyfikator użytkownika", value="", disabled=(mode == "anonimowy"),
        help="Wymagany w trybie spersonalizowanym",
    )
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Wiek", 18, 110, 40)
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0, 0.1)
        glucose = st.number_input("Glukoza [mg/dl]", 40, 300, 95)
    with c2:
        systolic_bp = st.number_input("SBP", 70, 260, 120)
        diastolic_bp = st.number_input("DBP", 40, 150, 80)

    submitted = st.form_submit_button("Zapisz pomiar")

if submitted:
    if mode == "spersonalizowany" and not user_id.strip():
        st.error("Brak user_id dla trybu spersonalizowanego.")
    else:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user_id": user_id.strip() if mode == "spersonalizowany" else "",
            "mode": mode,
            "age": int(age),
            "bmi": float(bmi),
            "glucose": int(glucose),
            "systolic_bp": int(systolic_bp),
            "diastolic_bp": int(diastolic_bp),
        }
        append_measurement(row)
        st.success("Zapisano.")

df = load_data()
st.write(f"Liczba pomiarów: {len(df)}")
st.dataframe(df.tail(10), use_container_width=True)

st.subheader("Etap 3 — modele")
col_g, col_l = st.columns(2)
with col_g:
    if st.button("Trenuj model globalny"):
        m, metrics = train_global(df)
        if m is None:
            st.error(metrics["reason"])
        else:
            st.success("Globalny model zapisany.")
            st.json(metrics)
with col_l:
    if mode == "spersonalizowany" and user_id.strip():
        if st.button(f"Trenuj model lokalny dla {user_id}"):
            m, metrics = train_local(df, user_id.strip())
            if m is None:
                st.error(metrics["reason"])
            else:
                st.success("Lokalny model zapisany.")
                st.json(metrics)

st.subheader("Etap 4 — predykcja")
X_one = pd.DataFrame(
    [
        {
            "age": int(age),
            "bmi": float(bmi),
            "glucose": int(glucose),
            "systolic_bp": int(systolic_bp),
            "diastolic_bp": int(diastolic_bp),
        }
    ]
)
if GLOBAL_MODEL_PATH.exists():
    g = joblib.load(GLOBAL_MODEL_PATH)["model"]
    p = g.predict_proba(X_one)[0, 1]
    st.write(f"Globalny model: P(podwyższone ryzyko) = **{p:.3f}**")
else:
    st.info("Brak modelu globalnego.")

if mode == "spersonalizowany" and user_id.strip():
    local_path = LOCAL_MODEL_DIR / f"local_{user_id.strip()}.joblib"
    if local_path.exists():
        L = joblib.load(local_path)["model"]
        p = L.predict_proba(X_one)[0, 1]
        st.write(f"Lokalny model ({user_id}): P(podwyższone ryzyko) = **{p:.3f}**")
    else:
        st.info("Brak modelu lokalnego dla tego użytkownika.")
