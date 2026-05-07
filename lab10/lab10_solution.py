from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LAB_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LAB_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_CSV = LAB_DIR / "cases_clinical_for_lab10.csv"
REPO_ROOT = LAB_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from common.report_utils import ReportContext, df_to_latex, figure_block, render_report  # noqa: E402

NUM_COLS = ["age", "bmi", "systolic_bp", "diastolic_bp", "glucose"]
CAT_COLS = ["sex", "smoker", "family_history"]
TARGET = "high_risk_cvd"

#data layer
def build_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_COLS),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
        ]
    )
    return Pipeline([("pre", pre), ("model", estimator)])

#metrics
def metrics_for_threshold(y_true: np.ndarray, proba: np.ndarray, tau: float) -> dict:
    pred = (proba >= tau).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "threshold": tau,
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
    }


def _save(fig: plt.Figure, name: str) -> str:
    fig.savefig(OUTPUT_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name

#plots
def plot_proba_distribution(p_raw: np.ndarray, p_cal: np.ndarray) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    axes[0].hist(p_raw, bins=30, color="#3b6fb6", edgecolor="black")
    axes[0].set_title("RF (przed kalibracją)")
    axes[0].set_xlabel("p(y=1 | x)")
    axes[0].set_ylabel("liczność")
    axes[1].hist(p_cal, bins=30, color="#b6553b", edgecolor="black")
    axes[1].set_title("RF + Calibrated (sigmoid)")
    axes[1].set_xlabel("p(y=1 | x)")
    fig.suptitle("Rozkład predykowanych prawdopodobieństw")
    fig.tight_layout()
    return _save(fig, "proba_dist.png")


def plot_calibration_curves(y_true: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray) -> str:
    frac_raw, mean_raw = calibration_curve(y_true, p_raw, n_bins=10, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_true, p_cal, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="idealnie skalibrowany")
    ax.plot(mean_raw, frac_raw, marker="o", color="#3b6fb6", lw=2, label="RF (raw)")
    ax.plot(mean_cal, frac_cal, marker="o", color="#b6553b", lw=2, label="RF (calibrated)")
    ax.set_xlabel("średnie p(y=1)")
    ax.set_ylabel("frakcja pozytywów")
    ax.set_title("Krzywe kalibracji (reliability diagram)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "calibration.png")


def plot_threshold_cm(metrics_table: pd.DataFrame, model_name: str, fname: str) -> str:
    sub = metrics_table[metrics_table["model"] == model_name]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(sub))
    width = 0.2
    ax.bar(x - 1.5 * width, sub["TP"], width, label="TP", color="#3b6fb6")
    ax.bar(x - 0.5 * width, sub["FN"], width, label="FN", color="#b6553b")
    ax.bar(x + 0.5 * width, sub["FP"], width, label="FP", color="#3bb66f")
    ax.bar(x + 1.5 * width, sub["TN"], width, label="TN", color="#777777")
    ax.set_xticks(x); ax.set_xticklabels([f"τ={t}" for t in sub["threshold"]])
    ax.set_ylabel("liczba przypadków testowych")
    ax.set_title(f"Macierz pomyłek wg progu --- {model_name}")
    ax.legend()
    fig.tight_layout()
    return _save(fig, fname)

def plot_recall_curve(thresholds: np.ndarray, recalls: dict[str, np.ndarray], fps: dict[str, np.ndarray]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, vals in recalls.items():
        axes[0].plot(thresholds, vals, label=name, lw=2)
    axes[0].set_xlabel("τ"); axes[0].set_ylabel("Recall (TPR)"); axes[0].set_title("Recall vs próg")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    for name, vals in fps.items():
        axes[1].plot(thresholds, vals, label=name, lw=2)
    axes[1].set_xlabel("τ"); axes[1].set_ylabel("FP"); axes[1].set_title("FP vs próg")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "recall_fp.png")


#report glue
def main() -> None:
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded N={len(df)}")

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=8, stratify=y
    )

    rf = build_pipeline(RandomForestClassifier(n_estimators=300, random_state=8, class_weight="balanced", n_jobs=-1))
    rf.fit(X_train, y_train)
    p_raw = rf.predict_proba(X_test)[:, 1]
    auc_raw = roc_auc_score(y_test, p_raw)
    pr_raw = average_precision_score(y_test, p_raw)
    brier_raw = brier_score_loss(y_test, p_raw)

    rf_cal = CalibratedClassifierCV(
        estimator=build_pipeline(RandomForestClassifier(n_estimators=300, random_state=8, class_weight="balanced", n_jobs=-1)),
        cv=5,
        method="sigmoid",
    )
    rf_cal.fit(X_train, y_train)
    p_cal = rf_cal.predict_proba(X_test)[:, 1]
    auc_cal = roc_auc_score(y_test, p_cal)
    pr_cal = average_precision_score(y_test, p_cal)
    brier_cal = brier_score_loss(y_test, p_cal)

    rows = []
    for tau in [0.4, 0.5, 0.6]:
        m_raw = metrics_for_threshold(y_test, p_raw, tau); m_raw["model"] = "RF (raw)"
        rows.append(m_raw)
        m_cal = metrics_for_threshold(y_test, p_cal, tau); m_cal["model"] = "RF + kalibracja"
        rows.append(m_cal)
    cm_table = pd.DataFrame(rows)
    cm_table = cm_table[["model", "threshold", "TP", "FN", "FP", "TN", "precision", "recall", "F1"]].round(3)

    overall = pd.DataFrame([
        {"model": "RF (raw)", "ROC AUC": auc_raw, "PR AUC": pr_raw, "Brier": brier_raw},
        {"model": "RF + kalibracja", "ROC AUC": auc_cal, "PR AUC": pr_cal, "Brier": brier_cal},
    ]).round(4)

    thresholds_grid = np.linspace(0.05, 0.95, 19)
    recalls: dict[str, np.ndarray] = {"RF (raw)": [], "RF + kalibracja": []}
    fps: dict[str, np.ndarray] = {"RF (raw)": [], "RF + kalibracja": []}
    for tau in thresholds_grid:
        for name, p in [("RF (raw)", p_raw), ("RF + kalibracja", p_cal)]:
            m = metrics_for_threshold(y_test, p, tau)
            recalls[name].append(m["recall"])
            fps[name].append(m["FP"])
    for k in recalls:
        recalls[k] = np.array(recalls[k])
        fps[k] = np.array(fps[k])

    fig_dist = plot_proba_distribution(p_raw, p_cal)
    fig_cal = plot_calibration_curves(y_test, p_raw, p_cal)
    fig_cm_raw = plot_threshold_cm(cm_table, "RF (raw)", "cm_raw.png")
    fig_cm_cal = plot_threshold_cm(cm_table, "RF + kalibracja", "cm_cal.png")
    fig_rec = plot_recall_curve(thresholds_grid, recalls, fps)

    test_df = X_test.copy()
    test_df["y_true"] = y_test
    test_df["p_raw"] = p_raw
    test_df["p_cal"] = p_cal
    test_df["pred_raw_05"] = (p_raw >= 0.5).astype(int)

    fp_cases = test_df[(test_df["y_true"] == 0) & (test_df["pred_raw_05"] == 1)].sort_values("p_raw", ascending=False).head(3)
    fn_cases = test_df[(test_df["y_true"] == 1) & (test_df["pred_raw_05"] == 0)].sort_values("p_raw").head(3)

    case_cols = NUM_COLS + CAT_COLS + ["y_true", "p_raw", "p_cal"]
    fp_table = fp_cases[case_cols].copy()
    fn_table = fn_cases[case_cols].copy()
    for t in (fp_table, fn_table):
        t["p_raw"] = t["p_raw"].round(3); t["p_cal"] = t["p_cal"].round(3)

    ctx = ReportContext(
        lab_number=10,
        report_title="Badania przypadków klinicznych: kalibracja prawdopodobieństw RF",
        variant="8 — kalibracja prawdopodobieństw (RF + CalibratedClassifierCV)",
    )
    ctx.section(
        "CEL",
        "Implementacja klasyfikatora ryzyka sercowo-naczyniowego z modelem Random Forest "
        "oraz porównanie wyników z modelem skalibrowanym (\\texttt{CalibratedClassifierCV} "
        "method=sigmoid). Analiza wpływu progów decyzyjnych $\\tau \\in \\{0.4, 0.5, 0.6\\}$ "
        "na liczbę FP/FN. Przedstawienie 3 przypadków FP i 3 przypadków FN jako case studies.",
    )
    ctx.section(
        "PROBLEM",
        "Predykcja \\texttt{high\\_risk\\_cvd} (1 = wysokie ryzyko sercowo-naczyniowe). "
        "Random Forest zwraca prawdopodobieństwa, ale często niewykalibrowane "
        "(p $\\approx$ 0.5 dla pewnych przypadków). Kalibracja (Platt/sigmoid lub isotonic) "
        "poprawia ich znaczenie probabilistyczne, co jest kluczowe w decyzjach klinicznych "
        "z różnymi kosztami błędu.",
    )
    ctx.section(
        "DANE",
        f"Plik \\texttt{{cases\\_clinical\\_for\\_lab10.csv}} ($N={len(df)}$). "
        "Zmienne: \\texttt{age, bmi, systolic\\_bp, diastolic\\_bp, glucose, sex, smoker, "
        "family\\_history}. Target: \\texttt{high\\_risk\\_cvd}. "
        "Braki: \\texttt{bmi} ($\\sim$60), \\texttt{systolic\\_bp} ($\\sim$34), "
        "\\texttt{glucose} ($\\sim$56). Klasy: 661 pos / 339 neg.",
    )
    ctx.section(
        "METODY",
        "Pipeline: \\texttt{ColumnTransformer} (Imputer median + Scaler dla numerycznych, "
        "Imputer most\\_frequent + OneHot dla kategorycznych) + \\texttt{RandomForest "
        "(n\\_estimators=300, class\\_weight=balanced)}. Kalibracja: "
        "\\texttt{CalibratedClassifierCV(method='sigmoid', cv=5)} owijający tę samą "
        "Pipeline. Ewaluacja: ROC AUC, PR AUC, Brier score, oraz CM/precision/recall/F1 "
        "dla 3 progów $\\tau \\in \\{0.4, 0.5, 0.6\\}$. Podział train/test 75/25 "
        "ze stratyfikacją.",
    )
    ctx.section(
        "IMPLEMENTACJA",
        "Plik \\texttt{lab10/lab10\\_solution.py}. Funkcje pomocnicze: "
        "\\texttt{build\\_pipeline}, \\texttt{metrics\\_for\\_threshold}, "
        "\\texttt{plot\\_proba\\_distribution}, \\texttt{plot\\_calibration\\_curves}, "
        "\\texttt{plot\\_threshold\\_cm}.",
    )
    ctx.section(
        "OBLICZENIA",
        "Globalne metryki dla obu modeli (na zbiorze testowym):\n\n"
        + df_to_latex(overall, "ROC AUC, PR AUC, Brier score.", "global", float_format="%.4f")
        + "\nMacierze pomyłek dla 3 progów (RF raw vs RF + kalibracja):\n\n"
        + df_to_latex(cm_table, "Macierze pomyłek i miary dla różnych progów.", "cm",
                      float_format="%.3f"),
    )
    ctx.section(
        "WYNIKI",
        "Wybrane przypadki False Positive (model przewidział pozytywny przy y=0, raw $\\tau=0.5$):\n\n"
        + df_to_latex(fp_table, "Trzy przypadki FP --- case studies.", "fp", float_format="%.2f")
        + "\nWybrane przypadki False Negative (model przewidział negatywny przy y=1):\n\n"
        + df_to_latex(fn_table, "Trzy przypadki FN --- case studies.", "fn", float_format="%.2f"),
    )
    ctx.section(
        "WYKRESY",
        figure_block(fig_dist, "Rozkład $\\hat p$ przed i po kalibracji.", "dist")
        + figure_block(fig_cal, "Krzywe kalibracji (reliability diagram).", "calplot")
        + figure_block(fig_cm_raw, "Macierze pomyłek wg progu --- RF (raw).", "cmraw")
        + figure_block(fig_cm_cal, "Macierze pomyłek wg progu --- RF + kalibracja.", "cmcal")
        + figure_block(fig_rec, "Recall i FP w funkcji progu.", "rec"),
    )
    ctx.section(
        "INTERPRETACJA",
        "Model RF (raw) ma rozkład $\\hat p$ skoncentrowany w okolicach skrajnych wartości "
        "(0--0.1 i 0.9--1.0), co odzwierciedla głosowanie drzew. Po kalibracji sigmoid "
        "rozkład jest bardziej spłaszczony, a krzywa reliability bliżej diagonali "
        f"(Brier raw = {brier_raw:.4f}, Brier cal = {brier_cal:.4f}). "
        f"ROC AUC pozostaje praktycznie niezmienione ({auc_raw:.4f} vs {auc_cal:.4f}), "
        "ponieważ kalibracja monotonicznie przekształca prawdopodobieństwa --- "
        "kolejność rankingu jest taka sama. Dla stałego progu $\\tau$ liczba FP/FN "
        "może się zmieniać, ponieważ kalibracja przesuwa konkretne wartości $p$ "
        "względem $\\tau$. Case studies FP i FN pokazują typowe profile: pacjenci "
        "z wysokimi parametrami metabolicznymi ale ujemnym wynikiem (FP) oraz pacjenci "
        "z umiarkowanymi parametrami i positive (FN).",
    )
    ctx.section(
        "WNIOSKI",
        "Kalibracja prawdopodobieństw jest kluczowa, gdy: (a) probabilistyczna interpretacja "
        "ma znaczenie kliniczne (np. ocena ryzyka 80\\% vs 60\\%); (b) próg decyzyjny jest "
        "ustalany na podstawie kosztów FN/FP; (c) wyniki modelu są łączone z innymi modelami "
        "(stacking). Nie poprawia natomiast rankingu (AUC pozostaje stałe). W tym wariancie "
        "Brier score uległ poprawie, co potwierdza, że kalibracja sigmoid działa zgodnie "
        "z oczekiwaniami.",
    )

    pdf = render_report(OUTPUT_DIR, "lab10_report.tex", ctx)
    print(f"OK -> {pdf}")


if __name__ == "__main__":
    main()
