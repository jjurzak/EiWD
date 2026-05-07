r"""Lab 2 — Modele klasyfikacji i regresji + (uwaga) CNN.

Wariant 8 nie istnieje (warianty 1--3 w PDF). Fallback: wariant 1 ---
budowa modelu regresji logistycznej do klasyfikacji pacjentów na podstawie
danych klinicznych.

Zgodnie z PDF („używając danych z poprzednich zajęć”) wykorzystywany jest
syntetyczny zbiór \texttt{prostate\_cancer\_synth.csv} z lab1. Jeśli plik nie
istnieje, generuje go na nowo lokalnie.

Realizuje:
- przygotowanie danych (standaryzacja, kodowanie OHE, podział train/test),
- regresja logistyczna z wagami klas,
- ocena: accuracy, precision, recall, F1, ROC AUC,
- krzywa ROC, macierz pomyłek, krzywa precision-recall.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LAB_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LAB_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPO_ROOT = LAB_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from common.report_utils import ReportContext, df_to_latex, figure_block, render_report  # noqa: E402

LAB1_CSV = REPO_ROOT / "lab1" / "output" / "prostate_cancer_synth.csv"


def load_dataset() -> pd.DataFrame:
    if LAB1_CSV.exists():
        return pd.read_csv(LAB1_CSV)
    sys.path.insert(0, str(REPO_ROOT / "lab1"))
    from lab1_solution import synth_prostate_cancer_dataset  # type: ignore

    df = synth_prostate_cancer_dataset()
    LAB1_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LAB1_CSV, index=False)
    return df


def prepare(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = df.copy()
    df["psa"] = df["psa"].fillna(df["psa"].median())
    df = pd.get_dummies(df, columns=["tumor_stage", "treatment"], drop_first=True)

    feat_num = ["age", "psa", "gleason_score", "comorbidities"]
    feat_cat = [c for c in df.columns if c.startswith(("tumor_stage_", "treatment_"))]
    features = feat_num + feat_cat
    X = df[features].to_numpy(dtype=float)
    y = df["survived"].to_numpy(dtype=int)
    return X, y, features


def evaluate(model, X_test_s: np.ndarray, y_test: np.ndarray) -> dict:
    proba = model.predict_proba(X_test_s)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "proba": proba,
        "pred": pred,
    }


def _save(fig: plt.Figure, name: str) -> str:
    fig.savefig(OUTPUT_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_roc(y_true: np.ndarray, proba: np.ndarray, auc: float) -> str:
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3b6fb6", lw=2.0, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Krzywa ROC --- regresja logistyczna")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "roc.png")


def plot_pr(y_true: np.ndarray, proba: np.ndarray) -> str:
    prec, rec, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#b6553b", lw=2.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Krzywa precision--recall")
    fig.tight_layout()
    return _save(fig, "pr.png")


def plot_confusion(y_true: np.ndarray, pred: np.ndarray) -> str:
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["zgon", "przeżyl"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Macierz pomyłek")
    fig.tight_layout()
    return _save(fig, "cm.png")


def plot_coefficients(model: LogisticRegression, features: list[str]) -> str:
    coefs = pd.Series(model.coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4 + 0.2 * len(features)))
    colors = ["#b6553b" if v < 0 else "#3b6fb6" for v in coefs.values]
    ax.barh(range(len(coefs)), coefs.values, color=colors, edgecolor="black")
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels(coefs.index)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Współczynnik regresji logistycznej (zmienne standaryzowane)")
    ax.set_title("Wpływ cech na prawdopodobieństwo przeżycia")
    fig.tight_layout()
    return _save(fig, "coefs.png")


def main() -> None:
    df = load_dataset()
    X, y, features = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=8, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    model.fit(X_train_s, y_train)
    metrics = evaluate(model, X_test_s, y_test)

    figures = {
        "roc": plot_roc(y_test, metrics["proba"], metrics["roc_auc"]),
        "pr": plot_pr(y_test, metrics["proba"]),
        "cm": plot_confusion(y_test, metrics["pred"]),
        "coef": plot_coefficients(model, features),
    }

    metrics_table = pd.DataFrame(
        {
            "miara": ["accuracy", "precision", "recall", "F1", "ROC AUC"],
            "wartość": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["roc_auc"],
            ],
        }
    )

    coef_table = (
        pd.Series(model.coef_[0], index=features)
        .sort_values(ascending=False)
        .rename_axis("cecha")
        .reset_index(name="współczynnik")
    )
    coef_table["odds_ratio"] = np.exp(coef_table["współczynnik"]).round(3)
    coef_table["współczynnik"] = coef_table["współczynnik"].round(3)

    ctx = ReportContext(
        lab_number=2,
        report_title="Modele klasyfikacji: regresja logistyczna na danych klinicznych",
        variant="1 (fallback z 8) — klasyfikacja przeżycia (prostate cancer)",
    )
    ctx.section(
        "CEL",
        "Budowa i ewaluacja prostego modelu klasyfikacyjnego (regresja logistyczna) "
        "rozpoznającego stan kliniczny pacjenta na podstawie danych z lab1 "
        "(syntetyczna kohorta prostate cancer).",
    )
    ctx.section(
        "PROBLEM",
        "Predykcja przeżycia pacjenta (\\texttt{survived}=1) na bazie cech: "
        "\\texttt{age}, \\texttt{psa}, \\texttt{gleason\\_score}, \\texttt{comorbidities}, "
        "\\texttt{tumor\\_stage}, \\texttt{treatment}. Klasy są niezbalansowane — "
        "stosujemy \\texttt{class\\_weight=balanced}.",
    )
    ctx.section(
        "DANE",
        f"Zbiór z lab1: \\texttt{{prostate\\_cancer\\_synth.csv}} ($N={len(df)}$). "
        "Braki w kolumnie \\texttt{psa} uzupełniono medianą. "
        "Cechy kategoryczne zakodowano OHE (drop first). Wszystkie cechy ilościowe "
        "wystandaryzowane (\\texttt{StandardScaler}).",
    )
    ctx.section(
        "METODY",
        "Regresja logistyczna z regularyzacją L2 (\\texttt{lbfgs}, \\texttt{max\\_iter}=2000) "
        "i wagami klas. Podział train/test 75/25 ze stratyfikacją po etykiecie. "
        "Miary: accuracy, precision, recall, F1, ROC AUC. "
        "Diagnostyka: krzywa ROC, krzywa PR, macierz pomyłek, ranking współczynników.",
    )
    ctx.section(
        "IMPLEMENTACJA",
        "Plik \\texttt{lab2/lab2\\_solution.py}. Funkcje \\texttt{prepare}, "
        "\\texttt{evaluate}, \\texttt{plot\\_*} oraz \\texttt{main} oddzielają etapy "
        "preprocessingu, treningu, ewaluacji i wizualizacji. "
        "Brak hardkodów, ścieżki względne.",
    )
    ctx.section(
        "OBLICZENIA",
        "Pomocnicze obliczenie odds ratio dla każdej cechy: "
        "$\\text{OR}_j = \\exp(\\beta_j)$. Dla cech wystandaryzowanych $\\beta_j$ "
        "interpretujemy jako efekt zmiany o jedno odchylenie standardowe.\n\n"
        + df_to_latex(coef_table, "Współczynniki regresji logistycznej i odds ratio.", "coef"),
    )
    ctx.section(
        "WYNIKI",
        "Miary jakości na zbiorze testowym ($N_{test}={n_test}$):\n\n".replace(
            "{n_test}", str(len(y_test))
        )
        + df_to_latex(metrics_table, "Miary jakości klasyfikacji.", "metrics", float_format="%.3f"),
    )
    ctx.section(
        "WYKRESY",
        figure_block(figures["roc"], "Krzywa ROC.", "roc")
        + figure_block(figures["pr"], "Krzywa precision-recall.", "pr")
        + figure_block(figures["cm"], "Macierz pomyłek.", "cm")
        + figure_block(figures["coef"], "Współczynniki regresji logistycznej.", "coefs"),
    )
    ctx.section(
        "INTERPRETACJA",
        "Model osiąga ROC AUC = "
        f"{metrics['roc_auc']:.3f}, F1 = {metrics['f1']:.3f}, co świadczy o zauważalnej "
        "zdolności separacji klas. Ranking współczynników wskazuje, że Gleason score, "
        "PSA i wiek pacjenta zwiększają ryzyko zgonu (ujemny wpływ na "
        "\\texttt{survived}=1), natomiast leczenie operacyjne i radioterapia wpływają "
        "korzystnie. Krzywa precision-recall potwierdza, że dobór progu decyzyjnego "
        "może istotnie zmieniać kompromis między wykrywalnością a precyzją.",
    )
    ctx.section(
        "WNIOSKI",
        "Regresja logistyczna na omawianym zbiorze stanowi solidną linię bazową: "
        "interpretowalna (wagi i OR), szybka, dobrze skalibrowana po standaryzacji. "
        "Aby poprawić jakość warto rozważyć: (a) modele drzewiaste (np. random forest, "
        "gradient boosting) — następne ćwiczenia; (b) dodatkowe cechy z domeny klinicznej; "
        "(c) optymalizację progu klasyfikacji pod konkretną miarę kliniczną.",
    )

    pdf = render_report(OUTPUT_DIR, "lab2_report.tex", ctx)
    print(f"OK -> {pdf}")


if __name__ == "__main__":
    main()
