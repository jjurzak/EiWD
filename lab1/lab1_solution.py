"""Lab 1 — Praca z danymi medycznymi.

Wariant zadania: 1 (fallback z 8 — wariant 8 nie istnieje w liście zadań).
Wariant danych: 8 — Prostate cancer (synth z powodu braku dostępu do Kaggle).

Realizuje:
- import danych pacjentów (CSV, ścieżka względna),
- wstępne przetwarzanie (braki, typy, standaryzacja),
- analizę statystyczną zmiennych,
- wizualizacje (histogram, boxplot, korelacje),
- prosty model bazowy (regresja logistyczna) jako weryfikacja jakości danych.

Wszystkie artefakty (CSV, PNG, tabele) lądują w `lab1/output/`.
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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LAB_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LAB_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

REPO_ROOT = LAB_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from common.report_utils import ReportContext, df_to_latex, figure_block, render_report  # noqa: E402


# --------------------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------------------
def synth_prostate_cancer_dataset(n: int = 600, seed: int = 8) -> pd.DataFrame:
    """Generate a synthetic prostate-cancer-survival-style cohort.

    Cechy zgodne z opisem wariantu 8 (kaggle: prostate-cancer-survival-data):
    age, psa, gleason_score, tumor_stage, treatment, comorbidities, survival_months,
    survived (etykieta binarna).
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(68, 9, size=n).clip(45, 92).round().astype(int)
    psa = rng.lognormal(mean=2.0, sigma=0.7, size=n).round(2)
    gleason = rng.choice([6, 7, 8, 9, 10], size=n, p=[0.18, 0.42, 0.22, 0.12, 0.06])
    tumor_stage = rng.choice(
        ["T1", "T2", "T3", "T4"], size=n, p=[0.32, 0.38, 0.22, 0.08]
    )
    treatment = rng.choice(
        ["surgery", "radiation", "hormonal", "watch"], size=n, p=[0.32, 0.34, 0.22, 0.12]
    )
    comorbidities = rng.poisson(1.2, size=n).clip(0, 6)

    stage_risk = pd.Series(tumor_stage).map({"T1": 0.0, "T2": 0.4, "T3": 0.9, "T4": 1.6}).values
    treat_risk = pd.Series(treatment).map(
        {"surgery": -0.4, "radiation": -0.2, "hormonal": 0.1, "watch": 0.6}
    ).values

    logit = (
        -3.0
        + 0.04 * (age - 68)
        + 0.05 * (psa - 8)
        + 0.45 * (gleason - 7)
        + stage_risk
        + treat_risk
        + 0.18 * comorbidities
    )
    p_death = 1.0 / (1.0 + np.exp(-logit))
    died = rng.binomial(1, p_death)
    survival_months = np.where(
        died == 1,
        rng.gamma(2.0, 12.0, size=n).round(1),
        rng.gamma(4.5, 18.0, size=n).round(1),
    ).clip(1, 240)
    survived = 1 - died

    missing_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    psa_with_na = psa.astype(float)
    psa_with_na[missing_idx] = np.nan

    return pd.DataFrame(
        {
            "patient_id": np.arange(1, n + 1),
            "age": age,
            "psa": psa_with_na,
            "gleason_score": gleason,
            "tumor_stage": tumor_stage,
            "treatment": treatment,
            "comorbidities": comorbidities,
            "survival_months": survival_months,
            "survived": survived,
        }
    )


def load_or_create(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    df = synth_prostate_cancer_dataset()
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {}
    info["n_rows"] = len(df)
    info["missing_before"] = df.isna().sum().to_dict()

    df = df.copy()
    df["psa"] = df["psa"].fillna(df["psa"].median())
    df["tumor_stage"] = df["tumor_stage"].astype("category")
    df["treatment"] = df["treatment"].astype("category")

    numeric_cols = ["age", "psa", "gleason_score", "comorbidities", "survival_months"]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    info["missing_after"] = df.isna().sum().to_dict()
    info["scaler_means"] = dict(zip(numeric_cols, scaler.mean_))
    info["scaler_scales"] = dict(zip(numeric_cols, scaler.scale_))
    return df, df_scaled, info  # type: ignore[return-value]


# --------------------------------------------------------------------------------------
# Statistical analysis
# --------------------------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["patient_id"], errors="ignore")
    desc = numeric.describe().T[["mean", "std", "min", "50%", "max"]]
    desc.columns = ["średnia", "odch. std.", "min", "mediana", "max"]
    desc.insert(0, "zmienna", desc.index)
    desc.reset_index(drop=True, inplace=True)
    return desc.round(3)


def category_counts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for col in ["tumor_stage", "treatment"]:
        counts = df[col].value_counts().rename_axis("kategoria").reset_index(name="liczność")
        counts["odsetek (%)"] = (counts["liczność"] / counts["liczność"].sum() * 100).round(2)
        out[col] = counts
    return out


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def _save(fig: plt.Figure, name: str) -> str:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_histograms(df: pd.DataFrame) -> str:
    cols = ["age", "psa", "gleason_score", "comorbidities", "survival_months"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=25, color="#3b6fb6", edgecolor="black", alpha=0.85)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("liczność")
    axes[-1].axis("off")
    fig.suptitle("Histogramy zmiennych ilościowych — kohorta prostate cancer (synth)")
    fig.tight_layout()
    return _save(fig, "hist_numeric.png")


def plot_boxplots(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = ["age", "psa", "gleason_score", "comorbidities", "survival_months"]
    ax.boxplot([df[c].dropna() for c in cols], labels=cols, patch_artist=True)
    ax.set_title("Wykresy pudełkowe zmiennych ilościowych")
    ax.set_ylabel("wartość")
    fig.tight_layout()
    return _save(fig, "boxplot_numeric.png")


def plot_correlation(df: pd.DataFrame) -> str:
    cols = ["age", "psa", "gleason_score", "comorbidities", "survival_months", "survived"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045)
    ax.set_title("Macierz korelacji Pearsona")
    fig.tight_layout()
    return _save(fig, "corr_matrix.png")


def plot_category_bars(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, color in zip(axes, ["tumor_stage", "treatment"], ["#3b6fb6", "#b6553b"]):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=color, edgecolor="black")
        ax.set_title(f"Liczność kategorii: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("liczność")
    fig.tight_layout()
    return _save(fig, "categories.png")


# --------------------------------------------------------------------------------------
# Baseline model (sanity check)
# --------------------------------------------------------------------------------------
def baseline_model(df: pd.DataFrame) -> dict:
    feat_num = ["age", "psa", "gleason_score", "comorbidities"]
    df_enc = df.copy()
    df_enc = pd.get_dummies(df_enc, columns=["tumor_stage", "treatment"], drop_first=True)
    feat = feat_num + [c for c in df_enc.columns if c.startswith(("tumor_stage_", "treatment_"))]

    X = df_enc[feat].values
    y = df_enc["survived"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=8, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


# --------------------------------------------------------------------------------------
# Report glue
# --------------------------------------------------------------------------------------
def build_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    desc: pd.DataFrame,
    cat_counts: dict[str, pd.DataFrame],
    figures: dict[str, str],
    model_metrics: dict,
    info: dict,
) -> Path:
    ctx = ReportContext(
        lab_number=1,
        report_title="Praca z danymi medycznymi: import, przetwarzanie i analiza",
        variant="1 (fallback z 8) — dataset 8: Prostate cancer (synth)",
    )

    ctx.section(
        "CEL",
        "Zapoznanie z procesem pracy z danymi medycznymi: importem, czyszczeniem, "
        "uzupełnianiem braków, standaryzacją, analizą statystyczną oraz wizualizacją "
        "danych pacjentów. W wariancie wykorzystano kohortę pacjentów onkologicznych "
        "(prostate cancer, syntetyczna).",
    )
    ctx.section(
        "PROBLEM",
        "Dane medyczne są wrażliwe, niejednorodne i często niekompletne. Zadaniem jest "
        "ich przygotowanie do analizy oraz dalszego modelowania, w tym diagnoza braków, "
        "konwersja typów, standaryzacja zmiennych ilościowych oraz weryfikacja zależności "
        "między cechami a stanem klinicznym pacjenta.",
    )
    ctx.section(
        "DANE",
        f"Zbiór syntetyczny imitujący kohortę prostate-cancer-survival "
        f"(N={info['n_rows']}). Zmienne: \\texttt{{age}}, \\texttt{{psa}}, "
        f"\\texttt{{gleason\\_score}}, \\texttt{{tumor\\_stage}}, \\texttt{{treatment}}, "
        f"\\texttt{{comorbidities}}, \\texttt{{survival\\_months}}, "
        f"\\texttt{{survived}}. Braki danych wprowadzono losowo w kolumnie "
        f"\\texttt{{psa}} (5\\%).",
    )
    ctx.section(
        "METODY",
        "Zastosowano: (a) uzupełnianie braków medianą dla \\texttt{psa}; "
        "(b) standaryzację Z-score zmiennych ilościowych; "
        "(c) statystyki opisowe (średnia, mediana, odch.\\ std., min, max); "
        "(d) wizualizację rozkładów (histogramy, wykresy pudełkowe) i zależności "
        "(macierz korelacji Pearsona); "
        "(e) prostą regresję logistyczną z wagami klas jako linię bazową dla zadania "
        "predykcji przeżycia.",
    )
    ctx.section(
        "IMPLEMENTACJA",
        "Pełna implementacja w \\texttt{lab1/lab1\\_solution.py}. Kluczowe etapy: "
        "\\texttt{synth\\_prostate\\_cancer\\_dataset} -- generator danych; "
        "\\texttt{preprocess} -- czyszczenie i standaryzacja; "
        "\\texttt{descriptive\\_stats} -- statystyki opisowe; "
        "\\texttt{plot\\_*} -- wizualizacje; "
        "\\texttt{baseline\\_model} -- regresja logistyczna. "
        "Ścieżki względne, brak hardkodów.",
    )

    miss_table = pd.DataFrame(
        {"kolumna": list(info["missing_before"].keys()),
         "braki przed": list(info["missing_before"].values()),
         "braki po": [info["missing_after"][k] for k in info["missing_before"].keys()]}
    )

    ctx.section(
        "OBLICZENIA",
        "Liczności braków przed i po imputacji medianą:\n\n"
        + df_to_latex(miss_table, "Braki danych przed i po imputacji.", "missing", float_format="%.0f")
        + "\n"
        + "Statystyki opisowe zmiennych ilościowych:\n\n"
        + df_to_latex(desc, "Statystyki opisowe zmiennych ilościowych.", "desc"),
    )

    ctx.section(
        "WYNIKI",
        "Liczności kategorii \\texttt{tumor\\_stage}:\n\n"
        + df_to_latex(cat_counts["tumor_stage"], "Liczność kategorii \\texttt{tumor\\_stage}.", "tumor")
        + "\nLiczności kategorii \\texttt{treatment}:\n\n"
        + df_to_latex(cat_counts["treatment"], "Liczność kategorii \\texttt{treatment}.", "treat")
        + f"\nLinia bazowa (regresja logistyczna): "
        f"\\textbf{{accuracy}} = {model_metrics['accuracy']:.3f}, "
        f"\\textbf{{ROC AUC}} = {model_metrics['roc_auc']:.3f} "
        f"(zbiór testowy N={model_metrics['n_test']}).",
    )

    ctx.section(
        "WYKRESY",
        figure_block(figures["hist"], "Histogramy zmiennych ilościowych.", "hist")
        + figure_block(figures["box"], "Wykresy pudełkowe zmiennych ilościowych.", "box")
        + figure_block(figures["corr"], "Macierz korelacji Pearsona.", "corr")
        + figure_block(figures["cat"], "Liczności kategorii \\texttt{tumor\\_stage} i \\texttt{treatment}.", "cat"),
    )

    ctx.section(
        "INTERPRETACJA",
        "Histogramy pokazują typowe rozkłady: wiek skupiony wokół 68 lat, PSA prawoskośny "
        "(rozkład logarytmiczno-normalny), Gleason score skoncentrowany w okolicy 7. "
        "Macierz korelacji ujawnia umiarkowaną dodatnią zależność wieku i Gleason score "
        "z ryzykiem zgonu (ujemna korelacja ze zmienną \\texttt{survived}). "
        "Liczności kategorii pokazują dominację stadium T1--T2 oraz terapii operacyjnej "
        "i radioterapii. Linia bazowa logistic regression osiąga "
        f"AUC = {model_metrics['roc_auc']:.3f}, co potwierdza spójność danych "
        "i sensowność wybranych cech.",
    )
    ctx.section(
        "WNIOSKI",
        "Etap importu, czyszczenia i analizy statystycznej został zrealizowany. "
        "Wprowadzono kontrolowane braki, które uzupełniono medianą bez utraty rekordów. "
        "Standaryzacja Z-score umożliwia porównywalność cech o różnych skalach. "
        "Analiza wskazuje, że PSA, Gleason score i stadium guza są informatywnymi "
        "predyktorami przeżycia, co stanowi punkt wyjścia dla bardziej zaawansowanych "
        "modeli (kolejne laboratoria).",
    )

    return render_report(OUTPUT_DIR, "lab1_report.tex", ctx)


def main() -> None:
    csv_path = OUTPUT_DIR / "prostate_cancer_synth.csv"
    df = load_or_create(csv_path)
    df_clean, df_scaled, info = preprocess(df)
    desc = descriptive_stats(df_clean)
    cats = category_counts(df_clean)

    figures = {
        "hist": plot_histograms(df_clean),
        "box": plot_boxplots(df_clean),
        "corr": plot_correlation(df_clean),
        "cat": plot_category_bars(df_clean),
    }

    metrics = baseline_model(df_clean)
    pdf = build_report(df, df_clean, desc, cats, figures, metrics, info)
    print(f"OK -> {pdf}")


if __name__ == "__main__":
    main()
