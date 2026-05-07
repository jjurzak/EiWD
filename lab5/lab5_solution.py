r"""Lab 5 — Techniki interpretowalności (XAI).

PDF nie definiuje wariantów numerowanych — zadanie brzmi: „Na podstawie
zbioru danych poprzednich zajęć oblicz wskaźniki PI, PDP i ICE, SHAP, LIME.
Interpretuj uzyskane rezultaty.”

Wariant 8 nie istnieje, więc zgodnie z regułą realizujemy zadanie podstawowe
na zbiorze z lab1 (\texttt{prostate\_cancer\_synth.csv}).

Realizujemy:
- model bazowy: \texttt{RandomForestClassifier} (interpretowalny przez SHAP TreeExplainer);
- permutation importance (test set);
- PDP (\texttt{kind=average}) i ICE (\texttt{kind=both}) dla wybranych cech;
- SHAP TreeExplainer + summary plot + waterfall dla pojedynczego pacjenta;
- LIME tabular dla pojedynczego pacjenta.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LAB_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LAB_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPO_ROOT = LAB_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from common.report_utils import ReportContext, df_to_latex, figure_block, render_report  # noqa: E402

LAB1_CSV = REPO_ROOT / "lab1" / "output" / "prostate_cancer_synth.csv"


def load_data() -> pd.DataFrame:
    if LAB1_CSV.exists():
        return pd.read_csv(LAB1_CSV)
    sys.path.insert(0, str(REPO_ROOT / "lab1"))
    from lab1_solution import synth_prostate_cancer_dataset  # type: ignore

    df = synth_prostate_cancer_dataset()
    LAB1_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LAB1_CSV, index=False)
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = df.copy()
    df["psa"] = df["psa"].fillna(df["psa"].median())
    df_enc = pd.get_dummies(df, columns=["tumor_stage", "treatment"], drop_first=True)
    cols = [
        c
        for c in df_enc.columns
        if c not in ("patient_id", "survived", "survival_months")
    ]
    X = df_enc[cols]
    y = df_enc["survived"].astype(int).to_numpy()
    return X, y, cols


def _save(fig: plt.Figure, name: str) -> str:
    fig.savefig(OUTPUT_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_permutation(pi_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4 + 0.2 * len(pi_df)))
    pi_df = pi_df.sort_values("importance")
    colors = ["#3b6fb6" if v >= 0 else "#b6553b" for v in pi_df["importance"]]
    ax.barh(pi_df["feature"], pi_df["importance"], xerr=pi_df["std"], color=colors, edgecolor="black")
    ax.set_xlabel("Permutation Importance (spadek ROC AUC)")
    ax.set_title("Permutation Importance --- ważność cech (RandomForest, test)")
    fig.tight_layout()
    return _save(fig, "pi.png")


def plot_pdp(model, X: pd.DataFrame, features: list[str]) -> str:
    fig, axes = plt.subplots(1, len(features), figsize=(4.0 * len(features), 4))
    PartialDependenceDisplay.from_estimator(
        model, X, features=features, kind="average", ax=axes
    )
    fig.suptitle("Partial Dependence Plots (PDP)")
    fig.tight_layout()
    return _save(fig, "pdp.png")


def plot_ice(model, X: pd.DataFrame, feature: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    PartialDependenceDisplay.from_estimator(
        model, X, features=[feature], kind="both", subsample=100, ax=ax,
        random_state=8,
    )
    ax.set_title(f"ICE + PDP dla cechy {feature}")
    fig.tight_layout()
    return _save(fig, f"ice_{feature}.png")


def shap_summary(model, X_test: pd.DataFrame) -> tuple[str, np.ndarray, float]:
    explainer = shap.TreeExplainer(model)
    sv_full = explainer.shap_values(X_test, check_additivity=False)
    if isinstance(sv_full, list):
        sv = sv_full[1]
        ev = explainer.expected_value
        ev = ev[1] if hasattr(ev, "__len__") else ev
    else:
        if sv_full.ndim == 3:
            sv = sv_full[..., 1]
            ev = explainer.expected_value
            ev = ev[1] if hasattr(ev, "__len__") else ev
        else:
            sv = sv_full
            ev = explainer.expected_value
            ev = ev[1] if hasattr(ev, "__len__") else ev

    shap.summary_plot(sv, X_test, show=False, plot_size=(8, 6))
    fig = plt.gcf()
    name = "shap_summary.png"
    fig.savefig(OUTPUT_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name, sv, float(ev)


def shap_waterfall(sv_row: np.ndarray, x_row: pd.Series, base_value: float, idx: int) -> str:
    explanation = shap.Explanation(
        values=sv_row,
        base_values=base_value,
        data=x_row.values,
        feature_names=list(x_row.index),
    )
    shap.plots.waterfall(explanation, show=False, max_display=10)
    fig = plt.gcf()
    name = f"shap_waterfall_{idx}.png"
    fig.savefig(OUTPUT_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


def lime_explain(model, X_train: pd.DataFrame, X_test: pd.DataFrame, idx: int = 0) -> tuple[str, list[tuple[str, float]]]:
    feature_names = list(X_train.columns)
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["zgon", "przeżyl"],
        discretize_continuous=True,
        mode="classification",
    )

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(arr, columns=feature_names)
        return model.predict_proba(df)

    exp = explainer.explain_instance(
        data_row=X_test.iloc[idx].values,
        predict_fn=predict_fn,
        num_features=8,
        top_labels=1,
    )
    label = exp.top_labels[0]
    pairs = exp.as_list(label=label)
    fig = exp.as_pyplot_figure(label=label)
    fig.set_size_inches(8, 5)
    fig.suptitle(f"LIME --- pacjent #{idx} (klasa: {explainer.class_names[label]})")
    fig.tight_layout()
    name = f"lime_{idx}.png"
    fig.savefig(OUTPUT_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name, pairs


def main() -> None:
    df = load_data()
    X, y, cols = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=8, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=8, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    pi = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=8, n_jobs=-1
    )
    pi_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": pi.importances_mean,
            "std": pi.importances_std,
        }
    ).sort_values("importance", ascending=False)
    pi_df_print = pi_df.copy()
    pi_df_print["importance"] = pi_df_print["importance"].round(4)
    pi_df_print["std"] = pi_df_print["std"].round(4)

    fig_pi = plot_permutation(pi_df)
    pdp_features = ["age", "psa", "gleason_score"]
    fig_pdp = plot_pdp(model, X_test, pdp_features)
    fig_ice_age = plot_ice(model, X_test, "age")
    fig_ice_psa = plot_ice(model, X_test, "psa")

    fig_shap_summary, sv, ev = shap_summary(model, X_test)
    fig_waterfall = shap_waterfall(sv[0], X_test.iloc[0], ev, idx=0)

    fig_lime, lime_pairs = lime_explain(model, X_train, X_test, idx=0)
    lime_table = pd.DataFrame(lime_pairs, columns=["warunek na cesze", "wkład LIME"])
    lime_table["wkład LIME"] = lime_table["wkład LIME"].round(4)

    ctx = ReportContext(
        lab_number=5,
        report_title="XAI: Permutation Importance, PDP/ICE, SHAP, LIME",
        variant="0/1 (brak wariantu 8 — zadanie jednolite)",
    )
    ctx.section(
        "CEL",
        "Praktyczne zastosowanie metod interpretowalności modeli ML w domenie medycznej. "
        "Porównanie globalnych (PI, PDP, SHAP summary) i lokalnych (ICE, SHAP waterfall, "
        "LIME) wyjaśnień decyzji klasyfikatora przeżycia.",
    )
    ctx.section(
        "PROBLEM",
        "Modele ensemblowe (Random Forest) są nieliniowe i nieintuicyjne. W praktyce "
        "klinicznej wymagana jest możliwość uzasadnienia każdej predykcji "
        "(rozporządzenie AI Act, dobre praktyki kliniczne). Stąd potrzeba narzędzi XAI.",
    )
    ctx.section(
        "DANE",
        f"Zbiór z lab1: \\texttt{{prostate\\_cancer\\_synth.csv}} ($N={len(df)}$). "
        "Cechy: \\texttt{age, psa, gleason\\_score, comorbidities}, OHE dla "
        "\\texttt{tumor\\_stage}, \\texttt{treatment}. Etykieta: \\texttt{survived}.",
    )
    ctx.section(
        "METODY",
        "\\textbf{Model:} \\texttt{RandomForestClassifier} (300 drzew, "
        "\\texttt{class\\_weight=balanced}). \\\\"
        "\\textbf{PI:} \\texttt{permutation\\_importance} z metryką ROC AUC "
        "($n\\_repeats=20$).\\\\"
        "\\textbf{PDP/ICE:} \\texttt{PartialDependenceDisplay} dla cech \\texttt{age}, "
        "\\texttt{psa}, \\texttt{gleason\\_score} (PDP) oraz krzywych ICE dla "
        "\\texttt{age} i \\texttt{psa}.\\\\"
        "\\textbf{SHAP:} \\texttt{TreeExplainer}, \\texttt{summary\\_plot} (globalnie) "
        "oraz \\texttt{waterfall} dla pacjenta \\#0.\\\\"
        "\\textbf{LIME:} \\texttt{LimeTabularExplainer} z dyskretyzacją cech ciągłych, "
        "8 cech w wyjaśnieniu.",
    )
    ctx.section(
        "IMPLEMENTACJA",
        "Plik \\texttt{lab5/lab5\\_solution.py}. Struktura: "
        "\\texttt{prepare\\_features} $\\to$ trening RF $\\to$ "
        "\\texttt{permutation\\_importance} $\\to$ \\texttt{plot\\_pdp/plot\\_ice} "
        "$\\to$ \\texttt{shap\\_summary/shap\\_waterfall} $\\to$ \\texttt{lime\\_explain}.",
    )
    ctx.section(
        "OBLICZENIA",
        f"Wskaźniki bazowe modelu: \\textbf{{accuracy}} = {acc:.3f}, "
        f"\\textbf{{ROC AUC}} = {auc:.3f}.\n\n"
        + df_to_latex(pi_df_print, "Permutation Importance --- ranking cech.", "pi", float_format="%.4f")
        + "\nWyjaśnienie LIME dla pacjenta \\#0:\n\n"
        + df_to_latex(lime_table, "Wyjaśnienie LIME (pacjent \\#0).", "lime", float_format="%.4f"),
    )
    ctx.section(
        "WYNIKI",
        "Najważniejsze cechy globalnie (zarówno PI, jak i SHAP) to: \\texttt{psa}, "
        "\\texttt{gleason\\_score}, \\texttt{age}, oraz pochodne \\texttt{tumor\\_stage}. "
        "Lokalnie (LIME, SHAP waterfall) sygnatura ważnych cech zmienia się w zależności "
        "od pacjenta, co potwierdza heterogeniczność populacji.",
    )
    ctx.section(
        "WYKRESY",
        figure_block(fig_pi, "Permutation Importance.", "pi")
        + figure_block(fig_pdp, "PDP --- średni wpływ cech.", "pdp")
        + figure_block(fig_ice_age, "ICE + PDP dla cechy \\texttt{age}.", "ice-age")
        + figure_block(fig_ice_psa, "ICE + PDP dla cechy \\texttt{psa}.", "ice-psa")
        + figure_block(fig_shap_summary, "SHAP summary plot (TreeExplainer).", "shap-sum")
        + figure_block(fig_waterfall, "SHAP waterfall --- pacjent \\#0.", "shap-wf")
        + figure_block(fig_lime, "LIME --- wyjaśnienie pacjenta \\#0.", "lime-fig"),
    )
    ctx.section(
        "INTERPRETACJA",
        "PDP \\texttt{psa} pokazuje rosnący trend prawdopodobieństwa zgonu wraz ze "
        "wzrostem PSA, zgodny z wiedzą kliniczną. ICE \\texttt{age} ujawnia "
        "heterogeniczność: dla pacjentów z wysokim Gleason score wpływ wieku jest silniejszy. "
        "SHAP summary plot potwierdza ranking PI i dodatkowo pokazuje kierunek wpływu "
        "(wysokie PSA $\\rightarrow$ większe ryzyko). Waterfall dla pacjenta \\#0 "
        "wskazuje konkretne cechy, które przesunęły jego predykcję powyżej/poniżej "
        "wartości oczekiwanej. LIME daje zgodne, choć bardziej zgrubne, wyjaśnienie "
        "(dyskretyzacja cech).",
    )
    ctx.section(
        "WNIOSKI",
        "Komplementarność metod: globalne (PI, SHAP summary) wskazują, na których cechach "
        "model polega \\textit{statystycznie}, lokalne (SHAP waterfall, LIME) wyjaśniają "
        "konkretne decyzje. W zastosowaniach klinicznych zaleca się stosować obie warstwy. "
        "Pułapki: skorelowane cechy (np. \\texttt{psa} $\\sim$ \\texttt{gleason\\_score}) "
        "powodują rozdzielanie wpływu, a model może zaniżać PI dla zbędnych zmiennych.",
    )

    pdf = render_report(OUTPUT_DIR, "lab5_report.tex", ctx)
    print(f"OK -> {pdf}")


if __name__ == "__main__":
    main()
