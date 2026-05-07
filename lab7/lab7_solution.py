from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAB_DIR = Path(__file__).resolve().parent
DATA_CSV = LAB_DIR / "pacjenci_demo_system_ekspertowy.csv"
OUTPUT_DIR = LAB_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPO_ROOT = LAB_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from common.report_utils import ReportContext, df_to_latex, figure_block, render_report  # noqa: E402



#data layer
@dataclass
class CrispRule:
    name: str
    condition: Callable[[dict], bool]
    conclusion: str
    weight: float

#crisp rules
def crisp_rules() -> list[CrispRule]:
    return [
        CrispRule(
            name="R1_geriatric_basic",
            condition=lambda p: p["age"] >= 70,
            conclusion="Geriatric patient",
            weight=0.95,
        ),
        CrispRule(
            name="R2_geriatric_hypertension",
            condition=lambda p: p["age"] >= 70 and p["systolic_bp"] >= 140,
            conclusion="Geriatric high cardiovascular risk",
            weight=0.85,
        ),
        CrispRule(
            name="R3_geriatric_metabolic",
            condition=lambda p: p["age"] >= 70 and (p["bmi"] >= 30 or p["glucose"] >= 126),
            conclusion="Geriatric metabolic risk",
            weight=0.8,
        ),
        CrispRule(
            name="R4_hypertension",
            condition=lambda p: p["systolic_bp"] >= 140 and p["diastolic_bp"] >= 90,
            conclusion="Hypertension",
            weight=0.9,
        ),
        CrispRule(
            name="R5_diabetes",
            condition=lambda p: p["glucose"] >= 126,
            conclusion="Diabetes",
            weight=0.85,
        ),
    ]


def crisp_inference(patient: dict, rules: list[CrispRule]) -> tuple[dict[str, float], list[str]]:
    activated: list[str] = []
    conclusions: dict[str, float] = {}
    for r in rules:
        if r.condition(patient):
            activated.append(r.name)
            conclusions[r.conclusion] = max(conclusions.get(r.conclusion, 0.0), r.weight)
    return conclusions, activated


#fuzzy rules
def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal MF: 0 below a, ramps to 1 at b, 1 until c, ramps to 0 at d."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def triangle(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def mu_age_elderly(age: float) -> float:
    return trapezoid(age, 55, 70, 200, 200)


def mu_age_middle(age: float) -> float:
    return trapezoid(age, 35, 45, 60, 70)


def mu_sbp_high(sbp: float) -> float:
    return trapezoid(sbp, 130, 140, 250, 250)


def mu_dbp_high(dbp: float) -> float:
    return trapezoid(dbp, 80, 90, 200, 200)


def mu_glucose_high(g: float) -> float:
    return trapezoid(g, 110, 126, 400, 400)


def mu_bmi_high(b: float) -> float:
    return trapezoid(b, 27, 30, 100, 100)


# Output universe: geriatric_risk in [0, 100]
RISK_GRID = np.linspace(0, 100, 1001)


def mu_risk_low(y: np.ndarray) -> np.ndarray:
    return np.array([trapezoid(v, 0, 0, 20, 40) for v in y])


def mu_risk_med(y: np.ndarray) -> np.ndarray:
    return np.array([triangle(v, 30, 50, 70) for v in y])


def mu_risk_high(y: np.ndarray) -> np.ndarray:
    return np.array([trapezoid(v, 60, 80, 100, 100) for v in y])


#fuzzy rules
@dataclass
class FuzzyRule:
    name: str
    description: str
    antecedent: Callable[[dict], float]
    consequent: Callable[[np.ndarray], np.ndarray]


#fuzzy rules
def fuzzy_rules() -> list[FuzzyRule]:
    return [
        FuzzyRule(
            name="F1_elderly",
            description="IF age is elderly THEN risk is medium",
            antecedent=lambda p: mu_age_elderly(p["age"]),
            consequent=mu_risk_med,
        ),
        FuzzyRule(
            name="F2_elderly_sbp",
            description="IF age is elderly AND SBP is high THEN risk is high",
            antecedent=lambda p: min(mu_age_elderly(p["age"]), mu_sbp_high(p["systolic_bp"])),
            consequent=mu_risk_high,
        ),
        FuzzyRule(
            name="F3_elderly_glucose",
            description="IF age is elderly AND glucose is high THEN risk is high",
            antecedent=lambda p: min(mu_age_elderly(p["age"]), mu_glucose_high(p["glucose"])),
            consequent=mu_risk_high,
        ),
        FuzzyRule(
            name="F4_elderly_bmi",
            description="IF age is elderly AND BMI is high THEN risk is high",
            antecedent=lambda p: min(mu_age_elderly(p["age"]), mu_bmi_high(p["bmi"])),
            consequent=mu_risk_high,
        ),
        FuzzyRule(
            name="F5_middle_age",
            description="IF age is middle THEN risk is low",
            antecedent=lambda p: mu_age_middle(p["age"]),
            consequent=mu_risk_low,
        ),
    ]


#fuzzy inference
def fuzzy_inference(patient: dict, rules: list[FuzzyRule]) -> dict:
    activations: dict[str, float] = {}
    clipped: dict[str, np.ndarray] = {}
    for r in rules:
        alpha = r.antecedent(patient)
        activations[r.name] = alpha
        consequent_mu = r.consequent(RISK_GRID)
        clipped[r.name] = np.minimum(alpha, consequent_mu)

    aggregated = np.zeros_like(RISK_GRID)
    for clip in clipped.values():
        aggregated = np.maximum(aggregated, clip)

    if aggregated.sum() > 1e-9:
        centroid = float((RISK_GRID * aggregated).sum() / aggregated.sum())
    else:
        centroid = 0.0

    contributions = {}
    total_area = sum(c.sum() for c in clipped.values())
    for name, c in clipped.items():
        contributions[name] = float(c.sum() / total_area) if total_area > 0 else 0.0

    return {
        "activations": activations,
        "clipped": clipped,
        "aggregated": aggregated,
        "risk_score": centroid,
        "contributions": contributions,
    }


#explainability
def explain_patient(patient: dict, fuzzy_result: dict, rules: list[FuzzyRule]) -> str:
    lines = []
    lines.append(f"Pacjent ID = {patient.get('patient_id', '?')}, age = {patient['age']}, "
                 f"BMI = {patient['bmi']:.1f}, glukoza = {patient['glucose']}, "
                 f"SBP = {patient['systolic_bp']}, DBP = {patient['diastolic_bp']}.")
    lines.append("Stopnie przynależności wejść:")
    lines.append(
        f" age $\\to$ elderly = {mu_age_elderly(patient['age']):.2f}, "
        f"middle = {mu_age_middle(patient['age']):.2f}; "
        f"SBP $\\to$ high = {mu_sbp_high(patient['systolic_bp']):.2f}; "
        f"glukoza $\\to$ high = {mu_glucose_high(patient['glucose']):.2f}; "
        f"BMI $\\to$ high = {mu_bmi_high(patient['bmi']):.2f}."
    )
    lines.append("Aktywacje reguł rozmytych:")
    for r in rules:
        a = fuzzy_result["activations"][r.name]
        c = fuzzy_result["contributions"][r.name]
        name_tex = r.name.replace("_", r"\_")
        lines.append(f" \\textbf{{{name_tex}}} ($\\alpha={a:.2f}$, contrib = {c*100:.1f}\\%): {r.description}.")
    lines.append(f"Końcowy wskaźnik ryzyka geriatrycznego (defuzzifikacja centroid): "
                 f"\\textbf{{{fuzzy_result['risk_score']:.1f}/100}}.")
    return "\n\n".join(lines)


#plots
def _save(fig: plt.Figure, name: str) -> str:
    fig.savefig(OUTPUT_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


#plots
def plot_membership_age() -> str:
    ages = np.linspace(20, 100, 400)
    elderly = [mu_age_elderly(a) for a in ages]
    middle = [mu_age_middle(a) for a in ages]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ages, elderly, label="elderly", color="#3b6fb6", lw=2)
    ax.plot(ages, middle, label="middle", color="#b6553b", lw=2)
    ax.set_xlabel("age")
    ax.set_ylabel(r"$\mu(age)$")
    ax.set_title("Funkcje przynależności --- wiek (wariant geriatryczny)")
    ax.axvline(70, color="black", linestyle="--", lw=1, label="próg klasyczny age=70")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "mf_age.png")



def plot_membership_others() -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    sbp = np.linspace(80, 220, 400)
    axes[0, 0].plot(sbp, [mu_sbp_high(v) for v in sbp], color="#3b6fb6", lw=2)
    axes[0, 0].set_title("SBP high"); axes[0, 0].set_xlabel("SBP"); axes[0, 0].set_ylabel(r"$\mu$")
    dbp = np.linspace(50, 130, 400)
    axes[0, 1].plot(dbp, [mu_dbp_high(v) for v in dbp], color="#3b6fb6", lw=2)
    axes[0, 1].set_title("DBP high"); axes[0, 1].set_xlabel("DBP"); axes[0, 1].set_ylabel(r"$\mu$")
    glu = np.linspace(60, 250, 400)
    axes[1, 0].plot(glu, [mu_glucose_high(v) for v in glu], color="#3b6fb6", lw=2)
    axes[1, 0].set_title("glucose high"); axes[1, 0].set_xlabel("glucose"); axes[1, 0].set_ylabel(r"$\mu$")
    bmi = np.linspace(15, 50, 400)
    axes[1, 1].plot(bmi, [mu_bmi_high(v) for v in bmi], color="#3b6fb6", lw=2)
    axes[1, 1].set_title("BMI high"); axes[1, 1].set_xlabel("BMI"); axes[1, 1].set_ylabel(r"$\mu$")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "mf_others.png")


def plot_aggregation(fuzzy_result: dict, patient_id: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(RISK_GRID, mu_risk_low(RISK_GRID), color="#888888", linestyle="--", label="risk low")
    ax.plot(RISK_GRID, mu_risk_med(RISK_GRID), color="#888888", linestyle="--", label="risk medium")
    ax.plot(RISK_GRID, mu_risk_high(RISK_GRID), color="#888888", linestyle="--", label="risk high")
    ax.fill_between(RISK_GRID, fuzzy_result["aggregated"], color="#3b6fb6", alpha=0.4,
                    label="agregacja")
    ax.axvline(fuzzy_result["risk_score"], color="red", lw=2, label=f"centroid = {fuzzy_result['risk_score']:.1f}")
    ax.set_xlabel("risk")
    ax.set_ylabel(r"$\mu_{agg}(y)$")
    ax.set_title(f"Wnioskowanie Mamdaniego --- pacjent {patient_id}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"agg_{patient_id}.png")


def plot_global_activations(activation_matrix: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    means = activation_matrix.mean().sort_values(ascending=False)
    ax.bar(means.index, means.values, color="#3b6fb6", edgecolor="black")
    ax.set_ylabel("średnia siła aktywacji")
    ax.set_title("Globalne aktywacje reguł rozmytych w kohorcie")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _save(fig, "global_act.png")


def plot_risk_vs_age(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(df["age"], df["fuzzy_risk"], c=df["classical_geriatric"],
                    cmap="coolwarm", edgecolor="black", s=60)
    ax.axvline(70, color="black", linestyle="--", lw=1, label="próg klasyczny age=70")
    ax.set_xlabel("age")
    ax.set_ylabel("ryzyko geriatryczne (fuzzy)")
    ax.set_title("Ryzyko geriatryczne vs wiek (kolor = klasyczna decyzja binarna)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="classical geriatric (0/1)")
    fig.tight_layout()
    return _save(fig, "risk_vs_age.png")


#report glue
def main() -> None:
    df = pd.read_csv(DATA_CSV)
    crisp_rs = crisp_rules()
    fuzzy_rs = fuzzy_rules()

    crisp_outputs = []
    fuzzy_outputs = []
    activation_rows = []
    explanations: dict[str, str] = {}
    for _, row in df.iterrows():
        patient = row.to_dict()
        crisp, activated = crisp_inference(patient, crisp_rs)
        is_geriatric = int("Geriatric patient" in crisp)
        crisp_outputs.append({
            "patient_id": patient["patient_id"],
            "age": patient["age"],
            "classical_geriatric": is_geriatric,
            "classical_conclusions": "; ".join(f"{k}({v:.2f})" for k, v in crisp.items()) or "—",
            "rules_activated": ", ".join(activated) or "—",
        })

        fres = fuzzy_inference(patient, fuzzy_rs)
        fuzzy_outputs.append({
            "patient_id": patient["patient_id"],
            "age": patient["age"],
            "fuzzy_risk": round(fres["risk_score"], 2),
        })
        act_row = {"patient_id": patient["patient_id"]}
        act_row.update({k: round(v, 3) for k, v in fres["activations"].items()})
        activation_rows.append(act_row)

        explanations[patient["patient_id"]] = explain_patient(patient, fres, fuzzy_rs)

    crisp_df = pd.DataFrame(crisp_outputs)
    fuzzy_df = pd.DataFrame(fuzzy_outputs)
    act_df = pd.DataFrame(activation_rows).set_index("patient_id")

    summary = crisp_df.merge(fuzzy_df[["patient_id", "fuzzy_risk"]], on="patient_id")
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    n_geriatric_classic = int(summary["classical_geriatric"].sum())
    avg_risk_geriatric = float(summary[summary["classical_geriatric"] == 1]["fuzzy_risk"].mean())
    avg_risk_non = float(summary[summary["classical_geriatric"] == 0]["fuzzy_risk"].mean())

    fig_age = plot_membership_age()
    fig_others = plot_membership_others()
    fig_global = plot_global_activations(act_df)
    fig_risk_age = plot_risk_vs_age(summary)

    sample_id = "P02"
    sample_patient = df[df["patient_id"] == sample_id].iloc[0].to_dict()
    fres_sample = fuzzy_inference(sample_patient, fuzzy_rs)
    fig_agg_sample = plot_aggregation(fres_sample, sample_id)
    explanation_sample = explain_patient(sample_patient, fres_sample, fuzzy_rs)

    rule_table = pd.DataFrame([
        {"reguła": "R1_geriatric_basic", "warunek": "age $\\ge$ 70", "wniosek": "Geriatric patient", "waga": 0.95},
        {"reguła": "R2_geriatric_hypertension", "warunek": "age $\\ge$ 70 AND SBP $\\ge$ 140", "wniosek": "Geriatric high CV risk", "waga": 0.85},
        {"reguła": "R3_geriatric_metabolic", "warunek": "age $\\ge$ 70 AND (BMI $\\ge$ 30 OR glucose $\\ge$ 126)", "wniosek": "Geriatric metabolic risk", "waga": 0.8},
        {"reguła": "R4_hypertension", "warunek": "SBP $\\ge$ 140 AND DBP $\\ge$ 90", "wniosek": "Hypertension", "waga": 0.9},
        {"reguła": "R5_diabetes", "warunek": "glucose $\\ge$ 126", "wniosek": "Diabetes", "waga": 0.85},
    ])

    fuzzy_rule_table = pd.DataFrame([
        {"reguła": fr.name, "opis": fr.description} for fr in fuzzy_rs
    ])

    short_summary = summary[["patient_id", "age", "classical_geriatric", "fuzzy_risk"]].copy()

    ctx = ReportContext(
        lab_number=7,
        report_title="Regułowy system ekspertowy --- pacjent geriatryczny",
        variant="8 — pacjent geriatryczny (wiek $\\ge$ 70)",
    )
    ctx.section(
        "CEL",
        "Implementacja klasycznego i rozmytego systemu ekspertowego rozpoznającego "
        "pacjentów geriatrycznych oraz oceniającego ich ryzyko zdrowotne. Dodanie "
        "modułu wyjaśnialności (explainable expert system) z analizą wpływu wieku "
        "na decyzję.",
    )
    ctx.section(
        "PROBLEM",
        "Podstawowy próg wiekowy ($\\ge$ 70 lat) nie oddaje stopniowej natury starzenia. "
        "Wymagane jest połączenie reguł binarnych z fuzzy logic, aby uzyskać miękką ocenę "
        "ryzyka oraz uzasadnić decyzję dla pacjentów na granicy progu.",
    )
    ctx.section(
        "DANE",
        f"Plik \\texttt{{pacjenci\\_demo\\_system\\_ekspertowy.csv}} ($N={len(df)}$). "
        "Cechy: \\texttt{age, bmi, glucose, systolic\\_bp, diastolic\\_bp}.",
    )
    ctx.section(
        "METODY",
        "\\textbf{Etap 1.} Reguły binarne (\\texttt{IF-THEN}) z wagą pewności i agregacją "
        "$\\max$ po hipotezach.\\\\"
        "\\textbf{Etap 2.} Wnioskowanie Mamdaniego: 5 reguł rozmytych, fuzzifikacja "
        "(trapezoid/triangle MF), implikacja (\\texttt{min}), agregacja (\\texttt{max}), "
        "defuzyfikacja centroid.\\\\"
        "\\textbf{Etap 3.} Wyjaśnienia lokalne (per pacjent: stopnie MF, aktywacje, "
        "wkłady reguł) i globalne (średnie aktywacje w kohorcie).",
    )
    ctx.section(
        "IMPLEMENTACJA",
        "Plik \\texttt{lab7/lab7\\_solution.py}. Klasy: \\texttt{CrispRule}, "
        "\\texttt{FuzzyRule}. Funkcje MF: \\texttt{mu\\_age\\_elderly}, "
        "\\texttt{mu\\_sbp\\_high}, \\dots. Implementacja od podstaw, bez "
        "scikit-fuzzy, dla pełnej przezroczystości.",
    )
    ctx.section(
        "OBLICZENIA",
        "Reguły klasyczne:\n\n"
        + df_to_latex(rule_table, "Reguły klasyczne.", "crisp", float_format="%.2f")
        + "\nReguły rozmyte:\n\n"
        + df_to_latex(fuzzy_rule_table, "Reguły rozmyte (Mamdani).", "fuzzy")
        + "\nWyniki dla wszystkich pacjentów (skrót):\n\n"
        + df_to_latex(short_summary, "Wnioski klasyczne i rozmyte dla każdego pacjenta.", "all", float_format="%.2f"),
    )
    ctx.section(
        "WYNIKI",
        f"Z $N={len(df)}$ pacjentów: liczba klasyfikowanych jako geriatryczni "
        f"(klasyczne) = \\textbf{{{n_geriatric_classic}}}. "
        f"Średnie ryzyko (fuzzy) w grupie geriatrycznej = \\textbf{{{avg_risk_geriatric:.1f}}}, "
        f"poza nią = \\textbf{{{avg_risk_non:.1f}}}.\n\n"
        "\\textbf{Wyjaśnienie dla pacjenta " + sample_id + ":}\n\n"
        + explanation_sample,
    )
    ctx.section(
        "WYKRESY",
        figure_block(fig_age, "Funkcje przynależności dla wieku.", "mf-age")
        + figure_block(fig_others, "Pozostałe funkcje przynależności (SBP, DBP, glukoza, BMI).", "mf-rest")
        + figure_block(fig_global, "Średnie aktywacje reguł w kohorcie.", "global-act")
        + figure_block(fig_risk_age, "Ryzyko fuzzy vs wiek z klasyczną decyzją.", "risk-age")
        + figure_block(fig_agg_sample, f"Agregacja Mamdaniego --- pacjent {sample_id}.", "agg-sample"),
    )
    ctx.section(
        "INTERPRETACJA",
        "Próg wieku $\\ge$ 70 jest sztywny: pacjent z wiekiem 69 nie jest klasyfikowany jako "
        "geriatryczny w wariancie binarnym, mimo że fuzzy MF \\texttt{elderly} aktywuje się "
        "częściowo (\\texttt{age=69} $\\to$ $\\mu$ \\textgreater 0.9). System rozmyty "
        "łagodzi tę nieciągłość. Reguły F2--F4 (geriatric AND komorbidność) zwiększają "
        "ryzyko gdy pacjent w podeszłym wieku ma dodatkowe czynniki. Wyjaśnienia lokalne "
        "pokazują dokładnie, które reguły i w jakim stopniu zawiodły o końcowej ocenie ryzyka.",
    )
    ctx.section(
        "WNIOSKI",
        "Połączenie reguł klasycznych z rozmytymi pozwala na: (a) zachowanie zgodności "
        "z wytycznymi klinicznymi (sztywne progi), (b) realistyczne uchwycenie pacjentów "
        "granicznych, (c) interpretowalne wyjaśnienia decyzji. Wkład wieku w końcową "
        "ocenę ryzyka jest dominujący dla pacjentów geriatrycznych, ale tylko w połączeniu "
        "z innymi czynnikami klinicznymi (SBP, glukoza, BMI). Ograniczenia: brak uczenia "
        "się na danych, ręczne kalibrowanie MF, możliwa niespójność reguł.",
    )

    pdf = render_report(OUTPUT_DIR, "lab7_report.tex", ctx)
    print(f"OK -> {pdf}")


if __name__ == "__main__":
    main()
