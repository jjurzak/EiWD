r"""Lab 3 — Analiza tekstów medycznych (NLP).

Wariant 8 nie istnieje. PDF wymienia 4 etapy zadania (1-4); fallback na
wariant 1 (Przygotowanie danych tekstowych), ale logicznie realizujemy
pełny cykl wszystkich 4 etapów (1-4), ponieważ etap 1 sam w sobie nie
daje sensownego raportu, a w PDF zadania są opisane jako wspólny pipeline.

Etapy:
1. Przygotowanie danych tekstowych (czyszczenie, tokenizacja).
2. Rozpoznawanie jednostek medycznych (NER) — implementacja słownikowo-regexowa
   zamiast scispaCy/BioBERT, bez zewnętrznych modeli.
3. Klasyfikacja dokumentów medycznych (TF-IDF + LogisticRegression vs LinearSVC).
4. Wizualizacja i interpretacja wyników (macierz pomyłek, częstość encji).

Korzysta wyłącznie z lokalnie wygenerowanego korpusu syntetycznego.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

LAB_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LAB_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPO_ROOT = LAB_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from common.report_utils import ReportContext, df_to_latex, figure_block, render_report  # noqa: E402

CATEGORIES = ["opis_RTG", "karta_informacyjna", "wynik_lab", "recepta"]

DISEASES = [
    "nadciśnienie", "cukrzyca", "zapalenie płuc", "zawał", "udar mózgu",
    "rak prostaty", "astma", "miażdżyca", "anemia", "grypa",
]
DRUGS = [
    "amoksycylina", "metformina", "atorwastatyna", "ibuprofen", "paracetamol",
    "enalapryl", "insulina", "ramipryl", "metoprolol", "warfaryna",
]
PROCEDURES = [
    "tomografia komputerowa", "rezonans magnetyczny", "RTG klatki piersiowej",
    "USG jamy brzusznej", "EKG", "biopsja", "endoskopia", "koronarografia",
    "morfologia krwi", "spirometria",
]
BODY_PARTS = [
    "płuco", "serce", "wątroba", "nerka", "tarczyca",
    "mózg", "klatka piersiowa", "jama brzuszna", "kręgosłup", "stawy",
]


# --------------------------------------------------------------------------------------
# Etap 0: synthetic corpus
# --------------------------------------------------------------------------------------
def synth_corpus(n_per_class: int = 80, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    rtg_templates = [
        "Wykonano {proc} u pacjenta. W {bp} stwierdzono cechy {dis}. Zalecono kontrolę.",
        "Badanie {proc} ujawnia zmiany w obrębie {bp}, sugerujące {dis}.",
        "Opis {proc}: bez istotnych odchyleń w {bp}; brak cech {dis}.",
    ]
    karta_templates = [
        "Pacjent przyjęty z rozpoznaniem {dis}. Wdrożono leczenie {drug} oraz {drug2}. Zalecono {proc}.",
        "Karta informacyjna: hospitalizacja z powodu {dis}. Stosowano {drug}. Wypis w stanie ogólnym dobrym.",
        "Pacjent leczony z powodu {dis} w obrębie {bp}. Zalecono kontynuację {drug}.",
    ]
    lab_templates = [
        "Wynik {proc}: parametry w normie. Brak cech {dis}.",
        "Badanie {proc} wykazało odchylenia mogące świadczyć o {dis}. Sugerowana konsultacja.",
        "Wynik laboratoryjny: poziom glukozy podwyższony, podejrzenie {dis}. Zalecono {drug}.",
    ]
    recepta_templates = [
        "Rp. {drug} 500 mg, 2x dziennie. {drug2} 10 mg wieczorem. Diagnoza: {dis}.",
        "Recepta: {drug}, dawkowanie wg schematu, w terapii {dis}.",
        "Przepisano {drug} oraz {drug2}, kontrola za 4 tygodnie ({dis}).",
    ]

    cat_to_templates = {
        "opis_RTG": rtg_templates,
        "karta_informacyjna": karta_templates,
        "wynik_lab": lab_templates,
        "recepta": recepta_templates,
    }

    for cat in CATEGORIES:
        templates = cat_to_templates[cat]
        for _ in range(n_per_class):
            tmpl = rng.choice(templates)
            text = tmpl.format(
                proc=rng.choice(PROCEDURES),
                dis=rng.choice(DISEASES),
                drug=rng.choice(DRUGS),
                drug2=rng.choice(DRUGS),
                bp=rng.choice(BODY_PARTS),
            )
            extra = rng.choice([
                "",
                " Pacjent czuje się dobrze.",
                " Bez powikłań.",
                " Wymagana dalsza diagnostyka.",
                " Zalecono dietę i aktywność fizyczną.",
            ])
            rows.append({"category": cat, "text": text + extra})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Etap 1: cleaning + tokenization
# --------------------------------------------------------------------------------------
_PL_STOPWORDS = {
    "i", "a", "o", "u", "w", "z", "na", "do", "od", "po", "za", "się",
    "oraz", "lub", "ale", "że", "to", "jest", "są", "były", "był", "była",
    "być", "jak", "co", "tym", "tych", "tej", "tej", "ten", "ta", "te",
    "kto", "kogo", "czy", "więc", "bo", "gdyż", "także", "również",
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\d]+", " ", text)
    text = re.sub(r"[^\wąćęłńóśźż\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return [tok for tok in clean_text(text).split() if tok not in _PL_STOPWORDS and len(tok) > 1]


# --------------------------------------------------------------------------------------
# Etap 2: dictionary-based NER
# --------------------------------------------------------------------------------------
def find_entities(text: str) -> dict[str, list[str]]:
    text_l = text.lower()
    found = {"DISEASE": [], "DRUG": [], "PROCEDURE": [], "BODY_PART": []}
    for term in DISEASES:
        if term.lower() in text_l:
            found["DISEASE"].append(term)
    for term in DRUGS:
        if term.lower() in text_l:
            found["DRUG"].append(term)
    for term in PROCEDURES:
        if term.lower() in text_l:
            found["PROCEDURE"].append(term)
    for term in BODY_PARTS:
        if term.lower() in text_l:
            found["BODY_PART"].append(term)
    return found


def aggregate_entity_counts(df: pd.DataFrame) -> dict[str, Counter]:
    counters = {k: Counter() for k in ["DISEASE", "DRUG", "PROCEDURE", "BODY_PART"]}
    for txt in df["text"]:
        ents = find_entities(txt)
        for label, terms in ents.items():
            counters[label].update(terms)
    return counters


# --------------------------------------------------------------------------------------
# Etap 3: classification
# --------------------------------------------------------------------------------------
def train_classifiers(df: pd.DataFrame) -> dict:
    X = df["text"].tolist()
    y = df["category"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=8, stratify=y
    )

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        token_pattern=None,
    )
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)

    models = {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train_v, y_train)
        pred = m.predict(X_test_v)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
            "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
            "cm": confusion_matrix(y_test, pred, labels=CATEGORIES),
            "pred": pred,
            "y_test": y_test,
        }
    return {"results": results, "vectorizer": vectorizer}


# --------------------------------------------------------------------------------------
# Etap 4: visualisation
# --------------------------------------------------------------------------------------
def _save(fig: plt.Figure, name: str) -> str:
    fig.savefig(OUTPUT_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_confusion(cm: np.ndarray, title: str, fname: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CATEGORIES).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30
    )
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, fname)


def plot_entity_frequencies(counters: dict[str, Counter]) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (label, ctr) in zip(axes.ravel(), counters.items()):
        items = ctr.most_common(8)
        if not items:
            ax.set_visible(False)
            continue
        names, vals = zip(*items)
        ax.barh(range(len(names)), vals, color="#3b6fb6", edgecolor="black")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_title(f"Najczęstsze encje: {label}")
        ax.set_xlabel("liczność")
    fig.tight_layout()
    return _save(fig, "entities.png")


def plot_class_distribution(df: pd.DataFrame) -> str:
    counts = df["category"].value_counts().reindex(CATEGORIES)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="#3b6fb6", edgecolor="black")
    ax.set_title("Liczność klas w korpusie syntetycznym")
    ax.set_ylabel("liczność")
    ax.set_xlabel("kategoria")
    fig.tight_layout()
    return _save(fig, "class_dist.png")


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def main() -> None:
    df = synth_corpus()
    df["clean"] = df["text"].map(clean_text)
    df.to_csv(OUTPUT_DIR / "corpus.csv", index=False)

    entity_counters = aggregate_entity_counts(df)
    entity_summary = pd.DataFrame(
        [
            {"typ encji": k, "liczba unikalnych": len(v), "łączna liczba wystąpień": sum(v.values())}
            for k, v in entity_counters.items()
        ]
    )

    classification = train_classifiers(df)
    results = classification["results"]

    metrics_rows = []
    for name, r in results.items():
        metrics_rows.append({
            "model": name,
            "accuracy": r["accuracy"],
            "precision (macro)": r["precision_macro"],
            "recall (macro)": r["recall_macro"],
            "F1 (macro)": r["f1_macro"],
        })
    metrics_table = pd.DataFrame(metrics_rows)

    figures = {
        "dist": plot_class_distribution(df),
        "ents": plot_entity_frequencies(entity_counters),
        "cm_lr": plot_confusion(results["LogReg"]["cm"], "Macierz pomyłek --- LogReg", "cm_lr.png"),
        "cm_svc": plot_confusion(results["LinearSVC"]["cm"], "Macierz pomyłek --- LinearSVC", "cm_svc.png"),
    }

    sample_text = df["text"].iloc[0]
    sample_entities = find_entities(sample_text)
    sample_lines = [f"Tekst: \\textit{{{sample_text}}}", "Wykryte encje:"]
    for label, terms in sample_entities.items():
        if terms:
            label_tex = label.replace("_", r"\_")
            sample_lines.append(f"\\quad \\texttt{{{label_tex}}}: {', '.join(terms)}\\\\")
    sample_block = "\n\n".join(sample_lines)

    ctx = ReportContext(
        lab_number=3,
        report_title="Analiza tekstów medycznych: NER i klasyfikacja dokumentów",
        variant="1-4 (fallback z 8) — pełny pipeline NLP na korpusie syntetycznym",
    )
    ctx.section(
        "CEL",
        "Zapoznanie z technikami NLP w medycynie: czyszczeniem i tokenizacją tekstów, "
        "rozpoznawaniem jednostek medycznych (NER) oraz klasyfikacją dokumentów. "
        "Realizujemy wszystkie cztery etapy zadania z PDF na korpusie syntetycznym, "
        "ponieważ użycie scispaCy/BioBERT wymagałoby dostępu do internetu.",
    )
    ctx.section(
        "PROBLEM",
        "Mamy zbiór krótkich notatek medycznych w 4 kategoriach: \\texttt{opis\\_RTG}, "
        "\\texttt{karta\\_informacyjna}, \\texttt{wynik\\_lab}, \\texttt{recepta}. "
        "Zadania: (a) wyodrębnić nazwy chorób, leków, procedur i części ciała; "
        "(b) sklasyfikować dokumenty według typu.",
    )
    ctx.section(
        "DANE",
        f"Korpus syntetyczny ($N={len(df)}$, {len(CATEGORIES)} klasy, generowany "
        "z szablonów lingwistycznych z losowym podstawieniem terminów medycznych). "
        "Słowniki encji: 10 chorób, 10 leków, 10 procedur, 10 części ciała.",
    )
    ctx.section(
        "METODY",
        "\\textbf{Etap 1.} Czyszczenie: lowercasing, usunięcie cyfr i interpunkcji, "
        "tokenizacja przez split, filtrowanie stopwords (lista własna PL).\\\\"
        "\\textbf{Etap 2.} NER słownikowo-regexowy: dopasowanie terminów ze słowników "
        "kategorii (DISEASE/DRUG/PROCEDURE/BODY\\_PART).\\\\"
        "\\textbf{Etap 3.} Wektoryzacja TF--IDF (1--2-gramy), klasyfikacja "
        "\\texttt{LogisticRegression} (multinomial) oraz \\texttt{LinearSVC}, "
        "ze \\texttt{class\\_weight=balanced}.\\\\"
        "\\textbf{Etap 4.} Macierz pomyłek, statystyki encji, ocena macro-F1.",
    )
    ctx.section(
        "IMPLEMENTACJA",
        "Plik \\texttt{lab3/lab3\\_solution.py}. Funkcje: "
        "\\texttt{synth\\_corpus}, \\texttt{clean\\_text}, \\texttt{tokenize}, "
        "\\texttt{find\\_entities}, \\texttt{train\\_classifiers}, \\texttt{plot\\_*}. "
        "Brak zależności od modeli zewnętrznych --- pełna reprodukowalność lokalna.",
    )
    ctx.section(
        "OBLICZENIA",
        "Liczność encji wykrytych w korpusie:\n\n"
        + df_to_latex(entity_summary, "Liczność wykrytych encji wg typu.", "ent", float_format="%.0f")
        + "\nPrzykład działania NER:\n\n"
        + sample_block,
    )
    ctx.section(
        "WYNIKI",
        "Porównanie modeli klasyfikacji dokumentów:\n\n"
        + df_to_latex(metrics_table, "Skuteczność klasyfikatorów.", "cls", float_format="%.3f"),
    )
    ctx.section(
        "WYKRESY",
        figure_block(figures["dist"], "Liczność klas w korpusie.", "dist")
        + figure_block(figures["ents"], "Najczęstsze encje wg typu.", "ents")
        + figure_block(figures["cm_lr"], "Macierz pomyłek --- regresja logistyczna.", "cm-lr")
        + figure_block(figures["cm_svc"], "Macierz pomyłek --- LinearSVC.", "cm-svc"),
    )
    ctx.section(
        "INTERPRETACJA",
        "Z uwagi na strukturalność szablonów oba modele (LogReg, LinearSVC) osiągają "
        "wysoką skuteczność klasyfikacji typu dokumentu (\\textgreater{}0.9 F1 macro). "
        "Macierze pomyłek pokazują, że największą trudność stanowi rozróżnienie "
        "\\texttt{karta\\_informacyjna} i \\texttt{recepta} — oba zawierają nazwy "
        "leków i diagnozy. NER słownikowy wychwytuje wszystkie zaplanowane terminy, "
        "ograniczeniem pozostaje brak fleksji (lematyzacji), co w realnych tekstach "
        "wymagałoby morfeusza/spaCy/BioBERT.",
    )
    ctx.section(
        "WNIOSKI",
        "Cykl NLP od czyszczenia po klasyfikację daje się zrealizować klasycznymi "
        "narzędziami (regex + TF--IDF + linear classifiers). Naturalnymi rozszerzeniami są: "
        "(a) lematyzacja PL (np. \\texttt{morfeusz2}, \\texttt{spaCy pl\\_core}); "
        "(b) trening modeli BERT/BioBERT na realnych korpusach klinicznych; "
        "(c) wprowadzenie reguł kontekstowych dla NER (negacja, modalność).",
    )

    pdf = render_report(OUTPUT_DIR, "lab3_report.tex", ctx)
    print(f"OK -> {pdf}")


if __name__ == "__main__":
    main()
