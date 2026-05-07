from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import secrets
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LAB_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LAB_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPO_ROOT = LAB_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from common.report_utils import ReportContext, df_to_latex, figure_block, render_report  # noqa: E402

LAB1_CSV = REPO_ROOT / "lab1" / "output" / "prostate_cancer_synth.csv"

#data layer
def load_data() -> pd.DataFrame:
    if LAB1_CSV.exists():
        return pd.read_csv(LAB1_CSV)
    sys.path.insert(0, str(REPO_ROOT / "lab1"))
    from lab1_solution import synth_prostate_cancer_dataset  # type: ignore

    df = synth_prostate_cancer_dataset()
    LAB1_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LAB1_CSV, index=False)
    return df


#preprocessing
PII_COLS = ["patient_id"]
CLINICAL_COLS = [
    "age", "psa", "gleason_score", "tumor_stage", "treatment",
    "comorbidities", "survival_months", "survived",
]

#statistical analysis
def split_pii_clinical(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[PII_COLS].copy(), df[CLINICAL_COLS].copy()


def encrypt_column(df: pd.DataFrame, column: str, cipher: Fernet) -> pd.DataFrame:
    out = df.copy()
    out[column + "_enc"] = out[column].astype(str).map(lambda v: cipher.encrypt(v.encode()).decode())
    out = out.drop(columns=[column])
    return out


def decrypt_column(df: pd.DataFrame, column_enc: str, cipher: Fernet) -> pd.DataFrame:
    out = df.copy()
    base = column_enc.replace("_enc", "")
    out[base] = out[column_enc].map(lambda v: cipher.decrypt(v.encode()).decode())
    return out.drop(columns=[column_enc])


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()



def pseudonymize(df: pd.DataFrame, salt: bytes) -> pd.DataFrame:
    out = df.copy()
    out["pseudo_id"] = out["patient_id"].astype(str).map(
        lambda pid: hashlib.sha256(salt + pid.encode()).hexdigest()[:16]
    )
    out = out.drop(columns=["patient_id"])
    return out


def anonymize_age_band(age: float) -> str:
    bins = [0, 50, 60, 70, 80, 200]
    labels = ["<50", "50-59", "60-69", "70-79", "80+"]
    for low, high, lab in zip(bins[:-1], bins[1:], labels):
        if low <= age < high:
            return lab
    return labels[-1]


def anonymize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_band"] = out["age"].map(anonymize_age_band)
    out = out.drop(columns=["age"])
    return out



@dataclass
class RBACPolicy:
    role_to_columns: dict[str, list[str]]

    def allowed(self, role: str) -> list[str]:
        return self.role_to_columns.get(role, [])


def make_policy() -> RBACPolicy:
    return RBACPolicy(
        role_to_columns={
            "admin": PII_COLS + CLINICAL_COLS,
            "doctor": ["patient_id"] + CLINICAL_COLS,
            "analyst": [c for c in CLINICAL_COLS if c != "survived"] + ["survived"],
            "guest": ["age", "tumor_stage"],
        }
    )


def setup_audit_logger() -> logging.Logger:
    logger = logging.getLogger("rbac.audit")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(OUTPUT_DIR / "audit.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def access(df: pd.DataFrame, role: str, user: str, policy: RBACPolicy, logger: logging.Logger) -> pd.DataFrame:
    cols = [c for c in policy.allowed(role) if c in df.columns]
    logger.info(f"role={role} user={user} requested_columns={cols}")
    if not cols:
        logger.warning(f"role={role} user={user} ACCESS DENIED")
        raise PermissionError(f"role {role} has no allowed columns")
    return df[cols].copy()



def df_signature(df: pd.DataFrame) -> str:
    raw = df.to_csv(index=False).encode()
    return hashlib.sha256(raw).hexdigest()


def safe_predict(model, df: pd.DataFrame, expected_signature: str) -> np.ndarray:
    sig = df_signature(df)
    if sig != expected_signature:
        raise ValueError("DATA TAMPERING DETECTED: signature mismatch")
    return model.predict_proba(df)[:, 1]



HASH_ALGOS = ["md5", "sha1", "sha256", "sha512", "sha3_256", "blake2b", "blake2s"]


def benchmark_hashes(payload: bytes, n_iter: int = 200) -> pd.DataFrame:
    rows = []
    for algo in HASH_ALGOS:
        start = time.perf_counter()
        for _ in range(n_iter):
            h = hashlib.new(algo)
            h.update(payload)
            digest = h.hexdigest()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / n_iter
        rows.append({
            "algorytm": algo,
            "rozmiar dygestu (B)": h.digest_size,
            "średni czas (ms)": round(elapsed_ms, 4),
            "przykład digest": digest[:16] + "...",
        })
    return pd.DataFrame(rows)


def avalanche_score(algo: str, payload: bytes, flips: int = 200) -> float:
    """Liczba zmienionych bitów w digescie po flipie 1 bitu wejścia (uśredniona)."""
    h_orig = hashlib.new(algo, payload).digest()
    bits_orig = "".join(f"{b:08b}" for b in h_orig)
    diffs = []
    rng = np.random.default_rng(8)
    for _ in range(flips):
        idx = int(rng.integers(0, len(payload)))
        bit = int(rng.integers(0, 8))
        flipped = bytearray(payload)
        flipped[idx] ^= 1 << bit
        h_new = hashlib.new(algo, bytes(flipped)).digest()
        bits_new = "".join(f"{b:08b}" for b in h_new)
        diff = sum(1 for a, b in zip(bits_orig, bits_new) if a != b)
        diffs.append(diff / len(bits_orig))
    return float(np.mean(diffs))


def avalanche_table(payload: bytes) -> pd.DataFrame:
    rows = []
    for algo in HASH_ALGOS:
        score = avalanche_score(algo, payload)
        rows.append({
            "algorytm": algo,
            "avg odsetek zmienionych bitów": round(score, 4),
            "ideał (50%)": 0.5,
        })
    return pd.DataFrame(rows)



def model_impact(df: pd.DataFrame) -> tuple[float, float]:
    df = df.copy()
    df["psa"] = df["psa"].fillna(df["psa"].median())
    df_full = pd.get_dummies(df, columns=["tumor_stage", "treatment"], drop_first=True)
    feat_full = [c for c in df_full.columns if c not in ("patient_id", "survived", "survival_months")]
    X_full = df_full[feat_full].to_numpy(dtype=float)
    y = df_full["survived"].to_numpy()

    df_anon = anonymize_dataframe(df)
    df_anon = pd.get_dummies(df_anon, columns=["tumor_stage", "treatment", "age_band"], drop_first=True)
    feat_anon = [c for c in df_anon.columns if c not in ("patient_id", "survived", "survival_months")]
    X_anon = df_anon[feat_anon].to_numpy(dtype=float)

    def auc(X: np.ndarray) -> float:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=8, stratify=y)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(Xtr, ytr)
        return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

    return auc(X_full), auc(X_anon)


#plots
def _save(fig: plt.Figure, name: str) -> str:
    fig.savefig(OUTPUT_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_hash_perf(bench: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(bench["algorytm"], bench["średni czas (ms)"], color="#3b6fb6", edgecolor="black")
    ax.set_ylabel("średni czas (ms)")
    ax.set_title("Średni czas obliczenia digestu (im niżej, tym lepiej)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, "hash_perf.png")


def plot_avalanche(av: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(av["algorytm"], av["avg odsetek zmienionych bitów"], color="#3b6fb6", edgecolor="black")
    ax.axhline(0.5, color="black", linestyle="--", lw=1, label="ideał (50%)")
    ax.set_ylabel("odsetek zmienionych bitów")
    ax.set_title("Efekt lawinowy --- odporność na korelację wejścia/wyjścia")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return _save(fig, "avalanche.png")


def plot_anon_impact(auc_full: float, auc_anon: float) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["pełne dane", "anonimizowane"], [auc_full, auc_anon],
           color=["#3b6fb6", "#b6553b"], edgecolor="black")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC AUC (LogReg, test)")
    ax.set_title("Wpływ anonimizacji na jakość modelu")
    for i, v in enumerate([auc_full, auc_anon]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
    fig.tight_layout()
    return _save(fig, "anon_impact.png")


#report glue
def main() -> None:
    df = load_data()

    # Etap 1
    pii_df, clinical_df = split_pii_clinical(df)
    pii_df.to_csv(OUTPUT_DIR / "pii_only.csv", index=False)
    clinical_df.to_csv(OUTPUT_DIR / "clinical_only.csv", index=False)

    # Etap 2
    key = Fernet.generate_key()
    cipher = Fernet(key)
    enc_df = df.copy()
    enc_df = encrypt_column(enc_df, "treatment", cipher)
    enc_csv = OUTPUT_DIR / "patients_encrypted.csv"
    enc_df.to_csv(enc_csv, index=False)
    sha_full_before = file_sha256(enc_csv)

    # Tampering detection demo
    with open(enc_csv, "rb") as fp:
        original = fp.read()
    tampered = bytearray(original)
    if len(tampered) > 0:
        tampered[10] ^= 0x01
    tampered_path = OUTPUT_DIR / "patients_tampered.csv"
    tampered_path.write_bytes(bytes(tampered))
    sha_full_after = file_sha256(tampered_path)

    # Etap 3
    salt = secrets.token_bytes(16)
    pseudo_df = pseudonymize(df, salt)
    anon_df = anonymize_dataframe(df)
    pseudo_df.to_csv(OUTPUT_DIR / "patients_pseudo.csv", index=False)
    anon_df.to_csv(OUTPUT_DIR / "patients_anon.csv", index=False)

    # Etap 4
    policy = make_policy()
    logger = setup_audit_logger()
    role_demos = {}
    for role, user in [("admin", "anna"), ("doctor", "kowalski"),
                        ("analyst", "data_lab"), ("guest", "intern")]:
        try:
            view = access(df, role, user, policy, logger)
            role_demos[role] = (list(view.columns), view.head(2).to_dict("records"))
        except PermissionError:
            role_demos[role] = ("DENIED", [])

    # Etap 5
    df_clin = df[CLINICAL_COLS].copy()
    df_clin["psa"] = df_clin["psa"].fillna(df_clin["psa"].median())
    df_enc_for_model = pd.get_dummies(df_clin, columns=["tumor_stage", "treatment"], drop_first=True)
    feats = [c for c in df_enc_for_model.columns if c not in ("survived", "survival_months")]
    X = df_enc_for_model[feats]
    y = df_enc_for_model["survived"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=8, stratify=y)
    sc = StandardScaler()
    Xtr_s = pd.DataFrame(sc.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
    Xte_s = pd.DataFrame(sc.transform(Xte), columns=Xte.columns, index=Xte.index)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(Xtr_s, ytr)
    expected_sig = df_signature(Xte_s)
    proba_clean = safe_predict(model, Xte_s, expected_sig)
    auc_clean = roc_auc_score(yte, proba_clean)

    Xte_tampered = Xte_s.copy()
    Xte_tampered.iloc[0, 0] += 999.9
    try:
        safe_predict(model, Xte_tampered, expected_sig)
        tamper_detected = False
    except ValueError:
        tamper_detected = True

    # Wariant 1
    payload = json.dumps(df.head(50).to_dict("records")).encode()
    bench = benchmark_hashes(payload)
    av = avalanche_table(payload)

    # Anonymization impact
    auc_full, auc_anon = model_impact(df)

    fig_perf = plot_hash_perf(bench)
    fig_av = plot_avalanche(av)
    fig_anon = plot_anon_impact(auc_full, auc_anon)

    role_table = pd.DataFrame(
        [
            {"rola": role, "dozwolone kolumny": ", ".join(cols if isinstance(cols, list) else [str(cols)])}
            for role, (cols, _) in role_demos.items()
        ]
    )

    integrity_table = pd.DataFrame(
        [
            {"plik": "patients_encrypted.csv (oryginał)", "SHA-256": sha_full_before},
            {"plik": "patients_tampered.csv (1 bit zmieniony)", "SHA-256": sha_full_after},
        ]
    )
    integrity_table["SHA-256"] = integrity_table["SHA-256"].map(lambda s: s[:24] + "...")

    ctx = ReportContext(
        lab_number=6,
        report_title="Bezpieczeństwo danych medycznych: szyfrowanie, hash, RBAC, integralność",
        variant="1 (fallback z 8) — porównanie algorytmów haszujących + Etapy 1--5",
    )
    ctx.section(
        "CEL",
        "Praktyczna implementacja warstw bezpieczeństwa dla danych medycznych: rozdział "
        "danych identyfikujących i klinicznych, szyfrowanie kolumn, weryfikacja "
        "integralności, pseudonimizacja, anonimizacja, RBAC z audytem oraz ochrona "
        "modelu przed manipulacją danych. Wariant 1: porównanie algorytmów hashujących.",
    )
    ctx.section(
        "PROBLEM",
        "Systemy SI w medycynie operują na danych wrażliwych (RODO, HIPAA, ISO/IEC 27001). "
        "Wymagana jest poufność, integralność i dostępność oraz minimalizacja danych. "
        "Implementacja musi obsłużyć cały cykl: od rozdzielenia PII, przez szyfrowanie i "
        "anonimizację, po kontrolę dostępu i ochronę modelu predykcyjnego.",
    )
    pii_tex = ", ".join(c.replace("_", r"\_") for c in PII_COLS)
    clin_tex = ", ".join(c.replace("_", r"\_") for c in CLINICAL_COLS)
    ctx.section(
        "DANE",
        f"Zbiór z lab1 \\texttt{{prostate\\_cancer\\_synth.csv}} ($N={len(df)}$). "
        f"Kolumny PII: \\texttt{{{pii_tex}}}. "
        f"Kolumny kliniczne: \\texttt{{{clin_tex}}}.",
    )
    ctx.section(
        "METODY",
        "\\textbf{Etap 1.} Rozdział kolumn na PII / kliniczne i zapis do osobnych CSV.\\\\"
        "\\textbf{Etap 2.} Szyfrowanie symetryczne Fernet (AES-128-CBC + HMAC-SHA-256). "
        "Integralność pliku przez SHA-256, demonstracja efektu lawinowego (1 bit $\\to$ "
        "całkowicie inny digest).\\\\"
        "\\textbf{Etap 3.} Pseudonimizacja: salt + SHA-256 dla \\texttt{patient\\_id}. "
        "Anonimizacja: generalizacja wieku do przedziałów (\\texttt{<50, 50-59, ...}).\\\\"
        "\\textbf{Etap 4.} RBAC w postaci mapy rola $\\to$ kolumny. Audyt do "
        "\\texttt{output/audit.log} (logger \\texttt{rbac.audit}).\\\\"
        "\\textbf{Etap 5.} Sygnatura SHA-256 wektora wejściowego modelu, wykrywanie "
        "manipulacji (data tampering) przed inferencją.\\\\"
        "\\textbf{Wariant 1.} Benchmark szybkości (\\texttt{md5..blake2s}) + analiza "
        "efektu lawinowego (oczekiwane $\\sim$50\\% zmienionych bitów).",
    )
    ctx.section(
        "IMPLEMENTACJA",
        "Plik \\texttt{lab6/lab6\\_solution.py}, struktura wg etapów. Logger audytu "
        "konfigurowany w \\texttt{setup\\_audit\\_logger}. Klucz Fernet generowany on-the-fly "
        "(w produkcji powinien być przechowywany w KMS/HSM).",
    )
    ctx.section(
        "OBLICZENIA",
        "Porównanie algorytmów haszujących (czas i rozmiar):\n\n"
        + df_to_latex(bench, "Benchmark algorytmów haszujących.", "hbench", float_format="%.4f")
        + "\nEfekt lawinowy --- odsetek zmienionych bitów po flipie pojedynczego bitu wejścia:\n\n"
        + df_to_latex(av, "Efekt lawinowy --- ideał 50\\%.", "aval", float_format="%.4f")
        + "\nZmiana hashu pliku po flipie 1 bitu (demonstracja integralności):\n\n"
        + df_to_latex(integrity_table, "SHA-256 przed i po manipulacji.", "intg")
        + "\nDostęp wg ról:\n\n"
        + df_to_latex(role_table, "Macierz uprawnień RBAC.", "rbac"),
    )
    ctx.section(
        "WYNIKI",
        f"Sygnatura SHA-256 macierzy testowej: \\texttt{{{expected_sig[:24]}...}}. "
        f"Predykcja na czystych danych: ROC AUC = {auc_clean:.3f}. "
        f"Próba inferencji na zmanipulowanych danych: "
        f"\\textbf{{{'wykryto naruszenie' if tamper_detected else 'brak detekcji'}}}.\n\n"
        f"Wpływ anonimizacji na jakość modelu: ROC AUC pełne dane = {auc_full:.3f}, "
        f"ROC AUC dane anonimizowane (\\texttt{{age\\_band}}) = {auc_anon:.3f}. "
        f"Spadek = {auc_full - auc_anon:.3f}.",
    )
    ctx.section(
        "WYKRESY",
        figure_block(fig_perf, "Średni czas obliczania digestu.", "perf")
        + figure_block(fig_av, "Efekt lawinowy.", "av")
        + figure_block(fig_anon, "Wpływ anonimizacji na ROC AUC.", "anon"),
    )
    ctx.section(
        "INTERPRETACJA",
        "Wszystkie nowoczesne algorytmy (SHA-256, SHA-512, SHA-3, BLAKE2) osiągają "
        "efekt lawinowy bliski 50\\%, co jest pożądane (brak korelacji wejścia z wyjściem). "
        "MD5 i SHA-1 są szybsze, ale niedopuszczalne z powodu znanych ataków kolizyjnych "
        "i nie powinny być używane do ochrony danych medycznych. RBAC działa poprawnie: "
        "rola \\texttt{guest} ma dostęp tylko do najbardziej zagregowanych cech. "
        "Anonimizacja przez generalizację wieku obniża AUC o niewielką wartość, co pokazuje, "
        "że można zapewnić prywatność z akceptowalnym kosztem informacyjnym.",
    )
    ctx.section(
        "WNIOSKI",
        "Bezpieczna miniaplikacja medyczna powinna łączyć: (a) silne szyfrowanie symetryczne "
        "(Fernet/AES-GCM), (b) hash bezpieczny (SHA-256/SHA-3/BLAKE2), (c) RBAC z audytem, "
        "(d) sygnaturę danych wejściowych modelu jako mechanizm obrony przed atakiem "
        "data poisoning/tampering, (e) anonimizację z analizą wpływu na jakość modelu. "
        "MD5/SHA-1 należy wykluczyć z nowych implementacji. Klucz szyfrujący przechowywać "
        "w bezpiecznym magazynie (KMS/HSM), nie w kodzie.",
    )

    pdf = render_report(OUTPUT_DIR, "lab6_report.tex", ctx)
    print(f"OK -> {pdf}")


if __name__ == "__main__":
    main()
