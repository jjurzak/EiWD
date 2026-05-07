from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAB_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LAB_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPO_ROOT = LAB_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from common.report_utils import ReportContext, df_to_latex, figure_block, render_report  # noqa: E402

STATES = ["Healthy", "MildInfection", "SevereInfection"]
ACTIONS = ["Observe", "Antibiotic", "Hospitalize"]
GAMMA = 0.95

S_IDX = {s: i for i, s in enumerate(STATES)}
A_IDX = {a: i for i, a in enumerate(ACTIONS)}

#data layer

def build_mdp() -> tuple[np.ndarray, np.ndarray]:
    """Zwraca: P[s,a,s'] (transition), R[s,a,s'] (reward)."""
    nS, nA = len(STATES), len(ACTIONS)
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA, nS))

    transitions = {
        ("MildInfection", "Observe"): {"MildInfection": 0.60, "SevereInfection": 0.30, "Healthy": 0.10},
        ("MildInfection", "Antibiotic"): {"Healthy": 0.55, "MildInfection": 0.40, "SevereInfection": 0.05},
        ("MildInfection", "Hospitalize"): {"Healthy": 0.70, "MildInfection": 0.25, "SevereInfection": 0.05},
        ("SevereInfection", "Observe"): {"SevereInfection": 0.75, "MildInfection": 0.20, "Healthy": 0.05},
        ("SevereInfection", "Antibiotic"): {"MildInfection": 0.55, "Healthy": 0.25, "SevereInfection": 0.20},
        ("SevereInfection", "Hospitalize"): {"MildInfection": 0.55, "Healthy": 0.35, "SevereInfection": 0.10},
        ("Healthy", "Observe"): {"Healthy": 0.90, "MildInfection": 0.10},
        ("Healthy", "Antibiotic"): {"Healthy": 0.92, "MildInfection": 0.08},
        ("Healthy", "Hospitalize"): {"Healthy": 0.93, "MildInfection": 0.07},
    }
    action_costs = {"Observe": 0.0, "Antibiotic": -2.0, "Hospitalize": -4.0}
    transition_rewards = {"Healthy": 9.0, "MildInfection": 0.0, "SevereInfection": -7.0}

    for (s, a), dist in transitions.items():
        si = S_IDX[s]; ai = A_IDX[a]
        for sp, p in dist.items():
            spi = S_IDX[sp]
            P[si, ai, spi] = p
            R[si, ai, spi] = transition_rewards[sp] + action_costs[a]
    return P, R


#value iteration
def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float = GAMMA, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    nS, nA, _ = P.shape
    V = np.zeros(nS)
    while True:
        Q = np.einsum("ijk,ijk->ij", P, R) + gamma * (P @ V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = np.einsum("ijk,ijk->ij", P, R) + gamma * (P @ V)
    pi = Q.argmax(axis=1)
    return V, Q, pi  # type: ignore[return-value]


#tabular Q-learning
def step(P: np.ndarray, R: np.ndarray, s: int, a: int, rng: np.random.Generator) -> tuple[int, float]:
    sp = int(rng.choice(P.shape[2], p=P[s, a]))
    r = R[s, a, sp]
    return sp, r


def q_learning(
    P: np.ndarray,
    R: np.ndarray,
    n_episodes: int = 4000,
    max_steps: int = 50,
    alpha: float = 0.2,
    gamma: float = GAMMA,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    seed: int = 8,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    nS, nA, _ = P.shape
    Q = np.zeros((nS, nA))
    eps = eps_start
    eps_decay = (eps_start - eps_end) / n_episodes
    rewards: list[float] = []
    for ep in range(n_episodes):
        s = int(rng.integers(0, nS))
        total = 0.0
        for _ in range(max_steps):
            if rng.random() < eps:
                a = int(rng.integers(0, nA))
            else:
                a = int(np.argmax(Q[s]))
            sp, r = step(P, R, s, a, rng)
            Q[s, a] += alpha * (r + gamma * Q[sp].max() - Q[s, a])
            total += r
            s = sp
        eps = max(eps_end, eps - eps_decay)
        rewards.append(total)
    return Q, rewards


#dqn-replacement via MLPRegressor (function approximator)
def state_features(s: int) -> np.ndarray:
    """One-hot state encoding (3 cech)."""
    f = np.zeros(len(STATES))
    f[s] = 1.0
    return f


def fitted_q_iteration(
    P: np.ndarray,
    R: np.ndarray,
    n_iters: int = 30,
    n_samples: int = 5000,
    gamma: float = GAMMA,
    seed: int = 8,
) -> tuple:
    """Approximate Q with separate MLP per action; iterative target update (DQN spirit)."""
    from sklearn.neural_network import MLPRegressor

    rng = np.random.default_rng(seed)
    nS, nA, _ = P.shape
    feats = np.vstack([state_features(s) for s in range(nS)])
    Q_table = np.zeros((nS, nA))

    samples_s: list[int] = []
    samples_a: list[int] = []
    samples_sp: list[int] = []
    samples_r: list[float] = []
    for _ in range(n_samples):
        s = int(rng.integers(0, nS))
        a = int(rng.integers(0, nA))
        sp, r = step(P, R, s, a, rng)
        samples_s.append(s); samples_a.append(a); samples_sp.append(sp); samples_r.append(r)
    samples_s = np.array(samples_s); samples_a = np.array(samples_a)
    samples_sp = np.array(samples_sp); samples_r = np.array(samples_r)

    models: list[MLPRegressor] = []
    for ai in range(nA):
        m = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=200, random_state=seed + ai, warm_start=True)
        models.append(m)

    losses: list[float] = []
    for it in range(n_iters):
        Q_next = np.zeros((nS, nA))
        for ai in range(nA):
            if it == 0:
                Q_next[:, ai] = 0
            else:
                Q_next[:, ai] = models[ai].predict(feats)
        V_next = Q_next.max(axis=1)

        for ai in range(nA):
            mask = samples_a == ai
            if mask.sum() < 5:
                continue
            X_train = feats[samples_s[mask]]
            y_target = samples_r[mask] + gamma * V_next[samples_sp[mask]]
            models[ai].fit(X_train, y_target)
            preds = models[ai].predict(X_train)
            losses.append(float(np.mean((preds - y_target) ** 2)))

    Q_final = np.zeros((nS, nA))
    for ai in range(nA):
        Q_final[:, ai] = models[ai].predict(feats)
    return Q_final, losses


#explainable RL
def counterfactual_table(Q: np.ndarray) -> pd.DataFrame:
    rows = []
    for s in range(Q.shape[0]):
        order = np.argsort(-Q[s])
        best = order[0]
        second = order[1]
        rows.append({
            "stan": STATES[s],
            "akcja_best": ACTIONS[best],
            "Q_best": round(Q[s, best], 3),
            "alt": ACTIONS[second],
            "Q_alt": round(Q[s, second], 3),
            "deltaQ": round(Q[s, best] - Q[s, second], 3),
        })
    return pd.DataFrame(rows)


def sensitivity_analysis(P_orig: np.ndarray, R_orig: np.ndarray, hospital_costs: list[float]) -> pd.DataFrame:
    base_action_costs = {"Observe": 0.0, "Antibiotic": -2.0}
    rows = []
    transitions = {
        ("MildInfection", "Observe"): {"MildInfection": 0.60, "SevereInfection": 0.30, "Healthy": 0.10},
        ("MildInfection", "Antibiotic"): {"Healthy": 0.55, "MildInfection": 0.40, "SevereInfection": 0.05},
        ("MildInfection", "Hospitalize"): {"Healthy": 0.70, "MildInfection": 0.25, "SevereInfection": 0.05},
        ("SevereInfection", "Observe"): {"SevereInfection": 0.75, "MildInfection": 0.20, "Healthy": 0.05},
        ("SevereInfection", "Antibiotic"): {"MildInfection": 0.55, "Healthy": 0.25, "SevereInfection": 0.20},
        ("SevereInfection", "Hospitalize"): {"MildInfection": 0.55, "Healthy": 0.35, "SevereInfection": 0.10},
        ("Healthy", "Observe"): {"Healthy": 0.90, "MildInfection": 0.10},
        ("Healthy", "Antibiotic"): {"Healthy": 0.92, "MildInfection": 0.08},
        ("Healthy", "Hospitalize"): {"Healthy": 0.93, "MildInfection": 0.07},
    }
    transition_rewards = {"Healthy": 9.0, "MildInfection": 0.0, "SevereInfection": -7.0}

    for ho_cost in hospital_costs:
        action_costs = {**base_action_costs, "Hospitalize": ho_cost}
        nS, nA = len(STATES), len(ACTIONS)
        P = np.zeros((nS, nA, nS)); R = np.zeros((nS, nA, nS))
        for (s, a), dist in transitions.items():
            si = S_IDX[s]; ai = A_IDX[a]
            for sp, p in dist.items():
                spi = S_IDX[sp]
                P[si, ai, spi] = p
                R[si, ai, spi] = transition_rewards[sp] + action_costs[a]
        _, _, pi = value_iteration(P, R)
        rows.append({
            "koszt hospitalizacji": ho_cost,
            **{f"pi*({s})": ACTIONS[pi[S_IDX[s]]] for s in STATES},
        })
    return pd.DataFrame(rows)


#plots
def _save(fig: plt.Figure, name: str) -> str:
    fig.savefig(OUTPUT_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_learning_curve(rewards: list[float]) -> str:
    rewards = np.array(rewards)
    window = 100
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rewards, alpha=0.3, color="#3b6fb6", label="suma nagród (epizod)")
    ax.plot(np.arange(window - 1, len(rewards)), smooth, color="#b6553b", lw=2, label=f"średnia ruchoma (okno={window})")
    ax.set_xlabel("epizod")
    ax.set_ylabel("suma nagród")
    ax.set_title("Krzywa uczenia Q-learning")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "learning_curve.png")


def plot_q_heatmap(Q: np.ndarray, title: str, fname: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(Q, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS)
    ax.set_yticks(range(len(STATES)))
    ax.set_yticklabels(STATES)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            ax.text(j, i, f"{Q[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, fname)


def plot_policy_comparison(pi_vi: np.ndarray, Q_q: np.ndarray, Q_dqn: np.ndarray) -> str:
    pi_q = Q_q.argmax(axis=1); pi_dqn = Q_dqn.argmax(axis=1)
    df = pd.DataFrame(
        {
            "Value Iteration": [ACTIONS[a] for a in pi_vi],
            "Q-learning": [ACTIONS[a] for a in pi_q],
            "FQI (NN)": [ACTIONS[a] for a in pi_dqn],
        },
        index=STATES,
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    cell_text = df.values.tolist()
    table = ax.table(cellText=cell_text, rowLabels=df.index, colLabels=df.columns, loc="center", cellLoc="center")
    table.scale(1, 1.5)
    ax.set_title("Polityki optymalne według metod")
    fig.tight_layout()
    return _save(fig, "policy_comp.png")


def plot_sensitivity(sens: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    cell_text = sens.values.tolist()
    table = ax.table(cellText=cell_text, colLabels=sens.columns, loc="center", cellLoc="center")
    table.scale(1, 1.5)
    ax.set_title("Wrażliwość polityki na koszt hospitalizacji")
    fig.tight_layout()
    return _save(fig, "sensitivity.png")


#report glue
def main() -> None:
    P, R = build_mdp()

    V_star, Q_star, pi_star = value_iteration(P, R)
    Q_qlearn, rewards = q_learning(P, R)
    Q_dqn, losses = fitted_q_iteration(P, R)

    fig_curve = plot_learning_curve(rewards)
    fig_q_vi = plot_q_heatmap(Q_star, "Q* (Value Iteration)", "q_vi.png")
    fig_q_q = plot_q_heatmap(Q_qlearn, "Q (Q-learning, koniec uczenia)", "q_q.png")
    fig_q_dqn = plot_q_heatmap(Q_dqn, "Q (Fitted Q Iteration --- MLP)", "q_dqn.png")
    fig_policy = plot_policy_comparison(pi_star, Q_qlearn, Q_dqn)

    cf = counterfactual_table(Q_star)
    sens = sensitivity_analysis(P, R, hospital_costs=[-2.0, -4.0, -8.0, -12.0])
    fig_sens = plot_sensitivity(sens)

    optimum_table = pd.DataFrame(
        {
            "stan": STATES,
            "V_star": np.round(V_star, 3),
            "pi_star": [ACTIONS[a] for a in pi_star],
        }
    )

    qstar_df = pd.DataFrame(np.round(Q_star, 3), index=STATES, columns=ACTIONS)
    qstar_df.insert(0, "stan", qstar_df.index); qstar_df.reset_index(drop=True, inplace=True)

    qq_df = pd.DataFrame(np.round(Q_qlearn, 3), index=STATES, columns=ACTIONS)
    qq_df.insert(0, "stan", qq_df.index); qq_df.reset_index(drop=True, inplace=True)

    ctx = ReportContext(
        lab_number=9,
        report_title="RL w optymalizacji procedur medycznych: leczenie infekcji",
        variant="8 — leczenie infekcji (Healthy / Mild / Severe; Observe / Antibiotic / Hospitalize)",
    )
    ctx.section(
        "CEL",
        "Sformalizowanie problemu wyboru terapii infekcji jako MDP, znalezienie polityki "
        "optymalnej (Value Iteration), porównanie z Q-learningiem oraz aproksymacją sieci "
        "neuronowych (Fitted Q Iteration). Analiza interpretowalności decyzji (Explainable RL).",
    )
    ctx.section(
        "PROBLEM",
        "Pacjent znajduje się w jednym z trzech stanów: \\texttt{Healthy}, "
        "\\texttt{MildInfection}, \\texttt{SevereInfection}. Lekarz wybiera akcję: "
        "\\texttt{Observe}, \\texttt{Antibiotic}, \\texttt{Hospitalize}. Cel: maksymalizować "
        "zdyskontowaną sumę nagród $G_t = \\sum_k \\gamma^k r_{t+k+1}$ z $\\gamma=0.95$.",
    )
    ctx.section(
        "DANE",
        "MDP $\\langle S,A,P,R,\\gamma\\rangle$ z PDF (wariant 8). "
        "Tablice $P(s'|s,a)$ kalibrowane intuicyjnie: leczenie agresywniejsze "
        "$\\Rightarrow$ większa szansa wyzdrowienia, ale koszt; brak leczenia w stanach "
        "infekcji $\\Rightarrow$ wzrost ryzyka stanu ciężkiego.",
    )
    ctx.section(
        "METODY",
        "\\textbf{1.} Value Iteration: $V_{k+1}(s) = \\max_a \\sum_{s'} P(s'|s,a)[R(s,a,s') + \\gamma V_k(s')]$.\\\\"
        "\\textbf{2.} Q-learning ($\\epsilon$-greedy, $\\alpha=0.2$, 4000 epizodów po 50 kroków).\\\\"
        "\\textbf{3.} Fitted Q Iteration jako odpowiednik DQN: 3 osobne sieci MLP (hidden=(16,16)) "
        "uczone iteracyjnie na próbkach $(s, a, r, s')$.\\\\"
        "\\textbf{4.} Explainable RL: \\textit{(a)} tablice $Q^*$, \\textit{(b)} kontrfaktyczne "
        "$\\Delta Q = Q(s, a^*) - Q(s, a')$, \\textit{(c)} analiza wrażliwości polityki na "
        "koszt hospitalizacji.",
    )
    ctx.section(
        "IMPLEMENTACJA",
        "Plik \\texttt{lab9/lab9\\_solution.py}. Funkcje: \\texttt{build\\_mdp}, "
        "\\texttt{value\\_iteration}, \\texttt{q\\_learning}, \\texttt{fitted\\_q\\_iteration}. "
        "Wszystkie elementy w czystym numpy + sklearn (bez TF/PyTorch).",
    )
    ctx.section(
        "OBLICZENIA",
        "Polityka optymalna i wartości stanów ($\\pi^*$, $V^*$):\n\n"
        + df_to_latex(optimum_table, "Polityka optymalna z Value Iteration.", "vi", float_format="%.3f")
        + "\nMacierz $Q^*(s,a)$ z Value Iteration:\n\n"
        + df_to_latex(qstar_df, "$Q^*(s,a)$ --- Value Iteration.", "qstar", float_format="%.3f")
        + "\nMacierz $Q$ z tablicowego Q-learningu:\n\n"
        + df_to_latex(qq_df, "$Q(s,a)$ --- tabular Q-learning.", "qq", float_format="%.3f"),
    )
    ctx.section(
        "WYNIKI",
        "Wyjaśnienia kontrfaktyczne (najlepsza vs druga akcja):\n\n"
        + df_to_latex(cf, "Wyjaśnienia kontrfaktyczne $\\Delta Q$.", "cf", float_format="%.3f")
        + "\nWrażliwość polityki na koszt hospitalizacji:\n\n"
        + df_to_latex(sens, "Wrażliwość $\\pi^*$ na koszt hospitalizacji.", "sens"),
    )
    ctx.section(
        "WYKRESY",
        figure_block(fig_curve, "Krzywa uczenia Q-learning.", "lc")
        + figure_block(fig_q_vi, "Macierz $Q^*$ z Value Iteration.", "qvi")
        + figure_block(fig_q_q, "Macierz $Q$ z Q-learningu.", "qq")
        + figure_block(fig_q_dqn, "Macierz $Q$ z Fitted Q Iteration (MLP).", "qdqn")
        + figure_block(fig_policy, "Polityki według metod.", "polcmp")
        + figure_block(fig_sens, "Wrażliwość polityki na koszt hospitalizacji.", "senstab"),
    )
    ctx.section(
        "INTERPRETACJA",
        "Polityka optymalna ($\\pi^*$): w stanie \\texttt{SevereInfection} agresywne leczenie "
        "(\\texttt{Hospitalize}) zwiększa wartość mimo wysokiego kosztu, ponieważ przewaga "
        "w prawdopodobieństwie powrotu do \\texttt{Healthy} jest dominująca. W "
        "\\texttt{MildInfection} optimum to \\texttt{Antibiotic} lub \\texttt{Hospitalize} "
        "(zależnie od kosztu). $\\Delta Q$ wskazuje, że dla \\texttt{Healthy} decyzja jest "
        "trywialna (każda akcja podobna), co implikuje niską niepewność. Wzrost kosztu "
        "hospitalizacji zmienia $\\pi^*$ w stanach infekcji ku tańszym terapiom --- "
        "to ważny insight dla decyzji ekonomiczno-klinicznych. Q-learning i FQI zbliżają się "
        "do $\\pi^*$, choć dla małej liczby próbek FQI może odbiegać.",
    )
    ctx.section(
        "WNIOSKI",
        "RL pozwala formalnie modelować sekwencyjne decyzje terapeutyczne. Dla małych MDP "
        "Value Iteration daje ground truth; Q-learning konwerguje empirycznie; aproksymatory "
        "neuronowe (DQN/FQI) skalują się na duże przestrzenie stanów. Explainable RL --- "
        "$Q$, $\\Delta Q$, wrażliwość na nagrodę --- pozwala uzasadnić decyzję klinicznie. "
        "Modele RL w medycynie wymagają walidacji offline i zachowania trybu human-in-the-loop.",
    )

    pdf = render_report(OUTPUT_DIR, "lab9_report.tex", ctx)
    print(f"OK -> {pdf}")


if __name__ == "__main__":
    main()
