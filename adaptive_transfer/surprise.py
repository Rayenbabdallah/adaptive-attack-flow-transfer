"""Bayesian-surprise mechanism layer on adaptive attack-flow transfer.

(Named 'Bayesian surprise' during development; the article calls the summary a
predictive divergence. See divergence.py for the quantity reported in the article.)

Static-BAS belief = prior; adaptive reality = posterior. Tests whether
adaptation-blind (static-model) control prioritization measurably mis-allocates
(HS1 regret, HS2 divergence), whether Bayesian surprise KL(adaptive||static)
quantifies & localizes it (HS3), and whether a cheap surprise-weighted heuristic
helps (HS4). Reuses the transfer engine.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import importlib.util
import numpy as np
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realism"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# preload defense_surface/experiment.py AS 'experiment' so the transfer module's
# internal `import experiment as E` resolves to it (not to itself).
E = _load("experiment", ROOT / "defense_surface" / "experiment.py")
X = _load("xfer_experiment", ROOT / "adaptive_transfer" / "transfer_engine.py")
R = X.R
SEED = X.SEED


def kl_bern(p1, p0, eps=1e-6):
    p1 = min(max(p1, eps), 1 - eps); p0 = min(max(p0, eps), 1 - eps)
    return p1 * np.log2(p1 / p0) + (1 - p1) * np.log2((1 - p1) / (1 - p0))


def applicable_controls(company, controls):
    out = []
    for mid, c in controls.items():
        if c["class"] is None or c["class"] in company["exposure"]:
            out.append(mid)
    return out


def success_vec(camps, cov, pool, mode):
    return np.array([X.succeeds(c, cov, pool, mode) for c in camps], dtype=float)


def greedy_select(company, camps, controls, pool, k, mode, weights=None):
    """pick k applicable controls minimizing (weighted) believed campaign success under `mode`."""
    appl = applicable_controls(company, controls)
    chosen, w = [], (np.ones(len(camps)) if weights is None else weights)
    for _ in range(k):
        best, bestmid = None, None
        for mid in appl:
            if mid in chosen:
                continue
            cov = X.covered(chosen + [mid], controls)
            score = float((success_vec(camps, cov, pool, mode) * w).sum())
            if best is None or score < best:
                best, bestmid = score, mid
        if bestmid is None:
            break
        chosen.append(bestmid)
    return chosen


def campaign_surprise(camps, pool):
    """structural re-routability s_j in [0,1]: mean over the campaign's tactics of the
    number of substitute options available in the corpus pool (more = static model less
    reliable). Computed from the corpus only (cheap; no per-company adaptive solve)."""
    s = []
    maxpool = max((len(v) for v in pool.values()), default=1)
    for c in camps:
        vals = [len(pool.get(ta, [])) for ta in set(c.tactics)]
        s.append(np.mean(vals) / maxpool if vals else 0.0)
    s = np.array(s)
    return s / (s.max() + 1e-9)


def run(corpus, attack, out, n_comp=160):
    rng = np.random.default_rng(SEED)
    camps = R.load_corpus(corpus, min_len=4)
    ta_techs, controls = E.load_attack(attack)
    pool = defaultdict(set)
    for c in camps:
        for t, ta in zip(c.techniques, c.tactics):
            pool[ta].add(t)
    pool = {k: sorted(v) for k, v in pool.items()}
    priors = rng.uniform(0.15, 0.85, size=(8, len(X.CLASSES)))
    comps = [X.gen_company(rng, priors, controls, s) for s in range(8) for _ in range(n_comp // 8)]
    s_j = campaign_surprise(camps, pool)

    report = {"n_companies": len(comps), "n_campaigns": len(camps)}

    # ---- defender Bayesian surprise over the DEPLOYED configs (HS3 headline) ----
    p_static = np.zeros(len(camps)); p_adapt = np.zeros(len(camps))
    for m in comps:
        cov = X.covered(m["D"], controls)
        p_static += success_vec(camps, cov, pool, "static")
        p_adapt += success_vec(camps, cov, pool, "adaptive")
    p_static /= len(comps); p_adapt /= len(comps)
    surprise_bits = float(sum(kl_bern(p_adapt[j], p_static[j]) for j in range(len(camps))))
    report["HS3_defender_surprise_bits_total"] = surprise_bits
    report["HS3_mean_static_belief"] = float(p_static.mean())
    report["HS3_mean_adaptive_reality"] = float(p_adapt.mean())

    for k in (3, 5):
        reg, jac, static_true, adapt_true, saware_true = [], [], [], [], []
        per_camp_regret = np.zeros(len(camps))
        for m in comps:
            pf_static = greedy_select(m, camps, controls, pool, k, "static")
            pf_adapt = greedy_select(m, camps, controls, pool, k, "adaptive")
            pf_saware = greedy_select(m, camps, controls, pool, k, "static", weights=1 + 3 * s_j)
            a_static = success_vec(camps, X.covered(pf_static, controls), pool, "adaptive")
            a_adapt = success_vec(camps, X.covered(pf_adapt, controls), pool, "adaptive")
            a_saware = success_vec(camps, X.covered(pf_saware, controls), pool, "adaptive")
            static_true.append(a_static.mean()); adapt_true.append(a_adapt.mean()); saware_true.append(a_saware.mean())
            reg.append(a_static.mean() - a_adapt.mean())
            per_camp_regret += (a_static - a_adapt)
            sa, ad = set(pf_static), set(pf_adapt)
            jac.append(len(sa & ad) / len(sa | ad) if (sa or ad) else 1.0)
        per_camp_regret /= len(comps)
        reg = np.array(reg)
        bs = [float(reg[rng.integers(0, len(reg), len(reg))].mean()) for _ in range(2000)]
        # HS3 localization: does structural surprise predict per-campaign regret?
        rho = float(st.spearmanr(s_j, per_camp_regret).statistic)
        report[f"k={k}"] = {
            "HS1_static_rule_adaptive_breach": float(np.mean(static_true)),
            "HS1_adaptive_rule_adaptive_breach": float(np.mean(adapt_true)),
            "HS1_regret_static_minus_adaptive": float(reg.mean()),
            "HS1_regret_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "HS2_mean_jaccard_static_vs_adaptive_portfolio": float(np.mean(jac)),
            "HS3_spearman_surprise_vs_regret": rho,
            "HS4_surprise_aware_adaptive_breach": float(np.mean(saware_true)),
            "HS4_saware_minus_static_rule": float(np.mean(saware_true) - np.mean(static_true)),
            "HS4_closes_regret_frac": float((np.mean(static_true) - np.mean(saware_true)) /
                                            (np.mean(static_true) - np.mean(adapt_true) + 1e-9)),
        }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "surprise.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--attack", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args()
    run(a.corpus, a.attack, a.out)


if __name__ == "__main__":
    main()
