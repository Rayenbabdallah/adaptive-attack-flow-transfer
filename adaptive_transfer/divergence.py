"""Predictive divergence between static and adaptive campaign success (article Secs. 4.2 and 5.4).

For each campaign j, p_stat_j and p_adap_j are the fractions of the organization
population in which campaign j succeeds under the static and adaptive (corpus-pool)
models. The reported summary is the mean per-campaign
    D = (1/N) sum_j KL( Bernoulli(p_adap_j) || Bernoulli(p_stat_j) )
in bits, with probability clipping p <- min(max(p, eps), 1 - eps), eps = 1e-3, plus
an eps sweep and the raw mean rates as the smoothing-independent summary.
"""
from __future__ import annotations
import argparse, json, sys, importlib.util
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realism"))


def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


E = _load("experiment", ROOT / "defense_surface" / "experiment.py")
X = _load("xfer_experiment", ROOT / "adaptive_transfer" / "transfer_engine.py")
R = X.R
SEED = X.SEED


def kl_bern_bits(p1, p0, eps):
    p1 = np.clip(p1, eps, 1 - eps); p0 = np.clip(p0, eps, 1 - eps)
    return p1 * np.log2(p1 / p0) + (1 - p1) * np.log2((1 - p1) / (1 - p0))


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

    S = np.zeros((len(comps), len(camps))); A = np.zeros_like(S)
    for i, m in enumerate(comps):
        cov = X.covered(m["D"], controls)
        for j, c in enumerate(camps):
            S[i, j] = X.succeeds(c, cov, pool, "static")
            A[i, j] = X.succeeds(c, cov, pool, "adaptive")
    p_stat, p_adap = S.mean(0), A.mean(0)

    def summary(eps):
        d = kl_bern_bits(p_adap, p_stat, eps)
        return {"mean_bits": float(d.mean()), "median_bits": float(np.median(d)),
                "total_bits": float(d.sum()), "max_bits": float(d.max())}

    reroute = [float(np.mean([len(pool.get(ta, [])) for ta in set(c.tactics)])) for c in camps]
    report = {
        "n_companies": len(comps), "n_campaigns": len(camps),
        "mean_static_success": float(p_stat.mean()),
        "mean_adaptive_success": float(p_adap.mean()),
        "D_eps_1e-3": summary(1e-3),
        "eps_sweep": {f"{e:g}": summary(e) for e in (1e-2, 1e-3, 1e-4)},
        "mean_substitute_pool_size": float(np.mean(reroute)),
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "divergence.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--attack", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args(); run(a.corpus, a.attack, a.out)


if __name__ == "__main__":
    main()
