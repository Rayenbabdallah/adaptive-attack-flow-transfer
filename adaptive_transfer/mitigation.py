"""Mitigation layer: does CHOKEPOINT COVERAGE close the adaptive gap?

Per-control greedy fails because re-routing absorbs single controls (flat objective).
The gap closes only by fully covering a required tactic's substitute pool (a chokepoint
re-routing cannot bypass). We test a set-level chokepoint-aware selector (smooth
pool-coverage surrogate) against static-model and greedy-adaptive selection, and use
surprise/re-routability to separate closeable from irreducible campaigns (HM1-HM4).
"""
from __future__ import annotations
import argparse, json, sys, importlib.util
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realism"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


E = _load("experiment", ROOT / "defense_surface" / "experiment.py")
X = _load("xfer_experiment", ROOT / "adaptive_transfer" / "transfer_engine.py")
R = X.R
SEED = X.SEED


def required_tactics(camp):
    """tactics whose removal (unachievable) makes the objective unreachable under STATE_MODEL."""
    obj = X.objective_of(camp)
    if obj is None:
        return []
    tacs = {ta for ta in camp.tactics if ta in X.STATE_MODEL}
    req = []
    for ta in tacs:
        if not X.reachable(tacs - {ta}, obj):
            req.append(ta)
    return req


def chokepoints(camp, pool):
    """closeable chokepoints = required tactics, ranked by smallest substitute pool."""
    req = required_tactics(camp)
    cand = [(ta, pool.get(ta, [])) for ta in req if pool.get(ta)]
    return sorted(cand, key=lambda x: len(x[1]))


def applicable(company, controls):
    return [m for m, c in controls.items() if c["class"] is None or c["class"] in company["exposure"]]


def choke_select(company, camps, cps, controls, pool, k):
    """greedy maximize smooth surrogate: sum_camp max_chokepoint (fraction of pool covered)."""
    appl = applicable(company, controls); chosen = []

    def surrogate(D):
        cov = X.covered(D, controls); tot = 0.0
        for cp in cps:  # cp = list of (tactic,pool) chokepoints for that campaign
            best = 0.0
            for ta, pl in cp:
                if pl:
                    best = max(best, sum(t in cov for t in pl) / len(pl))
            tot += best
        return tot
    for _ in range(k):
        best, bestmid = None, None
        for mid in appl:
            if mid in chosen:
                continue
            v = surrogate(chosen + [mid])
            if best is None or v > best:
                best, bestmid = v, mid
        if bestmid is None:
            break
        chosen.append(bestmid)
    return chosen


def greedy_select(company, camps, controls, pool, k, mode):
    appl = applicable(company, controls); chosen = []
    for _ in range(k):
        best, bestmid = None, None
        for mid in appl:
            if mid in chosen:
                continue
            cov = X.covered(chosen + [mid], controls)
            sc = sum(X.succeeds(c, cov, pool, mode) for c in camps)
            if best is None or sc < best:
                best, bestmid = sc, mid
        if bestmid is None:
            break
        chosen.append(bestmid)
    return chosen


def run(corpus, attack, out, n_comp=120):
    rng = np.random.default_rng(SEED)
    camps = R.load_corpus(corpus, min_len=4)
    ta_techs, controls = E.load_attack(attack)
    pool = defaultdict(set)
    for c in camps:
        for t, ta in zip(c.techniques, c.tactics):
            pool[ta].add(t)
    pool = {k: sorted(v) for k, v in pool.items()}
    cps = [chokepoints(c, pool) for c in camps]
    # closeability: a campaign is coverable-in-principle if it has >=1 chokepoint whose
    # whole pool is covered by SOME single control or union of controls that exist
    all_cov_union = set().union(*[controls[m]["cover"] for m in controls])
    coverable = []
    for cp in cps:
        ok = any(pl and set(pl).issubset(all_cov_union) for ta, pl in cp)
        coverable.append(ok)
    report = {"n_campaigns": len(camps),
              "frac_with_any_chokepoint": float(np.mean([len(cp) > 0 for cp in cps])),
              "frac_coverable_in_principle": float(np.mean(coverable)),
              "mean_smallest_chokepoint_pool": float(np.mean([len(cp[0][1]) for cp in cps if cp]))}

    priors = rng.uniform(0.15, 0.85, size=(8, len(X.CLASSES)))
    comps = [X.gen_company(rng, priors, controls, s) for s in range(8) for _ in range(n_comp // 8)]
    s_j = None
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    for k in (3, 5):
        b_static, b_greedy, b_choke = [], [], []
        resid_by_camp = np.zeros(len(camps))
        for m in comps:
            pf_s = greedy_select(m, camps, controls, pool, k, "static")
            pf_g = greedy_select(m, camps, controls, pool, k, "adaptive")
            pf_c = choke_select(m, camps, cps, controls, pool, k)
            a_s = np.array([X.succeeds(c, X.covered(pf_s, controls), pool, "adaptive") for c in camps])
            a_g = np.array([X.succeeds(c, X.covered(pf_g, controls), pool, "adaptive") for c in camps])
            a_c = np.array([X.succeeds(c, X.covered(pf_c, controls), pool, "adaptive") for c in camps])
            b_static.append(a_s.mean()); b_greedy.append(a_g.mean()); b_choke.append(a_c.mean())
            resid_by_camp += a_c
        resid_by_camp /= len(comps)
        b_static = np.array(b_static); b_greedy = np.array(b_greedy); b_choke = np.array(b_choke)

        def ci(a, b):
            d = [float((a[idx] - b[idx]).mean()) for idx in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
            return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        report[f"k={k}"] = {
            "breach_static_rule": float(b_static.mean()),
            "breach_greedy_adaptive": float(b_greedy.mean()),
            "breach_chokepoint_aware": float(b_choke.mean()),
            "HM1_choke_vs_static": float(b_static.mean() - b_choke.mean()), "HM1_ci_vs_static": ci(b_static, b_choke),
            "HM1_choke_vs_greedy": float(b_greedy.mean() - b_choke.mean()), "HM1_ci_vs_greedy": ci(b_greedy, b_choke),
            "HM3_residual_success_after_choke": float(b_choke.mean()),
            "HM3_irreducible_campaigns_frac": float(np.mean(resid_by_camp > 0.5)),
        }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "mitigation.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def selftest(corpus, attack):
    camps = R.load_corpus(corpus, min_len=4)
    ta_techs, controls = E.load_attack(attack)
    pool = defaultdict(set)
    for c in camps:
        for t, ta in zip(c.techniques, c.tactics):
            pool[ta].add(t)
    pool = {k: sorted(v) for k, v in pool.items()}
    cps = chokepoints(camps[0], pool)
    assert isinstance(cps, list)
    # covering a chokepoint tactic's whole pool must block the campaign adaptively
    if cps:
        ta, pl = cps[0]
        cov = set(pl)
        assert not X.succeeds(camps[0], cov, pool, "adaptive"), "closing chokepoint pool should block campaign"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--attack", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.corpus, a.attack); return
    run(a.corpus, a.attack, a.out)


if __name__ == "__main__":
    main()
