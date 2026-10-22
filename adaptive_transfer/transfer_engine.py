"""Adaptive attack-flow transfer: the transfer engine.

Replay the 34 real Attack Flow campaigns against each company's deployed controls,
STATIC (BAS-style: a step is just blocked) vs ADAPTIVE (attacker substitutes another
real-world technique for the same tactic when blocked), and ask which real campaigns
still reach the objective. Positive claim: adaptive > static (static over-states
defense), heterogeneous across companies, driven by precondition/coverage structure
(not coverage count), with per-company keystone controls.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realism"))
sys.path.insert(0, str(ROOT / "defense_surface"))
import realism as R  # noqa: E402
import experiment as E  # load_attack, kev_rates  # noqa: E402

SEED = 20261021
CLASSES = ["firewall_edge_gw", "vpn_remote_access", "endpoint_edr_av", "identity_mfa_sso",
           "email_web_security", "backup_recovery", "security_mgmt_siem"]
# tactic precondition state machine (inlined; same as control_roi)
STATE_MODEL = {
    "TA0043": ([], ["recon"]), "TA0042": ([], ["resources"]), "TA0001": ([], ["foothold"]),
    "TA0002": (["foothold"], ["code_exec"]), "TA0003": (["code_exec"], ["persistence"]),
    "TA0004": (["code_exec"], ["elevated"]), "TA0005": (["code_exec"], []),
    "TA0006": (["code_exec"], ["credentials"]), "TA0007": (["foothold"], ["discovery"]),
    "TA0008": (["credentials|discovery"], ["foothold"]), "TA0009": (["foothold"], ["collected"]),
    "TA0011": (["foothold"], ["c2"]), "TA0010": (["collected|foothold"], []), "TA0040": (["code_exec"], []),
}


def req_met(reqs, state):
    return all(any(tok in state for tok in g.split("|")) for g in reqs)


def reachable(achievable_tactics, objective):
    present = {t for t in achievable_tactics if t in STATE_MODEL}
    state, changed = set(), True
    while changed:
        changed = False
        for ta in present:
            req, prov = STATE_MODEL[ta]
            if prov and req_met(req, state) and not set(prov).issubset(state):
                state.update(prov); changed = True
    return objective in present and req_met(STATE_MODEL.get(objective, ([], []))[0], state)


def objective_of(camp):
    present = [t for t in camp.tactics if t in R.TACTIC_ORDER]
    return max(present, key=lambda t: R.TACTIC_ORDER.index(t)) if present else None


def covered(D, controls):
    s = set()
    for c in D:
        s |= controls[c]["cover"]
    return s


def gen_company(rng, priors, controls, s):
    exposure = {CLASSES[j] for j in range(len(CLASSES)) if rng.random() < priors[s, j]}
    D = []
    for mid, c in controls.items():
        cls = c["class"]
        if cls is None:
            if rng.random() < 0.35:   # policy controls: deployable by anyone
                D.append(mid)
        elif cls in exposure:         # product control only if the company runs that class
            if rng.random() < 0.6:
                D.append(mid)
    return {"sector": s, "exposure": exposure, "D": D}


def achievable_tactics(camp, cov, pool, mode):
    """mode: 'static' (reported technique must be unblocked) |
            'adaptive' (any corpus technique of that tactic unblocked) |
            'null' (always achievable -- unconstrained re-route)."""
    tac_tech = defaultdict(list)
    for t, ta in zip(camp.techniques, camp.tactics):
        tac_tech[ta].append(t)
    out = set()
    for ta in set(camp.tactics):
        if mode == "null":
            out.add(ta)
        elif mode == "static":
            if any(t not in cov for t in tac_tech[ta]):
                out.add(ta)
        else:  # adaptive
            if any(t not in cov for t in pool.get(ta, [])):
                out.add(ta)
    return out


def succeeds(camp, cov, pool, mode):
    obj = objective_of(camp)
    if obj is None:
        return False
    ach = achievable_tactics(camp, cov, pool, mode)
    if obj not in ach:
        return False
    if mode == "null":
        return True  # unconstrained: structure removed
    return reachable(ach, obj)


def run(corpus, attack, kev, out, n_comp=200):
    rng = np.random.default_rng(SEED)
    camps = R.load_corpus(corpus, min_len=4)
    ta_techs, controls = E.load_attack(attack)
    # substitute pool = real-world techniques per tactic seen across the corpus (bounded, grounded)
    pool = defaultdict(set)
    for c in camps:
        for t, ta in zip(c.techniques, c.tactics):
            pool[ta].add(t)
    pool = {k: sorted(v) for k, v in pool.items()}
    priors = rng.uniform(0.15, 0.85, size=(8, len(CLASSES)))
    comps = [gen_company(rng, priors, controls, s) for s in range(8) for _ in range(n_comp // 8)]

    # per (company,campaign) static/adaptive success
    S = np.zeros((len(comps), len(camps))); A = np.zeros_like(S); Nl = np.zeros_like(S)
    for i, m in enumerate(comps):
        cov = covered(m["D"], controls)
        for j, c in enumerate(camps):
            S[i, j] = succeeds(c, cov, pool, "static")
            A[i, j] = succeeds(c, cov, pool, "adaptive")
            Nl[i, j] = succeeds(c, cov, pool, "null")

    static_rate = S.mean(1); adapt_rate = A.mean(1); null_rate = Nl.mean(1)
    gap = adapt_rate - static_rate
    # H1 bootstrap over companies
    bs = [float(np.mean(gap[rng.integers(0, len(comps), len(comps))])) for _ in range(2000)]
    # H2 heterogeneity: distinct adaptive success-sets; per-campaign success variance across companies
    distinct = len({tuple(A[i].astype(int)) for i in range(len(comps))})
    per_camp_var = float(np.mean(A.var(0)))
    # H3 null: adaptive (precond+tactic-pool) vs unconstrained null
    h3_gap = float(null_rate.mean() - adapt_rate.mean())
    # H4 coverage-count permutation: random controls matched on #covered techniques
    all_mids = list(controls)
    cov_sizes = {m: len(controls[m]["cover"]) for m in all_mids}
    real_gap = float(gap.mean()); perm_gaps = []
    for _ in range(200):
        pg = []
        for m in comps:
            k = len(m["D"])
            fake = list(rng.choice(all_mids, size=min(k, len(all_mids)), replace=False))
            cov = covered(fake, controls)
            sr = np.mean([succeeds(c, cov, pool, "static") for c in camps])
            ar = np.mean([succeeds(c, cov, pool, "adaptive") for c in camps])
            pg.append(ar - sr)
        perm_gaps.append(np.mean(pg))
    h4_real_vs_perm_pct = float(np.mean([real_gap >= p for p in perm_gaps]))

    # H5 keystone control per company (adaptive): removing which deployed control admits most campaigns
    keystones = []
    for m in comps[:60]:
        base = sum(succeeds(c, covered(m["D"], controls), pool, "adaptive") for c in camps)
        best, bestmid = -1, None
        for mid in m["D"]:
            cov = covered([x for x in m["D"] if x != mid], controls)
            extra = sum(succeeds(c, cov, pool, "adaptive") for c in camps) - base
            if extra > best:
                best, bestmid = extra, mid
        if bestmid is not None:
            keystones.append(controls[bestmid]["name"])
    from collections import Counter
    key_dist = Counter(keystones)

    report = {
        "n_companies": len(comps), "n_campaigns": len(camps),
        "H1_static_success_rate": float(static_rate.mean()),
        "H1_adaptive_success_rate": float(adapt_rate.mean()),
        "H1_gap_adaptive_minus_static": real_gap,
        "H1_gap_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "H2_distinct_adaptive_success_sets": distinct,
        "H2_mean_per_campaign_variance_across_companies": per_camp_var,
        "H3_null_rate": float(null_rate.mean()), "H3_null_minus_adaptive": h3_gap,
        "H4_real_gap": real_gap, "H4_perm_mean_gap": float(np.mean(perm_gaps)),
        "H4_real_percentile_vs_perm": h4_real_vs_perm_pct,
        "H5_keystone_distribution_top": key_dist.most_common(8),
        "H5_distinct_keystones": len(key_dist),
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "results.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def selftest(corpus, attack):
    camps = R.load_corpus(corpus, min_len=4)
    ta_techs, controls = E.load_attack(attack)
    pool = defaultdict(set)
    for c in camps:
        for t, ta in zip(c.techniques, c.tactics):
            pool[ta].add(t)
    pool = {k: sorted(v) for k, v in pool.items()}
    c0 = camps[0]
    # no controls -> static == adaptive == success
    assert succeeds(c0, set(), pool, "static") and succeeds(c0, set(), pool, "adaptive")
    # block the reported technique of a tactic that has a substitute -> static fails, adaptive may succeed
    # pick a tactic with >=2 corpus techniques
    multi = [ta for ta in pool if len(pool[ta]) >= 2 and ta in set(c0.tactics)]
    if multi:
        ta = multi[0]
        reported = [t for t, x in zip(c0.techniques, c0.tactics) if x == ta]
        cov = set(reported)  # block only reported technique(s) of that tactic
        st = achievable_tactics(c0, cov, pool, "static")
        ad = achievable_tactics(c0, cov, pool, "adaptive")
        assert ta not in st and ta in ad, "substitution mechanic broken"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--attack", required=True)
    ap.add_argument("--kev", required=True); ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-comp", type=int, default=160, help="organizations (20 per sector x 8)")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.corpus, a.attack); return
    run(a.corpus, a.attack, a.kev, a.out, n_comp=a.n_comp)


if __name__ == "__main__":
    main()
