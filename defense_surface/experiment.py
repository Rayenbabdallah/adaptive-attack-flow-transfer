"""Shared ATT&CK / KEV loaders (load_attack, kev_rates) used by adaptive_transfer/.

The module also contains a separate dual-nature-control experiment that is not part
of the article; only the loaders are used here.

Non-monotonic control value under dual-nature controls.

A defensive control PRUNES attack paths (ATT&CK mitigation->technique coverage)
AND adds its own exploitable INITIAL-ACCESS route (its control class's CISA KEV
self-exploit rate). We test whether the risk-minimizing control portfolio is
non-monotonic (adding a control can raise breach probability) and whether a
surface-aware optimizer beats the standard coverage-greedy on held-out configs.

Breach model: a config is a kill chain of required tactic steps,
each with candidate techniques (sampled from real ATT&CK techniques of that
tactic) with base success probs. Deploying control C multiplies the success of
each technique in cover(C) by (1-e_C). Breach probability:
    normal   = PROD_over_steps  max_tech step_success
    shortcut = for each deployed PRODUCT control C: sigma*s_class(C) * downstream
               (exploit the control directly -> foothold, then proceed; later
                steps still pruned by the deployed portfolio)
    breach   = max(normal, best shortcut)
sigma=0 (surface off) => breach=normal => monotone non-increasing in controls.
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.stats as st

SEED = 20261021
# canonical ransomware kill chain (kill-chain order): IA, exec, priv-esc, cred, lateral, impact
CHAIN = ["TA0001", "TA0002", "TA0004", "TA0006", "TA0008", "TA0040"]

# ---- control class map: ATT&CK mitigation -> KEV product class (None = policy/config, no surface) ----
CLASS_MAP = {
    "M1032": "identity_mfa_sso", "M1015": "identity_mfa_sso", "M1026": "identity_mfa_sso",
    "M1030": "firewall_edge_gw", "M1037": "firewall_edge_gw", "M1031": "firewall_edge_gw",
    "M1020": "firewall_edge_gw",
    "M1035": "vpn_remote_access",
    "M1049": "endpoint_edr_av", "M1040": "endpoint_edr_av", "M1050": "endpoint_edr_av",
    "M1021": "email_web_security", "M1057": "email_web_security",
    "M1053": "backup_recovery", "M1029": "backup_recovery",
    "M1016": "security_mgmt_siem", "M1019": "security_mgmt_siem",
}  # every other mitigation -> None (policy/config control; surface 0)

# curated vendor/product KEV patterns per class (tight, to limit false positives)
KEV_PATTERNS = {
    "firewall_edge_gw": [r"fortigate", r"fortios", r"\bforti", r"palo alto", r"pan-os", r"sonicwall",
                         r"cisco asa", r"firepower", r"check ?point", r"\bngfw\b", r"firewall",
                         r"watchguard", r"zyxel", r"draytek", r"pfsense", r"juniper", r"sophos"],
    "vpn_remote_access": [r"globalprotect", r"anyconnect", r"pulse secure", r"ivanti connect",
                          r"connect secure", r"netscaler", r"citrix gateway", r"sslvpn", r"forticlient",
                          r"openvpn", r"\bvpn\b", r"secure access", r"remote access"],
    "endpoint_edr_av": [r"crowdstrike", r"sentinelone", r"carbon black", r"symantec endpoint", r"mcafee",
                        r"trend micro", r"\beset\b", r"bitdefender", r"kaspersky", r"windows defender",
                        r"microsoft defender", r"\bedr\b", r"antivirus", r"endpoint protection"],
    "identity_mfa_sso": [r"\bokta\b", r"\bduo\b", r"ping identity", r"forgerock", r"keycloak", r"\badfs\b",
                         r"active directory federation", r"azure ad", r"\bentra\b", r"single sign",
                         r"\bsaml\b", r"multi-?factor", r"\bmfa\b", r"cyberark", r"beyondtrust",
                         r"privileged access", r"delinea", r"thycotic"],
    "email_web_security": [r"proofpoint", r"mimecast", r"barracuda", r"ironport", r"websense", r"forcepoint",
                           r"zscaler", r"bluecoat", r"web gateway", r"secure web", r"\bdlp\b",
                           r"data loss prevention", r"secure email"],
    "backup_recovery": [r"veeam", r"commvault", r"veritas", r"networker", r"rubrik", r"cohesity",
                        r"backup exec", r"acronis", r"\bbackup\b"],
    "security_mgmt_siem": [r"\bsplunk\b", r"qualys", r"nessus", r"tenable", r"rapid7", r"solarwinds",
                           r"manageengine", r"\bsiem\b", r"forescout", r"\bnac\b"],
}


def load_attack(path):
    objs = json.load(open(path, encoding="utf-8"))["objects"]

    def ext(o):
        for r in o.get("external_references", []):
            if r.get("source_name") == "mitre-attack" and "external_id" in r:
                return r["external_id"]
        return None
    tech_tactics = defaultdict(set)  # parent technique -> set of tactic shortnames
    tech_id = {}                     # stix id -> parent technique id
    SHORT2TA = {"initial-access": "TA0001", "execution": "TA0002", "persistence": "TA0003",
                "privilege-escalation": "TA0004", "defense-evasion": "TA0005", "credential-access": "TA0006",
                "discovery": "TA0007", "lateral-movement": "TA0008", "collection": "TA0009",
                "command-and-control": "TA0011", "exfiltration": "TA0010", "impact": "TA0040",
                "reconnaissance": "TA0043", "resource-development": "TA0042"}
    for o in objs:
        if o.get("type") != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        e = ext(o)
        if not e or "." in e:
            continue
        tech_id[o["id"]] = e
        for ph in o.get("kill_chain_phases", []):
            ta = SHORT2TA.get(ph.get("phase_name"))
            if ta:
                tech_tactics[e].add(ta)
    mit_name, cover = {}, defaultdict(set)
    mit_ref = {}
    for o in objs:
        if o["type"] == "course-of-action" and not o.get("revoked") and not o.get("x_mitre_deprecated"):
            mit_ref[o["id"]] = ext(o); mit_name[o["id"]] = o["name"]
    for r in objs:
        if r.get("type") == "relationship" and r.get("relationship_type") == "mitigates":
            if r["source_ref"] in mit_ref and r["target_ref"] in tech_id:
                cover[mit_ref[r["source_ref"]]].add(tech_id[r["target_ref"]])
    # tactic -> techniques
    ta_techs = defaultdict(set)
    for t, tas in tech_tactics.items():
        for ta in tas:
            ta_techs[ta].add(t)
    controls = {}
    for sid, mid in mit_ref.items():
        if mid and cover[mid]:
            controls[mid] = {"name": mit_name[sid], "cover": set(cover[mid]),
                             "class": CLASS_MAP.get(mid)}
    return ta_techs, controls


def kev_rates(kev_path, ransomware=False):
    """Return RELATIVE per-class exploitability s_class in [0,1], normalized so the
    most-exploited control class = 1.0 (a technique-success scale). sigma (swept)
    sets the absolute anchor = per-attempt IA-exploit success of the worst class.
    Also return raw counts + catalog share for transparency."""
    v = json.load(open(kev_path, encoding="utf-8"))["vulnerabilities"]
    total = 0
    counts = defaultdict(int)
    for e in v:
        if ransomware and e.get("knownRansomwareCampaignUse") != "Known":
            continue
        total += 1
        t = (e.get("vendorProject", "") + " " + e.get("product", "") + " " +
             e.get("vulnerabilityName", "") + " " + e.get("shortDescription", "")).lower()
        for cls, pats in KEV_PATTERNS.items():
            if any(re.search(p, t) for p in pats):
                counts[cls] += 1
    mx = max(counts.values()) if counts else 1
    s_rel = {c: counts[c] / mx for c in KEV_PATTERNS}        # relative, max class = 1.0
    share = {c: counts[c] / total for c in KEV_PATTERNS}     # raw catalog share
    return s_rel, {"counts": dict(counts), "catalog_share": share}, total


# ---------------- breach model ----------------
def step_success(cands, portfolio, controls, e_C):
    """cands: list of (tech, base). portfolio: list of mids. return max pruned success."""
    best = 0.0
    for tech, base in cands:
        p = base
        for mid in portfolio:
            if tech in controls[mid]["cover"]:
                p *= (1 - e_C)
        best = max(best, p)
    return best


def breach_prob(config, portfolio, controls, s_class, sigma, e_C):
    steps = [step_success(config[ta], portfolio, controls, e_C) for ta in CHAIN]
    normal = float(np.prod(steps))
    downstream = float(np.prod(steps[1:]))  # after initial access
    shortcut = 0.0
    if sigma > 0:
        for mid in portfolio:
            cls = controls[mid]["class"]
            if cls:  # product control -> exploitable surface route to foothold
                shortcut = max(shortcut, sigma * s_class.get(cls, 0.0) * downstream)
    return max(normal, shortcut)


def gen_config(rng, ta_techs, n_cand=4, lo=0.3, hi=0.9):
    cfg = {}
    for ta in CHAIN:
        pool = sorted(ta_techs.get(ta, []))
        if not pool:
            cfg[ta] = [("_none_", float(rng.uniform(lo, hi)))]
            continue
        k = min(n_cand, len(pool))
        chosen = rng.choice(pool, size=k, replace=False)
        cfg[ta] = [(t, float(rng.uniform(lo, hi))) for t in chosen]
    return cfg


# ---------------- hypotheses ----------------
def marginal(config, mid, base_pf, controls, s_class, sigma, e_C):
    before = breach_prob(config, base_pf, controls, s_class, sigma, e_C)
    after = breach_prob(config, base_pf + [mid], controls, s_class, sigma, e_C)
    return after - before  # >0 means adding the control RAISES breach (harmful)


def greedy_optimize(configs, controls, s_class, sigma, e_C, k, surface_aware):
    """pick k controls minimizing mean breach over configs. surface_aware uses sigma; else sigma=0."""
    sig = sigma if surface_aware else 0.0
    chosen = []
    pool = list(controls)
    for _ in range(k):
        best_mid, best_val = None, None
        for mid in pool:
            if mid in chosen:
                continue
            val = np.mean([breach_prob(c, chosen + [mid], controls, s_class, sig, e_C) for c in configs])
            if best_val is None or val < best_val:
                best_val, best_mid = val, mid
        chosen.append(best_mid)
    return chosen


def true_breach(configs, portfolio, controls, s_class, sigma, e_C):
    return float(np.mean([breach_prob(c, portfolio, controls, s_class, sigma, e_C) for c in configs]))


def run(ta_techs, controls, s_class, s_counts, kev_total, out, n_config=300, e_C=0.7, sigma=1.0):
    rng = np.random.default_rng(SEED)
    configs = [gen_config(rng, ta_techs) for _ in range(n_config)]
    products = [m for m in controls if controls[m]["class"]]
    report = {"params": {"n_config": n_config, "e_C": e_C, "sigma": sigma, "n_controls": len(controls),
                         "n_product_controls": len(products), "kev_total": kev_total},
              "s_class": s_class, "s_counts": s_counts}

    # ---- H1: non-monotonicity exists (dual) vs standard (should never) ----
    base_pf = []  # marginal of each control added to empty portfolio
    dual_harm, std_harm = [], []
    by_class = defaultdict(lambda: [0, 0])
    for mid in controls:
        d = [marginal(c, mid, base_pf, controls, s_class, sigma, e_C) for c in configs]
        s = [marginal(c, mid, base_pf, controls, s_class, 0.0, e_C) for c in configs]
        fh = float(np.mean([x > 1e-12 for x in d]))
        dual_harm.append(fh)
        std_harm.append(float(np.mean([x > 1e-12 for x in s])))
        cls = controls[mid]["class"] or "policy"
        by_class[cls][0] += sum(x > 1e-12 for x in d)
        by_class[cls][1] += len(d)
    report["H1_frac_control_config_harmful_dual"] = float(np.mean([h for h in dual_harm]))
    report["H1_frac_controls_ever_harmful_dual"] = float(np.mean([h > 0 for h in dual_harm]))
    report["H1_frac_harmful_standard"] = float(np.mean(std_harm))  # sanity: ~0
    report["H1_harm_by_class"] = {c: {"harmful": n[0], "pairs": n[1], "rate": n[0] / n[1]}
                                  for c, n in sorted(by_class.items())}

    # ---- H2: ranking flip (benefit = -mean marginal) dual vs standard(coverage) ----
    dual_ben = {m: -np.mean([marginal(c, m, base_pf, controls, s_class, sigma, e_C) for c in configs]) for m in controls}
    std_ben = {m: -np.mean([marginal(c, m, base_pf, controls, s_class, 0.0, e_C) for c in configs]) for m in controls}
    mids = list(controls)
    rho = float(st.spearmanr([dual_ben[m] for m in mids], [std_ben[m] for m in mids]).statistic)
    std_rank = sorted(mids, key=lambda m: -std_ben[m])
    dual_rank = sorted(mids, key=lambda m: -dual_ben[m])
    report["H2_spearman_dual_vs_standard_rank"] = rho
    report["H2_standard_top8"] = [(m, controls[m]["name"]) for m in std_rank[:8]]
    report["H2_dual_top8"] = [(m, controls[m]["name"]) for m in dual_rank[:8]]
    report["H2_demoted_by_surface"] = [(m, controls[m]["name"], controls[m]["class"])
                                       for m in std_rank[:8] if m not in dual_rank[:8]]

    # ---- H3: surface-aware optimizer vs standard, held-out configs ----
    idx = rng.permutation(n_config); cut = int(0.6 * n_config)
    train = [configs[i] for i in idx[:cut]]; test = [configs[i] for i in idx[cut:]]
    h3 = {}
    for k in (1, 3, 5):
        pf_std = greedy_optimize(train, controls, s_class, sigma, e_C, k, surface_aware=False)
        pf_sa = greedy_optimize(train, controls, s_class, sigma, e_C, k, surface_aware=True)
        b_std = true_breach(test, pf_std, controls, s_class, sigma, e_C)
        b_sa = true_breach(test, pf_sa, controls, s_class, sigma, e_C)
        diffs = []
        for _ in range(2000):
            samp = [test[i] for i in rng.integers(0, len(test), len(test))]
            diffs.append(true_breach(samp, pf_std, controls, s_class, sigma, e_C)
                         - true_breach(samp, pf_sa, controls, s_class, sigma, e_C))
        h3[f"k={k}"] = {"std_breach": b_std, "surface_aware_breach": b_sa,
                        "improvement": b_std - b_sa,
                        "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                        "std_portfolio": [controls[m]["name"] for m in pf_std],
                        "sa_portfolio": [controls[m]["name"] for m in pf_sa]}
    report["H3_heldout_optimizer"] = h3

    # ---- H4a: permutation null on s_class (is REAL assignment more harmful than random?) ----
    classes = [c for c in s_class]
    real_harm = report["H1_frac_control_config_harmful_dual"]
    perm_harms = []
    for _ in range(1000):
        perm_vals = rng.permutation([s_class[c] for c in classes])
        sperm = {c: perm_vals[i] for i, c in enumerate(classes)}
        hs = []
        for mid in controls:
            d = [marginal(c, mid, base_pf, controls, sperm, sigma, e_C) for c in configs[:60]]
            hs.append(np.mean([x > 1e-12 for x in d]))
        perm_harms.append(np.mean(hs))
    report["H4a_real_harm"] = real_harm
    report["H4a_perm_mean"] = float(np.mean(perm_harms))
    report["H4a_real_percentile_vs_perm"] = float(np.mean([real_harm >= p for p in perm_harms]))

    # ---- H4b: materiality threshold sweep on sigma (anchor exploitability of worst class) ----
    sweep = {}
    for sg in (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5):
        hs = []
        for mid in controls:
            d = [marginal(c, mid, base_pf, controls, s_class, sg, e_C) for c in configs[:100]]
            hs.append(np.mean([x > 1e-12 for x in d]))
        sweep[f"sigma={sg}"] = float(np.mean(hs))
    report["H4b_sigma_sweep_frac_harmful"] = sweep

    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "results.json").write_text(json.dumps(report, indent=1, default=float))
    printable = {k: v for k, v in report.items() if k not in ("H1_harm_by_class",)}
    print(json.dumps(printable, indent=1, default=float))
    print("\nH1_harm_by_class:", json.dumps(report["H1_harm_by_class"], indent=1, default=float))


def selftest(ta_techs, controls, s_class):
    """synthetic validation -- must pass before trusting results."""
    rng = np.random.default_rng(1)
    cfg = gen_config(rng, ta_techs)
    # (1) sigma=0 => adding any control is monotone non-increasing (never harmful)
    for mid in list(controls)[:20]:
        assert marginal(cfg, mid, [], controls, s_class, 0.0, 0.7) <= 1e-12, "monotone broken at sigma=0"
    # (2) a high-surface product control can be harmful at sigma large when it covers nothing downstream-helpful
    prod = [m for m in controls if controls[m]["class"] and s_class.get(controls[m]["class"], 0) > 0]
    assert prod, "no product controls mapped"
    harmful_exists = any(marginal(cfg, m, [], controls, s_class, 1.5, 0.7) > 1e-12 for m in prod)
    assert harmful_exists, "no harmful control even at sigma=1.5 (mechanism dead)"
    # (3) breach prob is a valid probability in [0,1]
    b = breach_prob(cfg, list(controls)[:5], controls, s_class, 1.0, 0.7)
    assert 0 <= b <= 1, "breach not a probability"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True)
    ap.add_argument("--kev", required=True)
    ap.add_argument("--out", default="research/defense_surface/results")
    ap.add_argument("--ransomware", action="store_true", help="use ransomware-weighted KEV rates")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    ta_techs, controls = load_attack(a.attack)
    s_class, s_counts, kev_total = kev_rates(a.kev, ransomware=a.ransomware)
    if a.selftest:
        selftest(ta_techs, controls, s_class); return
    print("s_class rates:", json.dumps(s_class, indent=1))
    print("s_counts:", s_counts, "kev_total:", kev_total)
    run(ta_techs, controls, s_class, s_counts, kev_total, a.out)


if __name__ == "__main__":
    main()
