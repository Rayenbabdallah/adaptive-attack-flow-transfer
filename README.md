# Adaptive attack-flow transfer

Reference implementation and benchmark for the article *Static replay overstates your
defense: adaptive attack-flow transfer, predictive divergence, and the limits of control
hardening* (under double-anonymous review).

Documented attack campaigns (MITRE Attack Flow corpus) are replayed against a synthetic
population of organizations' deployed controls (ATT&CK mitigations), in two ways:

- **static** (breach-and-attack-simulation style): a blocked step blocks the campaign;
- **adaptive**: when a step is blocked, the attacker substitutes another technique for the
  same ATT&CK tactic, subject to a tactic-precondition state machine.

Everything is built from public data and runs offline on a laptop.

## Quick start

```bash
pip install -r requirements.txt
python adaptive_transfer/run_all.py
```

`run_all.py` unpacks the ATT&CK bundle, runs the self-tests and every experiment, writes
JSON results to `adaptive_transfer/results/` and regenerates the two figures in
`adaptive_transfer/figures/`. Everything uses the fixed seed `20261021` and is deterministic.
Tested with Python 3.11, numpy 2.2, scipy 1.14, matplotlib 3.9.

## Where each reported number comes from

| Article result | Script | Output (`adaptive_transfer/results/`) |
|---|---|---|
| Static 0.051 vs adaptive 0.360, gap 0.310 (Table 1, Sec. 5.2) | `revision_experiments.py` | `revision.json` → `A_substitution_ablation` |
| Campaign-resample 95% CI [0.237, 0.376] | `revision_experiments.py` | `revision.json` → `B_campaign_bootstrap_gap` |
| Organization-resample CI, unconstrained-re-routing null, coverage-count permutation | `transfer_engine.py` | `results.json` → `H1_gap_ci`, `H3_null_rate`, `H4_real_percentile_vs_perm` |
| Platform 0.356 / co-occurrence 0.263 ablation (Table 2, Fig. 1) | `revision_experiments.py` | `revision.json` → `A_substitution_ablation` |
| Predictive divergence D ≈ 0.94 bits, ε sweep (Sec. 5.4) | `divergence.py` | `divergence.json` |
| No controls 0.706, greedy 0.324, exhaustive 0.000 at k = 3 (Fig. 2) | `revision_experiments.py` | `revision.json` → `C_optimizer_k3` |
| Coverage / density sweep, leave-one-campaign-out (Sec. 5.6) | `robustness.py` | `robustness.json` |
| Held-out 5-fold CV and per-fold portfolios (Tables 3–4) | `held_out_v2.py` | `held_out_v2.json` |
| Reachability decomposition and paired CV tests | `decomp.py` | `decomp.json` |
| Chokepoint-coverage selector | `mitigation.py` | `mitigation.json` |
| Figures 1–2 | `make_figures.py` | `adaptive_transfer/figures/*.pdf` |

`surprise.py` is a development-stage analysis (prioritization regret under the static
model); it was called "Bayesian surprise" during development and is kept for completeness.
The quantity reported in the article is computed by `divergence.py`.

## Layout

```
adaptive_transfer/
  transfer_engine.py       static vs adaptive replay, nulls, organization generator
  revision_experiments.py  feasibility ablation, campaign bootstrap, exhaustive k=3 optimizer
  divergence.py            predictive divergence (article Eq. 1)
  mitigation.py            chokepoint-coverage selector
  robustness.py            coverage/density sweeps, leave-one-campaign-out
  held_out_v2.py           leakage-free held-out cross-validation
  decomp.py                reachability decomposition, paired cross-validation
  surprise.py              development-stage regret analysis (see above)
  make_figures.py          regenerates the article figures
  run_all.py               one-command reproduction
  data/                    public inputs (see NOTICE.md)
realism/realism.py         Attack Flow (.afb) parser; only load_corpus / TACTIC_ORDER are used
defense_surface/experiment.py  ATT&CK / KEV loaders; only load_attack / kev_rates are used
```

## Data

All inputs are public and bundled in `adaptive_transfer/data/` (sources, versions and terms
in `NOTICE.md`): MITRE ATT&CK Enterprise v20.0 (gzip-compressed STIX bundle), the CISA
Known Exploited Vulnerabilities catalog (v2026.10.20), and the MITRE Attack Flow example
corpus (41 flows, of which 34 have at least four ordered technique steps and are used).
The organization population is synthetic, generated from these public priors.

## License

Code: MIT (`LICENSE`). Bundled data: the original providers' terms (`NOTICE.md`).
