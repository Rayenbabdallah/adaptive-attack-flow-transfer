"""One-command reproduction: self-tests + every experiment reported in the article.
Usage: python adaptive_transfer/run_all.py   (no arguments; resolves data/ next to this file)
Results are written to adaptive_transfer/results/, figures to adaptive_transfer/figures/.
"""
import gzip, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ATTACK = str(DATA / "enterprise-attack.json")
KEV = str(DATA / "known_exploited_vulnerabilities.json")
CORP = str(DATA / "attackflow_corpus")
PY = sys.executable

STEPS = [
    ("transfer_engine selftest",          ["transfer_engine.py", "--selftest", "--corpus", CORP, "--attack", ATTACK, "--kev", KEV]),
    ("transfer_engine (gap, nulls)",      ["transfer_engine.py", "--corpus", CORP, "--attack", ATTACK, "--kev", KEV]),
    ("revision (ablation, bootstrap, k=3)", ["revision_experiments.py", "--corpus", CORP, "--attack", ATTACK]),
    ("divergence",                        ["divergence.py", "--corpus", CORP, "--attack", ATTACK]),
    ("surprise (prioritization regret)",  ["surprise.py", "--corpus", CORP, "--attack", ATTACK]),
    ("mitigation selftest",               ["mitigation.py", "--selftest", "--corpus", CORP, "--attack", ATTACK]),
    ("mitigation (chokepoint)",           ["mitigation.py", "--corpus", CORP, "--attack", ATTACK]),
    ("robustness (sweeps, LOCO)",         ["robustness.py", "--corpus", CORP, "--attack", ATTACK]),
    ("held-out CV",                       ["held_out_v2.py", "--corpus", CORP, "--attack", ATTACK]),
    ("decomposition + paired CV",         ["decomp.py", "--corpus", CORP, "--attack", ATTACK]),
    ("figures",                           ["make_figures.py"]),
]


def ensure_attack():
    gz = Path(ATTACK + ".gz")
    if not Path(ATTACK).exists() and gz.exists():
        with gzip.open(gz, "rb") as fi, open(ATTACK, "wb") as fo:
            shutil.copyfileobj(fi, fo)


def main():
    ensure_attack()
    missing = [p for p in (ATTACK, KEV) if not Path(p).exists()] + ([] if Path(CORP).is_dir() else [CORP])
    if missing:
        print("MISSING DATA:", missing); sys.exit(1)
    results = []
    for name, args in STEPS:
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run([PY, str(HERE / args[0])] + args[1:])
        results.append((name, r.returncode))
    print("\n==== SUMMARY ====")
    ok = True
    for name, rc in results:
        print(f"  {'OK ' if rc == 0 else 'FAIL'}  {name}")
        ok = ok and rc == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
