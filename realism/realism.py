"""Attack Flow corpus parser (load_corpus, TACTIC_ORDER) used by adaptive_transfer/.

The realism metric below is not part of the article; only the parser is used here.

Realism metric for attacker technique sequences.

Parses MITRE Attack Flow `.afb` corpus into ordered ATT&CK technique/tactic
sequences, fits a smoothed Markov (bigram) model, and scores candidate
sequences by per-step perplexity. Lower = more realistic.

Pure/importable. No real-corpus scores are
computed on import.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

TACTIC_ORDER = [  # ATT&CK Enterprise kill-chain order
    "TA0043", "TA0042", "TA0001", "TA0002", "TA0003", "TA0004", "TA0005",
    "TA0006", "TA0007", "TA0008", "TA0009", "TA0011", "TA0010", "TA0040",
]
START, END = "<S>", "<E>"


@dataclass
class Incident:
    name: str
    techniques: tuple[str, ...]   # parent technique ids in flow order
    tactics: tuple[str, ...]      # tactic id per technique, same order


def _parent(tid: str) -> str:
    return tid.split(".")[0] if tid else tid


def _props(obj: dict) -> dict:
    out = {}
    for p in obj.get("properties", []):
        if isinstance(p, list) and len(p) == 2:
            out[p[0]] = p[1]
    return out


def parse_afb(path: str) -> Incident | None:
    d = json.load(open(path, encoding="utf-8"))
    objs = d["objects"]
    by = {o["instance"]: o for o in objs}
    layout = d.get("layout", {})
    actions = {o["instance"]: o for o in objs if o["id"] == "action"}
    if not actions:
        return None
    # latch -> anchor -> action
    latch2anchor = {}
    for o in objs:
        if o["id"] in ("horizontal_anchor", "vertical_anchor"):
            for la in o.get("latches", []):
                latch2anchor[la] = o["instance"]
    anchor2action = {}
    for a in actions.values():
        for anc in (a.get("anchors") or {}).values():
            anchor2action[anc] = a["instance"]

    def resolve(node):  # latch/handle/anchor/action -> owning action or None
        if node in actions:
            return node
        if node in latch2anchor:
            return anchor2action.get(latch2anchor[node])
        if node in anchor2action:
            return anchor2action[node]
        return None

    # directed graph over ALL object nodes via dynamic_line source/target
    succ = defaultdict(set)
    nodes = set(by)
    for o in objs:
        if o["id"] == "dynamic_line" and "source" in o and "target" in o:
            s = latch2anchor.get(o["source"], o["source"])
            t = latch2anchor.get(o["target"], o["target"])
            s = anchor2action.get(s, s)
            t = anchor2action.get(t, t)
            if s in nodes and t in nodes:
                succ[s].add(t)
    # Kahn topological sort, deterministic tie-break by layout x then y then id
    indeg = defaultdict(int)
    for s in succ:
        for t in succ[s]:
            indeg[t] += 1

    def key(n):
        xy = layout.get(n, [1e9, 1e9])
        return (xy[0], xy[1], n)

    ready = sorted([n for n in nodes if indeg[n] == 0], key=key)
    order, seen = [], set()
    while ready:
        n = ready.pop(0)
        if n in seen:
            continue
        seen.add(n)
        order.append(n)
        for t in sorted(succ.get(n, ()), key=key):
            indeg[t] -= 1
            if indeg[t] == 0:
                ready.append(t)
        ready.sort(key=key)
    # actions not reached (cycles/islands) appended by layout
    for n in sorted(actions, key=key):
        if n not in seen:
            order.append(n)
    techs, tacs = [], []
    for n in order:
        if n in actions:
            pr = _props(actions[n])
            tid = _parent(pr.get("technique_id") or "")
            if not tid:
                continue
            techs.append(tid)
            tacs.append(pr.get("tactic_id") or "NA")
    name = _props(next(o for o in objs if o["id"] == "flow")).get("name", os.path.basename(path))
    return Incident(name, tuple(techs), tuple(tacs))


def load_corpus(corpus_dir: str, min_len: int = 5) -> list[Incident]:
    out = []
    for p in sorted(glob.glob(os.path.join(corpus_dir, "*.afb"))):
        inc = parse_afb(p)
        if inc and len(inc.techniques) >= min_len:
            out.append(inc)
    return out


# ----------------------------------------------------------- Markov metric

class BigramModel:
    def __init__(self, alpha: float = 0.5, level: str = "tactic"):
        self.alpha, self.level = alpha, level
        self.vocab: set[str] = set()
        self.trans: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.uni: dict[str, int] = defaultdict(int)

    def _seq(self, inc: Incident) -> list[str]:
        return list(inc.tactics if self.level == "tactic" else inc.techniques)

    def fit(self, incidents: list[Incident]) -> "BigramModel":
        for inc in incidents:
            s = [START] + self._seq(inc) + [END]
            for x in s:
                self.vocab.add(x)
            for a, b in zip(s[:-1], s[1:]):
                self.trans[a][b] += 1
                self.uni[b] += 1
        self.vocab |= {START, END}
        return self

    def _logp(self, a: str, b: str) -> float:
        V = len(self.vocab)
        num = self.trans[a].get(b, 0) + self.alpha
        den = sum(self.trans[a].values()) + self.alpha * V
        return np.log(num / den)

    def perplexity(self, seq: list[str]) -> float:
        s = [START] + list(seq) + [END]
        lp = [self._logp(a, b) for a, b in zip(s[:-1], s[1:])]
        return float(np.exp(-np.mean(lp)))


class UnigramModel(BigramModel):
    """Order-insensitive set/unigram baseline."""

    def perplexity(self, seq: list[str]) -> float:
        V = len(self.vocab)
        tot = sum(self.uni.values())
        lp = [np.log((self.uni.get(x, 0) + self.alpha) / (tot + self.alpha * V)) for x in seq]
        return float(np.exp(-np.mean(lp))) if lp else float("inf")


def score_sequence(model: BigramModel, techniques, tactics) -> float:
    seq = list(tactics if model.level == "tactic" else techniques)
    return model.perplexity(seq)


# ----------------------------------------------------------- synthetics

def synthetic(real: Incident, family: str, all_techniques: list[str],
              tech2tactic: dict[str, str], group_sets: list[list[str]], rng) -> Incident:
    L = len(real.techniques)
    if family == "CAP-SHUFFLE":
        idx = rng.permutation(L)
        t = [real.techniques[i] for i in idx]
    elif family == "GROUP-CAP":
        pool = [g for g in group_sets if len(g) >= L]
        g = pool[rng.integers(len(pool))] if pool else all_techniques
        t = list(rng.choice(g, size=L, replace=False))
    elif family == "UNIFORM":
        t = list(rng.choice(all_techniques, size=L, replace=False))
    elif family == "SINGLE-TACTIC":
        tac = rng.choice(TACTIC_ORDER)
        pool = [x for x in all_techniques if tech2tactic.get(x) == tac] or all_techniques
        t = list(rng.choice(pool, size=min(L, len(pool)), replace=False))
        while len(t) < L:
            t.append(rng.choice(pool))
    else:
        raise ValueError(family)
    tac = [tech2tactic.get(x, "NA") for x in t]
    return Incident(f"{real.name}|{family}", tuple(t), tuple(tac))


def coverage_progression(tactics) -> float:
    """Realism-positive feature (higher = more realistic): tactic coverage x forward progression.
    """
    tac = [t for t in tactics if t in TACTIC_ORDER]
    if len(tac) < 2:
        return 0.0
    L = len(tac)
    coverage = len(set(tac)) / min(L, len(TACTIC_ORDER))
    pairs = list(zip(tac[:-1], tac[1:]))
    forward = sum(TACTIC_ORDER.index(b) > TACTIC_ORDER.index(a) for a, b in pairs) / len(pairs)
    return coverage * forward


def composite_scores(perplexities, cps, train_mask):
    """S = z(perplexity) - z(cp), standardised on training-fold REAL incidents (train_mask).
    Lower = more realistic. perplexities/cps are arrays aligned with train_mask over REAL incidents;
    for scoring synthetics, pass their raw ppl/cp and the same train mu/sd."""
    p = np.asarray(perplexities, float)
    c = np.asarray(cps, float)
    mu_p, sd_p = p[train_mask].mean(), p[train_mask].std() or 1.0
    mu_c, sd_c = c[train_mask].mean(), c[train_mask].std() or 1.0
    return (p - mu_p) / sd_p - (c - mu_c) / sd_c, (mu_p, sd_p, mu_c, sd_c)


def composite_apply(ppl, cp, params):
    mu_p, sd_p, mu_c, sd_c = params
    return (ppl - mu_p) / (sd_p or 1.0) - (cp - mu_c) / (sd_c or 1.0)


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC that REAL (pos) has LOWER perplexity than synthetic (neg)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p < neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(wins / (len(pos) * len(neg)))
