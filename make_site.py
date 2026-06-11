"""Regenerate the Oracle dashboard from the latest results.

Matchday routine:
    curl -sL https://raw.githubusercontent.com/martj42/international_results/master/results.csv -o data/results.csv
    python make_site.py            # full 10,000 sims (a few minutes)
    python make_site.py --sims 1000   # quick test run
    git add . && git commit -m "Matchday update" && git push   # Netlify redeploys

Writes site/index.html with fresh odds and today's date.
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
TEAMS = [t for g in GROUPS.values() for t in g]
HOSTS = {"United States", "Mexico", "Canada"}
ROUNDS = ["R32", "R16", "QF", "SF", "Final", "Champion"]


def k_factor(t):
    t = str(t)
    if "FIFA World Cup" in t and "qualification" not in t:
        return 60
    if any(x in t for x in ["Copa América", "UEFA Euro", "African Cup",
                            "AFC Asian Cup", "Gold Cup"]):
        return 50
    if "qualification" in t or "Nations League" in t:
        return 40
    return 20


def build_elo(cutoff):
    df = pd.read_csv("data/results.csv").dropna(subset=["home_score", "away_score"])
    df = df[df["date"] <= cutoff].sort_values("date")
    elo = defaultdict(lambda: 1500.0)
    for r in df.itertuples(index=False):
        adv = 0 if r.neutral else 80.0
        dr = elo[r.home_team] + adv - elo[r.away_team]
        we = 1 / (1 + 10 ** (-dr / 400))
        gd = abs(r.home_score - r.away_score)
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75 + 0.25 * (gd - 3))
        res = 1.0 if r.home_score > r.away_score else (
            0.5 if r.home_score == r.away_score else 0.0)
        d = k_factor(r.tournament) * g * (res - we)
        elo[r.home_team] += d
        elo[r.away_team] -= d
    return {t: elo[t] for t in TEAMS}


def simulate(R, n, seed=2026):
    rng = np.random.default_rng(seed)

    def lams(ra, rb):
        dr = ra - rb
        return 1.40 * np.exp(dr / 480), 1.40 * np.exp(-dr / 480)

    def hadj(t):
        return 60.0 if t in HOSTS else 0.0

    def gm(a, b):
        la, lb = lams(R[a] + hadj(a), R[b] + hadj(b))
        return rng.poisson(la), rng.poisson(lb)

    def ko(a, b):
        ga, gb = gm(a, b)
        if ga != gb:
            return a if ga > gb else b
        la, lb = lams(R[a] + hadj(a), R[b] + hadj(b))
        ea, eb = rng.poisson(la / 3), rng.poisson(lb / 3)
        if ea != eb:
            return a if ea > eb else b
        p = np.clip(0.5 + (R[a] - R[b]) / 4000, 0.35, 0.65)
        return a if rng.random() < p else b

    counts = {t: {rd: 0 for rd in ROUNDS} for t in TEAMS}
    finals = defaultdict(int)
    for _ in range(n):
        winners, runners, thirds = [], [], []
        for g, ms in GROUPS.items():
            pts = {t: 0 for t in ms}
            gd = {t: 0 for t in ms}
            gf = {t: 0 for t in ms}
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = ms[i], ms[j]
                    ga, gb = gm(a, b)
                    gd[a] += ga - gb; gd[b] += gb - ga
                    gf[a] += ga; gf[b] += gb
                    if ga > gb: pts[a] += 3
                    elif gb > ga: pts[b] += 3
                    else: pts[a] += 1; pts[b] += 1
            o = sorted(ms, key=lambda t: (pts[t], gd[t], gf[t], rng.random()),
                       reverse=True)
            winners.append(o[0]); runners.append(o[1])
            thirds.append((o[2], pts[o[2]], gd[o[2]], gf[o[2]]))
        best3 = [t for t, *_ in sorted(thirds, key=lambda x: (x[1], x[2], x[3],
                                       rng.random()), reverse=True)[:8]]
        for t in winners + runners + best3:
            counts[t]["R32"] += 1
        w = winners.copy(); rng.shuffle(w)
        r = runners.copy(); rng.shuffle(r)
        t3 = best3.copy(); rng.shuffle(t3)
        field = list(zip(w[:8], t3)) + list(zip(w[8:], r[:4])) + \
                list(zip(r[4:12:2], r[5:12:2]))
        for rd in ["R16", "QF", "SF", "Final"]:
            nxt = []
            for a, b in field:
                win = ko(a, b)
                counts[win][rd] += 1
                nxt.append(win)
            rng.shuffle(nxt)
            field = list(zip(nxt[0::2], nxt[1::2]))
        champ = ko(field[0][0], field[0][1])
        counts[champ]["Champion"] += 1
        finals[tuple(sorted((field[0][0], field[0][1])))] += 1
    return counts, finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--date", default=str(date.today()))
    args = ap.parse_args()

    print(f"Building Elo from results up to {args.date}...")
    R = build_elo(args.date)
    print(f"Simulating {args.sims:,} tournaments...")
    counts, finals = simulate(R, args.sims)

    payload = {
        "ratings": {t: round(R[t]) for t in TEAMS},
        "groups": GROUPS,
        "probs": {t: {rd: round(counts[t][rd] / args.sims * 100, 1)
                      for rd in ROUNDS} for t in TEAMS},
        "finals": [{"pair": list(p), "pct": round(c / args.sims * 100, 1)}
                   for p, c in sorted(finals.items(), key=lambda kv: -kv[1])[:8]],
    }
    tpl = open("site/template.html").read()
    nice = pd.Timestamp(args.date).strftime("%B %-d, %Y")
    html = tpl.replace("__ORACLE_DATA__",
                       json.dumps(payload, separators=(",", ":"),
                                  ensure_ascii=False)
                       ).replace("__DATE__", nice)
    open("site/index.html", "w").write(html)
    top = sorted(payload["probs"].items(), key=lambda kv: -kv[1]["Champion"])[:5]
    print(f"site/index.html written ({len(html):,} bytes), refreshed {nice}")
    print("Top 5:", ", ".join(f"{t} {p['Champion']}%" for t, p in top))


if __name__ == "__main__":
    main()
