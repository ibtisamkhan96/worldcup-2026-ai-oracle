# The World Cup 2026 AI Oracle
### 150 years of football, 10,000 simulated tournaments, updated as the real one unfolds

The 2026 FIFA World Cup: 48 teams, 12 groups, 104 matches across the USA, Canada and Mexico, June 11 to July 19. This project predicts the whole thing with an Elo engine trained on **49,405 real international matches since 1872** and a Monte Carlo simulator that plays the actual tournament (real groups, real format, eight best third-placed teams and all) **10,000 times**.

## The headline numbers (pre-tournament, June 11)

| Team | Win the World Cup |
|---|---|
| Spain | **28.4%** |
| Argentina | 20.3% |
| France | 10.4% |
| England | 7.1% |
| Brazil | 5.2% |

Eleven teams above 2%; the trophy is more open than any single favourite.

## Conversation starters the simulation produced

- **Ecuador are the stealth contender**: 8th in the world by 150-year Elo, above Germany and the Netherlands, with a 2.6% title chance from Germany's group.
- **Host check**: Mexico get a meaningful home boost (3.4% title odds), while the USA are only ~66% to escape Group D, the lowest round-of-32 probability of any host.
- **Group of Death**: measured by average Elo, it isn't where the pundits say; the index chart settles the argument.
- **Most likely final**: Spain v Argentina, a fitting collision of the Euro and World champions.

## How it works

1. **Elo engine**: every match since 1872, K-factor scaled by match importance (World Cup 60, friendlies 20), margin-of-victory multiplier, home advantage, host bonus.
2. **Match model**: expected goals scale with the Elo gap; goals are Poisson-sampled so wins, draws and goal difference emerge naturally; knockouts go to extra time and penalties.
3. **Tournament**: full group stage, third-place ranking, knockout tree to the final, 10,000 times.

## Update it live during the tournament

The dataset (martj42/international_results) updates with real results. After any matchday:

```bash
curl -sL https://raw.githubusercontent.com/martj42/international_results/master/results.csv -o data/results.csv
# bump TODAY in analysis.py to the current date, then re-run
```

Fresh odds after every round: a built-in reason to keep posting.

## Honest model notes

Elo knows results, not rosters: injuries, suspensions and tactics are invisible to it. The round-of-32 bracket is approximated by constrained random allocation (FIFA's third-place mapping has 495 valid combinations), which averages correctly across 10,000 runs. Host advantage (+60 Elo) is a judgement call grounded in hosts historically overperforming. Treat the odds as a strong prior, not prophecy; that is also how the betting markets treat theirs.

```
analysis.ipynb        # Executed notebook
analysis.py           # Script version
data/                 # 49,405 real international results (1872-2026)
charts/               # Title odds, survival heatmap, groups and finals
```

**Data:** Mart Jürisoo, International football results 1872–2026 (github.com/martj42/international_results), CC0.

*Author: Ibtisam Ahmed Khan, [linkedin.com/in/ibtisam-ahmed-khan](https://linkedin.com/in/ibtisam-ahmed-khan)*
