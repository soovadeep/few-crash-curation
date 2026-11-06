# Thrust-gain controller-cost experiment

These 30 files contain all 2,160 new evaluation dives: five training seeds,
two methods, three obstacle-penalty multipliers, and 72 dives per cell.
They were run on 2026-11-06 using the implementation in
`code/drone/work/ras_supplement.py`, based on repository commit
`b56555b7fa50b53ec116759e51c716ca9c69f822`.

The fault is `gain40j8`: +40% thrust in 32 of 95 flights. FACT uses the
normalised per-sample TracIn sign-of-flight-mean rule and 12-epoch retraining;
look-ahead reweighting uses the original 2,640-update schedule. Both use the
same first failing reference (task seed 11), generated before any cost change.
Each model is trained once, frozen, and evaluated at multipliers 0.5, 1 and 2.
The obstacle softplus coefficient is 5 times the multiplier; all other cost
terms and planner settings remain unchanged. Each file records the protocol.
Evaluation task seed 7 gives 24 test, 24 steep and 24 shallow dives per seed.
Tasks and rollout randomness are paired across methods and settings.

Safe completion means entering a 0.5 m goal radius with no collision anywhere
in the complete trajectory. No outcome cells were dropped or selected.
The six seed-1 pilot dives are excluded. This is a sensitivity analysis using
existing task families and seeds, not an independent generalisation test.

Environment: macOS arm64 CPU, Python 3.13.14, PyTorch 2.14.1, NumPy 2.5.4,
SciPy 1.18.1, one PyTorch thread per worker. The original published experiments
used a different host; the default controller was rerun here for a within-host
comparison. No timing comparison is inferred from the new runs.

## Reproduce

Prepare the NeuroBEM cache and plant as described in the main repository.
From `code/drone`, with that environment available:

```sh
PYTHONPATH=. python work/ras_supplement.py --root /path/to/new-run --seed 1
```

Repeat for seeds 2 through 5. Each seed directory contains model checkpoints,
preparation metadata and the six evaluation JSON files. Checkpoints and raw
NeuroBEM data are not committed. The three unit tests in
`code/drone/work/test_ras_supplement.py` passed; same-host look-ahead training
also matched every parameter tensor from the original training code.

From the repository root, with NumPy, SciPy and Matplotlib installed:

```sh
python paper/ras/supplement_results.py
python paper/ras/supplement_results.py --check
```

The generator derives Table 4, Figure 4 and paired confidence-interval macros
from these outcomes; it also derives the look-alike completion difference from
the existing `reach_{fact,gradfree}_prop50j8m_s{1,2,3}.json` files. SE and paired
95% Student-t intervals use training seeds as units, not individual dives.
Intervals are exploratory and unadjusted for multiple comparisons.
