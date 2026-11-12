# Thrust-gain controller-cost experiment

These 45 files contain all 3,240 evaluation dives: five training seeds, two
methods and 72 dives per cell. FACT is flown at obstacle-penalty multipliers
0.5, 1 and 2; look-ahead reweighting at 0.5, 1, 2, 4, 8 and 16, to look for
a setting at which it crashes as rarely as FACT. Each seed folder also holds
`preparation.json`: the failure reference used, the flights FACT removes
(all 32 faulty flights and no correct one, on every seed), each flight's
score and the environment.

The fault is `gain40j8`: +40% thrust in 32 of 95 flights. FACT uses the
normalised per-sample TracIn sign-of-flight-mean rule and 12-epoch retraining;
look-ahead reweighting uses the original 2,640-update schedule. Both use the
same first failing reference (task seed 11), generated with the default cost.
Each model is trained once and frozen across all multipliers. The obstacle
softplus coefficient is 5 times the multiplier; all other cost terms and
planner settings remain unchanged. Each file records the protocol.
Evaluation task seed 7 gives 24 test, 24 steep and 24 shallow dives per seed.
Tasks and rollout randomness are paired across methods and settings. At
multiplier 1 the dives are those of Table 2 (`../reach_{fact,l2rw_la}_gain40j8_s*.json`),
outcome for outcome; `supplement_results.py --check` asserts this.

Safe completion means entering a 0.5 m goal radius with no collision anywhere
in the complete trajectory. No outcome cells were dropped or selected.
This is a sensitivity analysis using existing task families and seeds, not an
independent generalisation test.

Environment: Linux x86_64 CPU, Python 3.13.16, PyTorch 2.14.1, NumPy 2.5.3,
SciPy 1.18.1, one PyTorch thread per worker (`NT=1`), five workers on four
cores. No timing comparison is inferred from these runs.

## Reproduce

Prepare the NeuroBEM cache and plant as described in the main repository.
From `code/drone`, with that environment available:

```sh
NT=1 PYTHONPATH=. python work/ras_supplement.py --root /path/to/new-run --seed 1
```

Repeat for seeds 2 through 5. Each seed directory then contains model
checkpoints, the preparation record and the nine evaluation JSON files.
Checkpoints and raw NeuroBEM data are not committed. The unit tests in
`code/drone/work/test_ras_supplement.py` check that only the obstacle penalty
changes and how a dive is scored.

From the repository root, with NumPy, SciPy and Matplotlib installed:

```sh
python paper/ras/supplement_results.py
python paper/ras/supplement_results.py --check
```

The generator derives Table 4, Figure 4 and the paired confidence-interval
macros of Section 7.3.1 from these outcomes, including the comparisons at about
equal crash rate (FACT at 0.5, reweighting at 2) and about equal safe completion
(FACT at 2, reweighting at 8). It also derives the look-alike completion
difference from the existing `reach_{fact,gradfree}_prop50j8m_s{1,2,3}.json`
files. SE and paired 95% Student-t intervals use training seeds as units, not
individual dives. Intervals are exploratory and unadjusted for multiple comparisons.
