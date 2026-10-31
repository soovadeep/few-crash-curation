# FACT: Curating Learned Dynamics for MPPI When Most Logs Are Wrong

Apurba Bose\*, Soovadeep Bakshi\*, Jiachen Li, Dongmei Chen (\* equal contribution)

Code, results and paper source for the manuscript submitted to *Robotics and Autonomous Systems* (`paper/ras/`).
Everything needed to check the paper is here: the flight splits and fault construction (`code/drone/work/attr_data.py`,
`code/crazyflie/cf.py`, `code/mujoco/`), the baseline settings (in each experiment script and Appendix B of the paper),
the per-seed result files (`results/`), the pinned environment (`requirements.txt`) and a script that recomputes every
number in the paper from those files (`paper/ras/check_numbers.py`).

An MPPI planner is only as safe as its learned dynamics model, and the model is only as good as its logs. When a
consistent fault (logs recorded before a propeller change, a miscalibrated thrust map, a fuller battery) covers most of
the logs in some regime, no rule that sees only the logs can tell which group is wrong: robust losses, worst-fit
filtering and ensembles keep the faulty majority, and a few failures added to training are outvoted. A failure observes
the deployed dynamics exactly where the model is wrong, so FACT (Failure-Anchored Curation of Training logs) uses it as
a reference rather than as data.

## Method

Given logs `l_1 ... l_M`, a model `f_theta` trained on all of them, and a failure run `tau` (a crash, or a run a safety
supervisor took over) whose transitions are observations of the deployed dynamics, FACT

1. scores every training sample `i` by `phi_i = -<grad L_tau, grad l_i> / ||grad l_i||`, where `L_tau` is the model's
   squared error along the failure and `l_i` the sample's training loss (TracIn at the final checkpoint, normalised).
   A positive score means a gradient step on the sample would increase the error along the failure;
2. rates each log by the mean score of its samples;
3. removes every log with a positive mean (the *sign rule*, so there is no removal count to choose); and
4. retrains with the unchanged training pipeline.

Scoring is one pass over the data. For a linear layer the per-sample gradient is an outer product, so the norms and
inner products need one backward pass per batch (`code/drone/work/timing_fact_fast.py`). A gradient-free variant,
`phi_i = -<mean failure error, e_i>`, needs only forward passes; on the quadrotor it does nearly as well as TracIn
(Table 2), and the gradient matters on the car (Table 8).

## Main results

Crash rates (% of dives or runs, mean over seeds) with one failure run, from the paper's Tables 1, 3 and 8.

| Benchmark | No fix | Add the failure to training | FACT, sign rule | Learning to reweight, failure as clean set | Oracle (bad logs removed) |
|---|---|---|---|---|---|
| Quadrotor (NeuroBEM), propeller fault | 98.9 | 94.7 | 3.1 | 2.8 | 1.9 |
| Quadrotor, thrust-gain fault | 91.1 | 74.7 | 1.9 | 9.7 | 1.9 |
| Quadrotor, propeller fault with look-alikes | 94.4 | 81.0 | 1.4 | 2.3 | 0.0 |
| MuJoCo car, icy corner | 39.7 | 26.7 | 1.4 (a regime filter here) | – | 0.0 |

Re-flown with every dive recorded (Table 2), FACT reaches the goal without crashing about as often as the oracle;
learning to reweight completes more dives safely, with a smaller margin above the floor, and crashes more on the
thrust-gain fault. What the failure contributes is coverage of the regime where the logs are wrong: a correctly labelled
expert dive or routine flight through that regime works as the reference too, a hover record does not (Table 4).
A gradient-free score that compares each sample's error with the failure's mean error does nearly as well on the
quadrotor (1.4–3.9% crashes against 0.9–3.1%), so the reference, not the gradient, does most of the work there.

On the Crazyflie battery fault built from real NanoBench logs (Table 9), curation alone does not help (50–52% crashes),
and three failure runs help mostly as added training data: 10.0% added alone, 5.6% curated and then added, 5.3% with
the gradient-free score, and 1.7% added to the oracle's flights (12.5% for the oracle alone, 73.6% unfixed).

On one CPU thread a complete fix on the propeller benchmark, from the logs and the crash record to a trained model,
takes 7 s with FACT (12 s with as many updates as a full-data training), 6 s with the gradient-free score, 11 s for an
ordinary training, and 27–31 s with learning to reweight implemented efficiently (443 s with per-sample gradients).

## Layout

| Path | What |
|---|---|
| `paper/ras/` | `main.tex` (the paper, Elsevier `elsarticle` class), `highlights.txt`, `check_numbers.py` (recomputes the paper's numbers from `results/`), `make_submission.py` (assembles the upload files) |
| `paper/` | `refs.bib`, `make_figs.py` and the figures it writes to `figs/` |
| `code/drone/work/` | Quadrotor with a plant learned from NeuroBEM logs: `drone.py` (plant, model, MPPI, evaluation), `attr_data.py` (fault definitions), one script per experiment, and the `pool_*` and `fill_*` scripts that pool results into tables |
| `code/mujoco/` | MuSHR car in MuJoCo with an icy corner: `env.py`, `core.py`, `attrlib.py`, log builders and experiments |
| `code/crazyflie/` | Battery-sag fault in real NanoBench Crazyflie logs: `build_cf.py`, `cf.py`, `cf_table.py` and controls |
| `results/` | The result JSON files behind every number in the paper, per platform |

## Setup

Python 3.13 with PyTorch 2.14 (CPU), NumPy, SciPy, scikit-learn, pandas, Matplotlib and MuJoCo 3.2; `requirements.txt`
pins the versions the results were produced with:

```
pip install -r requirements.txt
```

Raw data and fitted models are not in the repository (`.gitignore` excludes `*.npz` and `*.pt`); the scripts rebuild them.

- **NeuroBEM** (Bauersfeld et al., RSS 2021): download the processed flights from
  https://download.ifi.uzh.ch/rpg/NeuroBEM/ into `code/drone/` (`processed_data/`, `Flights.txt`, `testset.txt`), then
  run `python work/prep.py` from `code/drone/` to build `work/nb100.npz`. The plant is fit on first use and cached.
- **NanoBench** (BSD-3): clone https://github.com/syediu/nanobench and run
  `python build_cf.py <nanobench>/datasets/dataset` from `code/crazyflie/` to build `cf100.npz`.
- **MuJoCo car**: copy the `cars/` folder of the MuSHR MuJoCo model (prl-mushr/mushr_mujoco_ros, BSD-3) to
  `code/mujoco/mushr/cars/` (`env.py` reads `mushr/cars/base_car/buddy.xml` and `mushr/cars/meshes/`). Then, in
  `code/mujoco/`, `python make_dry_logs.py 100` builds the minority-fault logs `logs_V2.5_dry100.npz`, and
  `python mk_matched.py 100 200` adds 100 correct look-alike runs and 200 more dry-day runs, giving the majority-fault
  logs `logs_V2.5_dry300_m100.npz`.

## Reproducing the paper

Run drone scripts from `code/drone/` (`python work/<script>.py ...`), car scripts from `code/mujoco/` and Crazyflie
scripts from `code/crazyflie/`. Each writes its JSON next to the script (`work/` for the drone); copy the files to
`results/drone/`, `results/mujoco/` or `results/crazyflie/` to check them against the paper. Seeds are 1–5 unless
noted; the look-alike benchmark (`prop50j8m`) and some sweep settings use 1–3.

Fault names: `prop50j8` propeller fault, `gain40j8` thrust-gain fault, `prop50j8m` propeller fault with look-alikes,
`prop50mv` minority fault, `lag100j8` inconsistent majority fault (motor log 100 ms late). Every drone fault and seed
first needs its faulty and oracle models: `python work/nat_probe.py <fault> <seed>`.

| Paper element | Commands | Results |
|---|---|---|
| Table 1, Fig. 2 (a–c) | `mis_drone.py --fault F --seed S --k 40`, then `sign3add_drone.py --fault F --seed S`; `base_drone.py --fault F --seed S --k 40`; `cupid_drone.py --fault F --seed S` (propeller, look-alikes); `sup_drone.py --fault F --seed S --fly`. Pooled by `pool_quadrotor.py` | `drone/mis_*`, `base_*`, `cupid_*`, `sup_*` |
| Table 3 | `controls_drone.py --fault F --seed S` (thrust gain and look-alikes: `--only l2rw,add_5050,add_w2000,ft_replay,fact_matched,random_matched,unnorm`) | `drone/controls_*` |
| Table 2 (safe completion) | `reach_drone.py --fault F --model M --seed S` for `M` in `unfixed oracle add fact fact_matched gradfree l2rw l2rw_la`; pooled by `pool_reach.py` | `drone/reach_*` |
| Table 2 (time), Sec. 7.3 and App. B (cost) | On an idle machine: `NT=1 python work/timing_e2e.py` (complete fixes; `--methods M ...` times only those and merges them into an existing `work/timing_e2e.json`), `NT=1 python work/timing_l2rw.py` (per update), `NT=1 python work/timing_fact_fast.py` | `drone/timing_e2e.json`, `timing_l2rw.json` |
| Table 4 (which reference) | `reach_drone.py --fault F --model fact --ref R --seed S` for `R` in `expert routine hover` (the crash-dive rows are Table 2's); `reference_stats.py --fault prop50j8` for the crash dive's coverage and error on the propeller fault | `drone/reach_fact-*`, `reference_failure_prop50j8.json` |
| Measurement pipeline sensitivity (App. B) | `pipeline_drone.py --variant V --seed S --fly` for `V` in `filtered shifted differenced` | `drone/pipeline_*` |
| Supervised failure runs (Sec. 7.4) | `sup_drone.py` above | `drone/sup_*` |
| Other plants and imperfect failure records (Sec. 7.6, Table 5) | `plant_panel.py --plant {wide,subset,gbt,mismatch,heldout} --seed S`; `failrec_drone.py --seed S` and `failrec_drone.py --sweep 35,8,4 --seed S`; then `pool_robust.py` and `fill_robust.py ../../paper/ras/main.tex` | `drone/plant_*`, `failrec_*`, `pool_robust.json` |
| Severity, faulty share, look-alikes (Sec. 7.6, Table 6) | `sweep_drone.py --sev V --nper N --look L --seed S --fly` and `stress_ref.py` with the same arguments; then `fill_sweep.py ../../paper/ras/main.tex` (runs `pool_sweep.py`) and `pool_stress.py` | `drone/sweep_*`, `stressref_*`, `pool_stress.json` |
| Fault that starts mid-flight (Sec. 7.7, Table 7) | `local_drone.py --seed S`, `diag_local.py --seed S`, `iter_local.py --seed S` (also `--unit windows`, and `--nofly`); then `pool_local.py` and `fill_local.py ../../paper/ras/main.tex` | `drone/local*`, `pool_local_*` |
| Removal count, checkpoint and rating (Sec. 7.2) | `ksens_drone.py --fault F --seed S --fly 24,32,48,64`; `ckpt_drone.py --fault F --seed S`; the rating rows are in `mis_*` | `drone/ksens_*`, `ckpt_*` |
| Sign errors per sample, window and flight (Sec. 4.2, after Proposition 5) | `agg_check.py --fault F --seed 1` | `drone/agg_*` |
| How far TracIn is from the theory (Sec. 4.3) | `margin_drone.py --fault F --seed S` and `margin_drone.py --sweep V,N,L --seed S`; then `pool_margin.py` | `drone/margin_*`, `pooled_margin.txt` |
| Fig. 3 | `fig_scores.py prop50j8m 1` | `drone/fig_scores_prop50j8m_s1.json` |
| Minority and inconsistent faults (Sec. 7.10) | `mis_drone.py` and `base_drone.py` with `--fault prop50mv --k 20` and `--fault lag100j8 --k 40`; pooled by `pool_drone.py` (`--base` for the baselines) | `drone/mis_prop50mv_*`, `base_prop50mv_*`, `mis_lag100j8_*`, `base_lag100j8_*` |
| Table 8 (car) | `mj_major.py --logs logs_V2.5_dry300_m100.npz --seed S --k 360`; `mj_crude.py --seed S --k 360`; `mj_out.py --mode major --logs logs_V2.5_dry300_m100.npz --seed S --k 360 --tl 40` | `mujoco/major_*`, `crude_*`, `out_major_*` |
| The car in detail (Sec. 7.8) | `mj_sign_ana.py --seed S`; minority fault: `mj_seeds.py --logs logs_V2.5_dry100.npz --seed S --rs 1` and `mj_out.py --mode seeds --logs logs_V2.5_dry100.npz --seed S --k 100 --tl 10` | `mujoco/signana_*`, `seeds_*`, `out_seeds_*` |
| Table 9, Fig. 2 (d) (Crazyflie) | All with `CF_VOLTS=3.1 CF_EQ=1` (every model takes the same number of gradient steps): `cf_table.py --seed S` (`CF_ONLY` picks rows by regex, `CF_TAG` names the output file; the oracle-plus-crashes row is `CF_ONLY=^oracle,.and.add CF_TAG=O`), `cf_extra.py --seed S`, `cf_outfly.py --seed S`; pooled by `pool_eq.py` and `pool_cmp.py`; add `CF_SPLIT=1` for the held-out plant (`pool_eq.py --split`) | `crazyflie/cf_3.1_eq_*`, `cfsplit_*` |

### Checking the numbers and building the paper

- `python paper/ras/check_numbers.py` recomputes, from `results/`, every number in Tables 1–4, 8 and 9 and Fig. 2 and
  the numbers quoted in the related work, Sections 4.2, 7.1–7.5 and 7.9 and Appendix B, and lists any that differ from
  `paper/ras/main.tex`.
- The `fill_*` scripts above rewrite Tables 5–7 from `results/drone/`; run on the current results they leave the paper
  unchanged.
- `python paper/make_figs.py` redraws Figs. 2 and 3 from `results/`; Fig. 1 is drawn in the paper with TikZ.
- Build with `pdflatex main && bibtex main && pdflatex main && pdflatex main` in `paper/ras/` (TeX Live with the
  `elsarticle` class). Then `python3 make_submission.py` writes `paper/ras/submission/` (and `submission.zip`): the
  source with flat figure paths, `refs.bib` and `main.bbl`, each figure as its own file (`Figure_1.pdf` is the TikZ
  pipeline), the highlights and the PDF, after checking that the folder compiles on its own.

Runs are deterministic on one CPU thread (the default; set `NT` for more threads, which changes floating-point order).
Rerun from a fresh clone, the drone headline row, its reweighting row and the Crazyflie headline row reproduce the
stored removals and crash rates exactly. On another machine the removals reproduce, but rounding can change how a closed-loop dive ends:
the models re-flown for Table 2, some of them on a second machine, differ from Tables 1 and 3 by about one dive per seed. Goal reach can differ by a few of the 24 dives, because a correct planner
hovers at the edge of the 0.5 m goal radius and floating-point differences in the fitted plant (about 1e-7) decide
which side it ends on.

## Citation

If you use this code, please cite the paper; the reference will be added here on publication. Questions go to the
corresponding author, Soovadeep Bakshi (The University of Texas at Austin).
