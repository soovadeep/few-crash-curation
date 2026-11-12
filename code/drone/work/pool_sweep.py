"""Pool the robustness sweep (sweep_*.json) into table cells; prints a dict of LaTeX cells per config."""
import glob, json, sys
import numpy as np
T = ['test dives', 'steep dives', 'shallow dives']
CFG = {'SWEEPBASE': 'sev50_n8_l0', 'SEV20': 'sev20_n8_l0', 'SEV35': 'sev35_n8_l0', 'SEV80': 'sev80_n8_l0',
       'N3': 'sev50_n3_l0', 'N5': 'sev50_n5_l0', 'N11': 'sev50_n11_l0', 'L2': 'sev50_n8_l2', 'L4': 'sev50_n8_l4',
       'L8': 'sev50_n8_l8'}


def rng(v):
    a, b = min(v), max(v)
    return f'{a}' if a == b else f'{a}--{b}'


out = {}
for key, tag in CFG.items():
    ds = [json.load(open(f)) for f in sorted(glob.glob(f'sweep_{tag}_s*.json'))]
    ds = [d for d in ds if d.get('sign_rule_flown')]
    if not ds:
        continue
    det = [d['detection'] for d in ds]
    unf = rng([round(100 * d['attribution_dives_crashed'] / 24) for d in ds])
    cr = np.array([100 * np.mean([d['sign_rule_flown'][t] for t in T]) for d in ds])
    se = cr.std(ddof=1) / np.sqrt(len(cr)) if len(cr) > 1 else 0
    rem = f"{rng([x['sign_removed'] for x in det])} ({rng([x['sign_bad'] for x in det])} / {rng([x['sign_lookalike'] for x in det])})"
    seeds = [d['args']['seed'] for d in ds]
    refs = [json.load(open(f'stressref_{tag}_s{s}.json')) for s in seeds if glob.glob(f'stressref_{tag}_s{s}.json')]
    refs = [r for r in refs if 'oracle' in r]
    ref = lambda k: f"{np.mean([100 * np.mean([r[k][t] for t in T]) for r in refs]):.1f}" if len(refs) == len(ds) else '--'
    cells = rem, ref('unfixed'), f'{cr.mean():.1f}$\\pm${se:.1f}', ref('oracle')
    out[key] = dict(n=len(ds), unfixed=unf, removed=rem, crash=f'{cr.mean():.1f}+-{se:.1f}',
                    auroc=rng([round(x['auroc'], 2) for x in det]), per_seed=[round(c, 1) for c in cr],
                    tex=' & '.join(c for c in cells if c) + ' \\\\')
json.dump(out, sys.stdout, indent=1)
