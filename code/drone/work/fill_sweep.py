"""Write pooled sweep cells into Table tab:sweep (rows keyed by their label and bad share), from pool_sweep.py.

usage (from code/drone/): python work/fill_sweep.py ../../paper/ras/main.tex"""
import json, os, subprocess, sys
TEX = sys.argv[1]
PRE = {'SWEEPBASE': r'Benchmark & 67\% & ',
       'SEV20': r'Thrust +20\% & 67\% & ', 'SEV35': r'\quad +35\% & 67\% & ', 'SEV80': r'\quad +80\% & 67\% & ',
       'N3': r'Bad flights 12 & 32\% & ', 'N5': r'\quad 20 & 50\% & ', 'N11': r'\quad 44 & 80\% & ',
       'L2': r'Plus look-alikes 8 & 61\% & ', 'L4': r'\quad 16 & 56\% & ', 'L8': r'\quad 32 & 49\% & '}
cells = json.loads(subprocess.check_output([sys.executable, 'pool_sweep.py'], cwd=os.path.dirname(os.path.abspath(__file__))))
lines = open(TEX).read().split('\n')
for key, pre in PRE.items():
    idx = [i for i, l in enumerate(lines) if l.startswith(pre)]
    assert len(idx) == 1, (key, idx)
    lines[idx[0]] = pre + cells[key]['tex']
    print(key, cells[key]['n'], lines[idx[0]])
open(TEX, 'w').write('\n'.join(lines))
