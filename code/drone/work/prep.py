"""Load NeuroBEM processed flights, keep every 4th sample (100 Hz), save one npz."""
import glob, os, re
import numpy as np, pandas as pd
fs = sorted(glob.glob('processed_data/**/*.csv', recursive=True))
desc = {}
for line in open('Flights.txt'):
    m = re.search(r'"([0-9-]+)";\s*%\s*(.*)', line)
    if m:
        desc[m.group(1)] = m.group(2).strip()
test = set(l.strip() for l in open('testset.txt') if l.strip())
out = {k: [] for k in ['data', 'flight', 'seg', 'test']}
names = []
for i, f in enumerate(fs):
    b = os.path.basename(f).replace('merged_', '').replace('.csv', '')
    fl, sg = b.rsplit('_seg_', 1)
    d = pd.read_csv(f).values[::4].astype(np.float32)
    names.append(b)
    out['data'].append(d)
    out['flight'].append(np.full(len(d), sorted(desc).index(fl) if fl in desc else -1))
    out['seg'].append(np.full(len(d), i))
    out['test'].append(np.full(len(d), b in test))
np.savez_compressed('work/nb100.npz', data=np.concatenate(out['data']), flight=np.concatenate(out['flight']),
                    seg=np.concatenate(out['seg']), test=np.concatenate(out['test']), names=np.array(names),
                    flights=np.array(sorted(desc)), desc=np.array([desc[k] for k in sorted(desc)]),
                    cols=np.array(pd.read_csv(fs[0], nrows=1).columns.tolist()))
print('saved', sum(len(d) for d in out['data']), 'rows;', len(set(names) & test), 'test segments')
