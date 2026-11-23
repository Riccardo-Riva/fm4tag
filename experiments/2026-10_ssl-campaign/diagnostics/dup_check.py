"""How many distinct jets does each split hold, and do splits share jets?

Both datasets are resampled (`countup` to the c-jet pt/eta shape), which repeats
jets.  A jet is identified by (eventNumber, pt, eta, mass) — resampled copies are
exact row copies — hashed to one uint64 (collision odds ~1e-3 over all rows).

Reports, per split of 112022 (the old λ-scan data) and 152051 (this campaign):
rows, distinct jets, copies per distinct jet, the same per flavour; distinct jets
in the first 1.86M rows of 152051 train (the low-label regime); and the distinct
jets shared between splits, within and across the two datasets.
"""

import sys

import h5py
import numpy as np

ROOT = '/storage3/DSIP/rriva/datasets/transforming-jet-flavor/data_parallel_preprocess_251112_'
SPLITS = {'112022': ['train', 'val', 'test'], '152051': ['train', 'val', 'test', 'pretrain']}
BLOCK = 2_000_000
P = [np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F), np.uint64(0x165667B19E3779F9)]


def load(tag: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    hashes, labels = [], []
    with h5py.File(f'{ROOT}{tag}/{split}/output/pp_output_{split}.h5', 'r') as f:
        d = f['jets']
        for a in range(0, d.shape[0], BLOCK):
            j = d.fields(['eventNumber', 'pt', 'eta', 'mass', 'flavour_label'])[a : a + BLOCK]
            bits = lambda x: np.ascontiguousarray(j[x], dtype=np.float32).view(np.uint32).astype(np.uint64)
            with np.errstate(over='ignore'):
                h = j['eventNumber'].astype(np.int64).view(np.uint64) * P[0]
                h ^= ((bits('pt') << np.uint64(32)) | bits('eta')) * P[1]
                h ^= bits('mass') * P[2]
            hashes.append(h)
            labels.append(j['flavour_label'].astype(np.int8))
    return np.concatenate(hashes), np.concatenate(labels)


def report(name: str, h: np.ndarray, lab: np.ndarray) -> np.ndarray:
    u, counts = np.unique(h, return_counts=True)
    print(f'\n== {name}: {len(h):,} rows, {len(u):,} distinct jets, {len(h) / len(u):.2f} rows per distinct jet')
    k, n = np.unique(counts, return_counts=True)
    print('   copies per distinct jet:', ', '.join(f'{a}x: {b:,}' for a, b in list(zip(k, n))[:10]), f'(max {counts.max()})')
    for c in np.unique(lab):
        hc = h[lab == c]
        print(f'   flavour {c}: {len(hc):,} rows, {len(np.unique(hc)):,} distinct, {len(hc) / len(np.unique(hc)):.2f} rows per distinct')
    sys.stdout.flush()
    return u


def main() -> None:
    distinct = {}
    for tag, splits in SPLITS.items():
        for split in splits:
            h, lab = load(tag, split)
            distinct[tag, split] = report(f'{tag} {split}', h, lab)
            if (tag, split) == ('152051', 'train'):
                head = report('152051 train, first 1.86M rows (low-label regime)', h[:1_860_000], lab[:1_860_000])
                print(f'   = {len(head) / len(distinct[tag, split]):.1%} of the distinct jets of the full split')
            del h, lab

    print()
    keys = list(distinct)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            n = len(np.intersect1d(distinct[a], distinct[b], assume_unique=True))
            print(f'shared distinct jets {" ".join(a)} & {" ".join(b)}: {n:,} ({n / len(distinct[b]):.1%} of the latter)')


if __name__ == '__main__':
    main()
