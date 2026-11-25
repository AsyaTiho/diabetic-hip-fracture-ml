"""Read the November sequence corpus. Shared by pre-training and fine-tuning.

Why this exists rather than pd.read_parquet().

The corpus is stored as parquet because mightymac writes pandas 3.0.6 pickles
whose column index uses the new StringDtype, which pandas 2.3.3 here cannot
unpickle.  But the corpus has three nested list columns, and pyarrow refuses to
materialise those in one pass:

    pyarrow.lib.ArrowNotImplementedError: Nested data conversions not
    implemented for chunked array outputs

It fails on a single row group too, so it is the list column itself being
chunked, not the row grouping.  iter_batches() sidesteps it by handing back one
RecordBatch at a time, which is never chunked.

Reading `enrolid` alone works through the normal path, so a caller that only
needs the id list should use columns=['enrolid'] and skip this.
"""
import hashlib
import os
import time
import numpy as _np
import pandas as pd
import pyarrow.parquet as pq

BATCH = 100_000


def read_sequences(path, limit=None, batch_size=BATCH, verbose=True):
    """Return the corpus as a DataFrame: enrolid | code | age | day."""
    t0 = time.time()
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    parts, n = [], 0
    for b in pf.iter_batches(batch_size=batch_size):
        parts.append(b.to_pandas())
        n += len(parts[-1])
        if verbose and n % 1_000_000 < batch_size:
            print(f'    {n:,} / {total:,} rows ({time.time()-t0:.0f}s)', flush=True)
        if limit and n >= limit:
            break
    df = pd.concat(parts, ignore_index=True)
    del parts
    if limit:
        df = df.iloc[:limit]
    if verbose:
        print(f'  read {len(df):,} rows in {time.time()-t0:.0f}s', flush=True)
    return df


def read_enrolids(path):
    """Just the id column, through the normal path."""
    return pd.read_parquet(path, columns=['enrolid'])['enrolid']


def _cache_key(path, token2idx, age2idx, max_len):
    """Fingerprint of everything the encoded arrays depend on.

    The vocabulary is part of the key, not just the corpus: a cache built with a
    different token2idx holds ids that mean different phenotypes, and reusing it
    would silently train on remapped tokens.  Size and mtime cover the corpus
    without rehashing 5.7 GB.
    """
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(f'{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{max_len}'.encode())
    for v in (token2idx, age2idx):
        h.update(repr(sorted(v.items())).encode())
    return h.hexdigest()[:16]


def read_encoded(path, token2idx, age2idx, max_len=512, batch_size=BATCH,
                 verbose=True, cache_dir=None):
    """Return enrolid | code | age with code/age already as int32 id arrays,
    truncated to the last (max_len - 1) tokens.

    Holding the corpus as Python strings does not fit: measured at 12.5 GB per
    million patients, so 132 GB for 10,498,082, on a shared box with 157 GB free
    and the loader then building int arrays on top of that.  pandas with
    ArrowDtype is worse, 155 GB.

    Encoding here instead is EXACTLY equivalent, not an approximation:

      * the loader maps every token through token2idx/age2idx in __init__, with
        the same `.get(x, unk)` fallbacks used below;
      * __getitem__ then keeps `self.code_ids[index][-(max_len - 1):]`, so
        tokens earlier than that are discarded in every case.

    Truncating and encoding during the read therefore produces the same arrays
    the loader would have produced, without 4.3 billion Python string objects
    existing at any point.
    """
    keep = max_len - 1
    unk = token2idx.get('UNK', 0)
    t0 = time.time()

    # ── cache ─────────────────────────────────────────────────────────────
    # Encoding maps about 3 billion tokens through a Python dict one at a time,
    # which takes roughly 65 minutes and pins one core.  That cost is paid on
    # every launch, including a restart after a crash.  Stored as three flat
    # arrays plus offsets, the encoded corpus is ~24 GB on disk and loads in
    # seconds via mmap.  The key includes the vocabularies, so a changed
    # token2idx invalidates it rather than silently remapping every token.
    cache_dir = cache_dir or os.path.dirname(os.path.abspath(path))
    key = _cache_key(path, token2idx, age2idx, max_len)
    cpath = os.path.join(cache_dir, f'.encoded_{key}.npz')

    if os.path.exists(cpath):
        if verbose:
            print(f'  cache hit {os.path.basename(cpath)}', flush=True)
        z = _np.load(cpath, mmap_mode='r')
        eid = z['enrolid']
        off = z['offsets']
        flat_c, flat_a = z['code'], z['age']
        codes = [_np.asarray(flat_c[off[i]:off[i + 1]]) for i in range(len(eid))]
        ages  = [_np.asarray(flat_a[off[i]:off[i + 1]]) for i in range(len(eid))]
        df = pd.DataFrame({'enrolid': eid, 'code': codes, 'age': ages})
        if verbose:
            print(f'  loaded {len(df):,} rows from cache in '
                  f'{time.time() - t0:.0f}s', flush=True)
        return df

    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    eids, codes, ages = [], [], []
    n, raw_tokens, kept_tokens = 0, 0, 0

    for b in pf.iter_batches(batch_size=batch_size,
                             columns=['enrolid', 'code', 'age']):
        d = b.to_pydict()
        eids.extend(d['enrolid'])
        for c, a in zip(d['code'], d['age']):
            raw_tokens += len(c)
            c = c[-keep:]
            a = a[-keep:]
            kept_tokens += len(c)
            codes.append(_np.fromiter((token2idx.get(x, unk) for x in c),
                                      dtype=_np.int32, count=len(c)))
            ages.append(_np.fromiter((age2idx.get(str(x), 0) for x in a),
                                     dtype=_np.int32, count=len(a)))
        n += len(d['enrolid'])
        if verbose and n % 1_000_000 < batch_size:
            print(f'    {n:,} / {total:,} rows ({time.time()-t0:.0f}s)', flush=True)

    df = pd.DataFrame({'enrolid': eids, 'code': codes, 'age': ages})

    # write the cache, atomically, so a kill mid-write leaves no half file
    try:
        lens = _np.fromiter((len(c) for c in codes), dtype=_np.int64,
                            count=len(codes))
        off = _np.zeros(len(lens) + 1, dtype=_np.int64)
        _np.cumsum(lens, out=off[1:])
        tmp = cpath + f'.tmp{os.getpid()}'
        _np.savez(tmp,
                  enrolid=_np.asarray(eids, dtype=_np.int64),
                  offsets=off,
                  code=_np.concatenate(codes) if codes else _np.empty(0, _np.int32),
                  age=_np.concatenate(ages) if ages else _np.empty(0, _np.int32))
        os.replace(tmp + '.npz' if os.path.exists(tmp + '.npz') else tmp, cpath)
        if verbose:
            print(f'  cached to {os.path.basename(cpath)} '
                  f'({os.path.getsize(cpath)/1e9:.1f} GB)', flush=True)
    except Exception as e:                      # a cache is an optimisation
        print(f'  WARNING: could not write the cache: {e}', flush=True)

    if verbose:
        print(f'  read and encoded {len(df):,} rows in {time.time()-t0:.0f}s',
              flush=True)
        print(f'  tokens: {raw_tokens:,} in the corpus, {kept_tokens:,} kept '
              f'after truncation to the last {keep} '
              f'({100*kept_tokens/raw_tokens:.1f}%)', flush=True)
    return df
