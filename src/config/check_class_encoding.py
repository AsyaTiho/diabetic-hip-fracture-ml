#!/usr/bin/env python3
"""Guard: every file that encodes the four risk classes must agree.

A class mapping is a contract between code and the artifacts it reads.  The
saved predictions in iterations/ and baselines/ carry the JUNE order (native 1
is 1-5 years, native 2 is < 1 year); the November rerun's labels carry the PAPER
order from ver_8_npj.tex eq:risk-groups (native 1 is < 1 year).  A file whose
mapping disagrees with the artifact it reads produces a figure with two classes
swapped, no exception, no warning.

This script parses every .py file with ast (nothing is imported, so nothing is
executed) and runs three checks:

  1. INTERNAL.   Within one file, CLASS_NAMES/CLS/NAMES must agree with
                 DISPLAY_TO_NATIVE and DISPLAY_NAMES.  Valid whichever
                 convention the file uses, so it catches a half-edited file.
  2. CONSENSUS.  Every non-archive file must agree with every other.  Catches a
                 partial migration without needing to name the right answer.
  3. EXPECTED.   Every non-archive file must match --expect.  Use 'june' until
                 the artifacts are regenerated, then 'paper'.

Exit status is 0 only if all requested checks pass.

Usage:
    python3 foundation/config/check_class_encoding.py                  # expect june
    python3 foundation/config/check_class_encoding.py --expect paper
    python3 foundation/config/check_class_encoding.py --list           # show all mappings
"""
import argparse
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# repo layout: this file is src/config/, so ROOT is the repo root

# Directories whose contents are frozen provenance: parsed and reported, never enforced.
FROZEN = ()   # this repo carries no archive

# The declared source of truth.  Checked for internal consistency, then used as
# the reference; it is not a consumer, so it is excluded from the consumer checks.
CANONICAL = 'src/config/risk_classes.py'

# This file defines JUNE and PAPER as data; it is not a consumer of either.
SELF = 'src/config/check_class_encoding.py'

# The two conventions, as {native index: canonical token}.
JUNE  = {0: 'control', 1: 'y1_5', 2: 'lt1', 3: 'y5_10'}
PAPER = {0: 'control', 1: 'lt1', 2: 'y1_5', 3: 'y5_10'}
CONVENTIONS = {'june': JUNE, 'paper': PAPER}

# Any dict literal keyed by small ints whose values are class labels counts as a
# mapping, whatever it is called.  Naming it would let the next local CLS dict
# through.  Three entries is enough (fracture-only dicts omit control).
MIN_ENTRIES = 3


def token(label):
    """Normalise a human label to one of four tokens, or None if not a class label."""
    s = str(label).lower().replace('–', '-').replace('—', '-')
    s = s.replace('$', '').replace('\\', '').strip()
    if 'control' in s or '>10' in s or '> 10' in s or 'no hip fracture' in s and 'within 10' in s:
        return 'control'
    nums = re.findall(r'\d+', s)
    if re.search(r'<\s*1\b', s) or 'within 1 year' in s or ('1' in nums and len(nums) == 1
                                                            and re.search(r'\b1\s*(y|year)', s)
                                                            and '-' not in s):
        return 'lt1'
    if '1' in nums and '5' in nums and '10' not in nums:
        return 'y1_5'
    if '5' in nums and '10' in nums:
        return 'y5_10'
    return None


def const_ints(tree):
    """Module-level NAME = <int> and NAME, ... = <ints>, so dicts keyed by name resolve."""
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        tgt, val = node.targets[0], node.value
        if isinstance(tgt, ast.Name) and isinstance(val, ast.Constant) and isinstance(val.value, int):
            out[tgt.id] = val.value
        elif isinstance(tgt, ast.Tuple) and isinstance(val, ast.Tuple):
            for t, v in zip(tgt.elts, val.elts):
                if isinstance(t, ast.Name) and isinstance(v, ast.Constant) and isinstance(v.value, int):
                    out[t.id] = v.value
    return out


def extract(path):
    """Return {'maps': {varname: {idx: tok}}, 'd2n': [...], 'dnames': [tok,...]}."""
    try:
        src = open(path, errors='ignore').read()
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    ints = const_ints(tree)
    found = {'maps': {}, 'd2n': None, 'dnames': None, 'order_only': []}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not targets or not isinstance(targets[0], ast.Name):
            continue
        name, val = targets[0].id, node.value
        if name in ('CLASS_NAMES_LONG',):      # same mapping, longer strings
            pass

        if isinstance(val, ast.Dict):
            m = {}
            for k, v in zip(val.keys, val.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, int):
                    idx = k.value
                elif isinstance(k, ast.Name) and k.id in ints:
                    idx = ints[k.id]
                else:
                    continue
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    t = token(v.value)
                    if t:
                        m[idx] = t
            if len(m) >= MIN_ENTRIES and set(m.values()) <= {'control','lt1','y1_5','y5_10'} \
                    and len(set(m.values())) == len(m):
                found['maps'][name] = m

        elif isinstance(val, (ast.List, ast.Tuple)) and val.elts and all(
                isinstance(e, ast.Tuple) and len(e.elts) == 2 for e in val.elts):
            m = {}
            for e in val.elts:
                k, v = e.elts
                if isinstance(k, ast.Constant) and isinstance(k.value, int) \
                        and isinstance(v, ast.Constant) and isinstance(v.value, str):
                    t = token(v.value)
                    if t:
                        m[k.value] = t
            if len(m) >= MIN_ENTRIES and len(set(m.values())) == len(m):
                found['maps'][name] = m

        elif name == 'DISPLAY_TO_NATIVE' and isinstance(val, (ast.List, ast.Tuple)):
            xs = [e.value for e in val.elts
                  if isinstance(e, ast.Constant) and isinstance(e.value, int)]
            if len(xs) == 4:
                found['d2n'] = xs

        elif isinstance(val, (ast.List, ast.Tuple)) and val.elts and 3 <= len(val.elts) <= 4 \
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in val.elts) \
                and all(token(e.value) for e in val.elts) and name != 'DISPLAY_NAMES':
            found.setdefault('order_only', []).append(
                (name, [token(e.value) for e in val.elts]))

        elif name == 'DISPLAY_NAMES' and isinstance(val, (ast.List, ast.Tuple)):
            xs = [token(e.value) for e in val.elts
                  if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if len(xs) == 4 and all(xs):
                found['dnames'] = xs

    return found if (found['maps'] or found['d2n'] or found['dnames']
                     or found['order_only']) else None


def agrees(a, b):
    """True if two mappings agree on every index they both state."""
    shared = set(a) & set(b)
    return bool(shared) and all(a[k] == b[k] for k in shared)


def consensus_map(f):
    """The one native mapping a file implies, or None if it states none."""
    for m in f['maps'].values():
        return m
    if f['d2n'] and f['dnames']:
        return {native: tok for tok, native in zip(f['dnames'], f['d2n'])}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expect', choices=['june', 'paper', 'none'], default='june')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    files = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in ('.git', '__pycache__', 'node_modules')]
        for f in sorted(fn):
            if f.endswith('.py'):
                files.append(os.path.join(dp, f))

    rows = []
    for p in files:
        got = extract(p)
        if got:
            rows.append((os.path.relpath(p, ROOT), got))

    fail_internal, fail_consensus, fail_expected = [], [], []

    for rel, f in rows:
        frozen = rel.startswith(FROZEN) or rel == SELF

        # 1. internal
        maps = list(f['maps'].items())
        implied = None
        if f['d2n'] and f['dnames']:
            implied = {native: tok for tok, native in zip(f['dnames'], f['d2n'])}
        for nm, m in maps:
            if implied and not agrees(m, implied):
                fail_internal.append((rel, nm, m, implied))
        for i in range(len(maps)):
            for j in range(i + 1, len(maps)):
                if not agrees(maps[i][1], maps[j][1]):
                    fail_internal.append((rel, f'{maps[i][0]} vs {maps[j][0]}',
                                          maps[i][1], maps[j][1]))

        cm = consensus_map(f)
        if cm and not frozen and rel != CANONICAL:
            # 3. expected
            if args.expect != 'none' and not agrees(cm, CONVENTIONS[args.expect]):
                fail_expected.append((rel, cm))

    # 2. consensus across live files
    live = [(rel, consensus_map(f)) for rel, f in rows
            if not rel.startswith(FROZEN) and rel not in (CANONICAL, SELF)
            and consensus_map(f)]
    groups = {}          # convention name -> files compatible with it
    for rel, m in live:
        hit = [k for k, v in CONVENTIONS.items() if agrees(m, v)]
        key = hit[0] if len(hit) == 1 else ('AMBIGUOUS' if hit else 'OTHER')
        groups.setdefault(key, []).append((rel, m))
    enforced = {k: v for k, v in groups.items() if k not in ('AMBIGUOUS',)}
    if len(enforced) > 1:
        fail_consensus = [(k, [r for r, _ in v]) for k, v in enforced.items()]

    if args.list:
        for rel, f in rows:
            cm = consensus_map(f)
            tag = 'FROZEN ' if rel.startswith(FROZEN) else '       '
            which = next((k for k, v in CONVENTIONS.items() if cm == v), 'OTHER') if cm else '-'
            print(f'{tag}{which:6s}  {rel}')
            if cm:
                print(f'          {cm}')

    print(f'\nparsed {len(files)} .py files; {len(rows)} encode the risk classes '
          f'({len(live)} live, {len(rows) - len(live)} frozen or partial)')

    canon = next((consensus_map(f) for rel, f in rows if rel == CANONICAL), None)
    canon_name = next((k for k, v in CONVENTIONS.items() if canon == v), 'OTHER') if canon else None

    ok = True
    print(f'[0] CANONICAL module ........... ', end='')
    if canon is None:
        ok = False
        print(f'FAIL, {CANONICAL} declares no mapping')
    else:
        print(f'declares {canon_name.upper()}  {canon}')

    print(f'\n[1] INTERNAL consistency ....... ', end='')
    if fail_internal:
        ok = False
        print(f'FAIL, {len(fail_internal)} disagreements')
        for rel, what, a, b in fail_internal:
            print(f'      {rel}: {what}\n        {a}\n        {b}')
    else:
        print('PASS')

    print(f'[2] CONSENSUS across live files  ', end='')
    if fail_consensus:
        ok = False
        print(f'FAIL, {len(fail_consensus)} different conventions in use')
        for which, paths in fail_consensus:
            print(f'      {which}:')
            for p in paths:
                print(f'        {p}')
    else:
        print('PASS' if live else 'PASS (nothing to compare)')

    print(f'[3] EXPECTED = {args.expect:<16s} ', end='')
    if args.expect == 'none':
        print('skipped')
    elif fail_expected:
        ok = False
        print(f'FAIL, {len(fail_expected)} files disagree')
        for rel, m in fail_expected:
            print(f'      {rel}\n        got      {m}\n        expected {CONVENTIONS[args.expect]}')
    else:
        print('PASS')

    if canon and args.expect != 'none' and canon != CONVENTIONS[args.expect] and ok:
        n = len(live)
        print(f'\nTRANSITION IN PROGRESS, not an error:')
        print(f'  the canonical module declares {canon_name.upper()}, consumers are on '
              f'{args.expect.upper()}.')
        print(f'  {n} consumer files must migrate, and every artifact they read must be')
        print(f'  regenerated, before --expect {canon_name} can be the enforced value.')

    unver = [(rel, f['order_only']) for rel, f in rows
             if f.get('order_only') and not f['maps'] and not f['d2n']
             and not rel.startswith(FROZEN) and rel != SELF]
    if unver:
        print(f'\n[4] NOT STATICALLY VERIFIABLE ... {len(unver)} files')
        print('      These name the classes in a bare ordered list with no integer')
        print('      mapping, so the coupling to native indices is positional and')
        print('      only visible at run time. Check them by eye when migrating.')
        for rel, lists in unver:
            for nm, order in lists:
                print(f'      {rel}: {nm} = {order}')

    print('\n' + ('GUARD PASS' if ok else 'GUARD FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
