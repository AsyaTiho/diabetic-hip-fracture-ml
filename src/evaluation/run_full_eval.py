"""Unified eval pipeline entry point.

Default behaviour (`python run_full_eval.py` with no args):
  1. SKIP MarketScan inference (reuse existing prediction npz files)
  2. Run UChicago BEHRT-LoRA inference (the 05-22 npz is stale)
  3. Build all figures (MarketScan + UChicago)
  4. Build all tables (MarketScan only — AoU/UChicago tables not in scope)

Examples:
  python run_full_eval.py                      # default flow above
  python run_full_eval.py --skip-uchicago      # MarketScan figures+tables only
  python run_full_eval.py --force-rerun        # re-run ALL inference (~30 min GPU)
  python run_full_eval.py --only marketscan    # MarketScan steps only
  python run_full_eval.py --only uchicago      # UChicago inference + figs only

All output lands in outputs/ (figs/, tables/, predictions/). Logs in logs/.
"""

import os
import sys
import argparse
import subprocess
import datetime

PIPE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPE)
LOGS = os.path.join(PIPE, 'logs')
os.makedirs(LOGS, exist_ok=True)


def _ts():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def _run_module(mod_path, args=()):
    """Run a python module as a subprocess, tee output to a logfile."""
    log = os.path.join(LOGS, f'{os.path.basename(mod_path).replace(".py","")}_{_ts()}.log')
    cmd = [sys.executable, mod_path, *args]
    print(f'\n>>> {" ".join(cmd)}\n>>> log: {log}', flush=True)
    with open(log, 'w') as f:
        f.write(f'# command: {" ".join(cmd)}\n# started: {datetime.datetime.now().isoformat()}\n\n')
        f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=1, universal_newlines=True)
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f'{mod_path} exited with code {rc}; see {log}')
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', choices=['marketscan', 'uchicago'],
                        help='restrict to one cohort')
    parser.add_argument('--skip-uchicago', action='store_true',
                        help='skip UChicago inference + figures')
    parser.add_argument('--force-rerun', action='store_true',
                        help='re-run all model inference even if cached')
    parser.add_argument('--models', default=None,
                        help='comma-separated subset: behrt,xgb,nn')
    parser.add_argument('--skip-figures', action='store_true')
    parser.add_argument('--skip-tables',  action='store_true')
    args = parser.parse_args()

    do_marketscan = (args.only != 'uchicago')
    do_uchicago   = (args.only != 'marketscan') and not args.skip_uchicago

    common = []
    if args.force_rerun:
        common.append('--force-rerun')

    # ── MarketScan ───────────────────────────────────────────────────
    if do_marketscan:
        if args.force_rerun:
            # inference modules don't exist yet — placeholder until built
            for mod in ('inference/xgb_inference.py',
                        'inference/nn_inference.py',
                        'inference/behrt_marketscan.py'):
                path = os.path.join(PIPE, mod)
                if os.path.exists(path):
                    _run_module(path)
                else:
                    print(f'!! {mod} not yet implemented — skipping', flush=True)

        if not args.skip_figures:
            fig_args = list(common)
            if args.models:
                fig_args.extend(['--models', args.models])
            _run_module(os.path.join(PIPE, 'figures/make_figures.py'), fig_args)

        if not args.skip_tables:
            _run_module(os.path.join(PIPE, 'tables/make_tables.py'), common)

    # ── UChicago ─────────────────────────────────────────────────────
    if do_uchicago:
        uc_inf = os.path.join(PIPE, 'inference/behrt_uchicago.py')
        uc_fig = os.path.join(PIPE, 'figures/make_uchicago_figures.py')
        if os.path.exists(uc_inf):
            _run_module(uc_inf)
        else:
            print(f'!! behrt_uchicago.py not yet implemented — skipping', flush=True)
        if not args.skip_figures and os.path.exists(uc_fig):
            _run_module(uc_fig)
        elif not args.skip_figures:
            print(f'!! make_uchicago_figures.py not yet implemented — skipping', flush=True)

    print('\n=== eval pipeline finished ===', flush=True)


if __name__ == '__main__':
    main()
