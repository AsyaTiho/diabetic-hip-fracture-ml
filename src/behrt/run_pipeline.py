#!/usr/bin/env python
"""Drive one BEHRT-LoRA arm (primary, landmark, or matched) from a config file.

Each config in configs/ sets the environment variables that finetune_behrt_lora.py
and the BEHRT inference already read (BEHRT_FINETUNE_DATA, BEHRT_SPLIT_FILE,
BEHRT_OUTPUT_DIR, BEHRT_FINETUNE_CKPT, and BEHRT_PRED_OUT for scoring). The same
fine-tune script serves every arm; the arms differ only in which sequences and
split they point at. Age is always blanked, so there is no age-mode switch.

Usage:
    python src/behrt/run_pipeline.py --config configs/primary.env
    python src/behrt/run_pipeline.py --config configs/landmark_180.env --score

--score additionally re-runs inference on the arm's held-out test set, writing
predictions to BEHRT_PRED_OUT (default: the primary MarketScan prediction file).

Scripts are launched from the repository root so that repo-relative data paths
(e.g. data/behrt) resolve correctly and each script's own directory is on
sys.path for its local imports. Building the arm's inputs (sequences, splits) is
a separate prior step; see src/robustness/README.md.
"""
import argparse
import os
import shlex
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINETUNE  = os.path.join('src', 'behrt', 'finetune_behrt_lora.py')
INFERENCE = os.path.join('src', 'evaluation', 'inference', 'behrt_marketscan.py')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='path to a configs/*.env file')
    ap.add_argument('--score', action='store_true',
                    help='after fine-tuning, score the arm test set (writes BEHRT_PRED_OUT)')
    ap.add_argument('--python', default=sys.executable,
                    help='python interpreter to run the steps with (default: this one)')
    args = ap.parse_args()

    config = os.path.abspath(args.config)
    if not os.path.isfile(config):
        sys.exit(f'config not found: {config}')

    py = shlex.quote(args.python)
    cfg = shlex.quote(config)

    # Source the config so shell ${VAR:-default} expansion and `export` behave
    # exactly as they would if a user sourced it by hand, then run the step(s)
    # in that same shell. mkdir -p keeps the output/checkpoint dirs from tripping
    # the first write.
    steps = (f'"$PYTHON" -u {shlex.quote(FINETUNE)}',)
    if args.score:
        steps += (f'"$PYTHON" -u {shlex.quote(INFERENCE)}',)

    script = (
        'set -euo pipefail\n'
        f'set -a; source {cfg}; set +a\n'
        f'PYTHON={py}\n'
        'mkdir -p "${BEHRT_OUTPUT_DIR:-.}"\n'
        'if [ -n "${BEHRT_PRED_OUT:-}" ]; then mkdir -p "$(dirname "$BEHRT_PRED_OUT")"; fi\n'
        + ' && \\\n'.join(steps) + '\n'
    )

    print(f'[run_pipeline] config = {config}')
    print(f'[run_pipeline] steps  = fine-tune' + (' + score' if args.score else ''))
    subprocess.run(['bash', '-c', script], cwd=REPO_ROOT, check=True)
    print('[run_pipeline] done.')


if __name__ == '__main__':
    main()
