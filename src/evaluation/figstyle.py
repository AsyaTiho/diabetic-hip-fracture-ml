"""Shared figure style for the hipfx paper, extracted from fig_m3/m4/m5.

Import and call apply() at the top of any figure script:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> evaluation/
    import figstyle; figstyle.apply()

Then use the named colors (figstyle.BEHRT, figstyle.FRAX, ...) so every figure reads
as one system: DejaVu Sans, hairline rules, Okabe-Ito colour-blind-safe palette,
editable text in the PDF.
"""
import matplotlib

# ── ink + Okabe-Ito colour-blind-safe palette (as in fig_m3/m4/m5) ──────
INK   = '#1a1a1a'   # near-black text / rules
MUT   = '#555555'   # muted grey (footnotes, secondary)
RULE  = '#1a1a1a'   # table rules
GREY  = '#9a9a9a'   # de-emphasised marks

BLUE     = '#0072B2'  # Okabe-Ito blue   -> BEHRT throughout
ORANGE   = '#D55E00'  # vermillion       -> FRAX / comparator throughout
GREEN    = '#009E73'
SKY      = '#56B4E9'
YELLOW   = '#F0E442'
PURPLE   = '#CC79A7'

# canonical role -> colour, so series are consistent across every figure
BEHRT = BLUE
FRAX  = ORANGE
XGB   = GREEN
NN    = PURPLE
TREAT_ALL = GREY

# per-risk-class palette (control, then the three fracture classes)
CLASS_COLORS = {0: GREY, 1: BLUE, 2: ORANGE, 3: GREEN}


def apply():
    """Single source of truth for the data-plot figure style (ROC, PR, DCA,
    calibration, confusion, tornado, UChicago panels). DejaVu Sans, clean spines,
    Okabe-Ito palette, editable PDF. Dense table figures (fig_m3/m4/m5) keep their
    own smaller explicit text sizes; this shares the typeface and treatment, not a
    single point size."""
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.sans-serif': ['DejaVu Sans', 'Nimbus Sans', 'Liberation Sans', 'Arial'],
        'mathtext.fontset': 'stixsans',
        'font.size': 14,
        'axes.titlesize': 15, 'axes.labelsize': 16,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
        'axes.edgecolor': INK, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'legend.frameon': False,
        'axes.grid': False,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def panel_label(ax, letter, x=-0.02, y=1.02):
    """Bold lowercase panel label (a, b, ...) in the m-figure style."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.4,
            fontweight='bold', color=INK, ha='right', va='bottom')
