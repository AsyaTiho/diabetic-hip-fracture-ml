"""The four-class outcome definition, in one place.

Transcribed from ver_8_npj.tex, eq:risk-groups:

    y = 0   control: no fracture, dt <= 0, or dt > 3650
        1   1    <= dt <= 365     (< 1 year)
        2   366  <= dt <= 1825    (1-5 years)
        3   1826 <= dt <= 3650    (5-10 years)

where dt = t_HF - t_DM in days.

The June pipeline used 1 = 1-5 years and 2 = < 1 year, the reverse of the
manuscript equation, consistently across _build_canonical_labels.py and
build_one_hot_features.py.  No published number is wrong as a result (the
paper's prose reports the counts in chronological order), but the two
conventions cannot both survive a rebuild.  The manuscript is the reference, so
the code moves.

Every producer and consumer of risk_class imports from here.  Do not redefine
these boundaries or these names anywhere else.
"""
import numpy as np

CONTROL, LT_1Y, Y1_5, Y5_10 = 0, 1, 2, 3

CLASS_NAMES = {
    CONTROL: 'Control',
    LT_1Y:   '<1 year',
    Y1_5:    '1-5 years',
    Y5_10:   '5-10 years',
}

CLASS_NAMES_LONG = {
    CONTROL: 'Control (no hip fracture within 10 years of diabetes)',
    LT_1Y:   'Hip fracture within 1 year',
    Y1_5:    'Hip fracture 1-5 years',
    Y5_10:   'Hip fracture 5-10 years',
}

DAY_1Y, DAY_5Y, DAY_10Y = 365, 1825, 3650

# The June convention, kept only so old artifacts can be read and translated.
JUNE_TO_PAPER = {0: CONTROL, 1: Y1_5, 2: LT_1Y, 3: Y5_10}


def risk_class(dt):
    """Vectorised. `dt` is a Series or array of t_HF - t_DM, NaN if no fracture."""
    dt = np.asarray(dt, dtype='float64')
    return np.where(
        np.isnan(dt) | (dt <= 0) | (dt > DAY_10Y), CONTROL,
        np.where(dt <= DAY_1Y, LT_1Y,
        np.where(dt <= DAY_5Y, Y1_5, Y5_10))
    ).astype(np.int8)


def risk_class_scalar(dt):
    """Single value, for the one-hot builder's apply() call."""
    if dt is None or dt != dt or dt <= 0 or dt > DAY_10Y:
        return CONTROL
    if dt <= DAY_1Y:
        return LT_1Y
    if dt <= DAY_5Y:
        return Y1_5
    return Y5_10
