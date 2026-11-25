# How the cohort was built

This repository starts from `patient_sequences.pkl`, `hip_frac_labels_CANONICAL.pkl`
and `one_hot_features_4groups.npz`. The code that produces those files from the
raw MarketScan databases is not shipped, because it is useless without the
licensed source data and because it contains the database paths of the machine it
ran on. This document records what it does, in enough detail to reimplement or to
audit, including a defect found on 2026-11-24 and corrected.

## The source

Three SQLite databases built from the MarketScan Commercial Claims and Encounters
extract:

| Database | Table | Columns |
| --- | --- | --- |
| `PHE_2024.db` | `PHE` | `enrolid`, `intday`, `phecode` |
| `DEMO_2024.db` | `DEMO` | `enrolid`, `dobyr`, `sex` |
| `ENROL_INTERVALS_2024.db` | enrollment spans | `enrolid`, interval start and end |

`intday` is days since 1 January 2003. `phecode` is a PhecodeX phenotype, mapped
from ICD-9 and ICD-10 using the tables in `reference/phecode_maps/`. `PHE` holds
one row per patient per phenotype per day, about 10 billion rows, with three
covering indexes:

```sql
CREATE INDEX ind_enrolid ON PHE(enrolid, intday, phecode);
CREATE INDEX ind_intday  ON PHE(intday, enrolid, phecode);
CREATE INDEX ind_phecode ON PHE(phecode, enrolid, intday);
```

Age at an event is computed as `int(2003 + intday / 365.25 - dobyr)`. Every age in
the study derives from `dobyr`; there is no recorded age field.

## Inclusion criteria, in the order they are applied

Applied cheapest-set-first so each criterion narrows the next.

**1. Adult-onset diabetes.** The patient's *earliest* diabetes code falls at age
20 or older. Diabetes is `EM_202`, `EM_202.1`, `EM_202.2` (PhecodeX: diabetes
mellitus, type 1, type 2).

```sql
SELECT enrolid, MIN(intday) FROM PHE WHERE phecode = ? GROUP BY enrolid
```

One statement per phecode. `WHERE phecode = ? GROUP BY enrolid` matches the
leading columns of `ind_phecode`, so SQLite seeks to the phecode and walks
forward through rows already ordered by `enrolid`: the groups arrive pre-sorted,
`MIN` is the first row of each group, no temporary sort is built, and the table
itself is never read because the index carries `intday`. The three per-phecode
minima are then reduced to one per patient, and the age test is applied once, to
that earliest day.

**2. Adult at enrolment.** Aged 20 or older at the start of the first enrollment
interval, from `ENROL_INTERVALS_2024.db` joined to `DEMO`.

**3. Continuous enrollment over one year.** Total covered days, summed across
intervals and gap-aware, exceeding 365.

**4. Consistent demographics.** Patients whose `DEMO` rows disagree with each
other on birth year or sex are excluded, because every age in the study is
computed from `dobyr` and theirs is unresolvable.

## The defect in criterion 1, and the correction

Until 2026-11-24 criterion 1 was written as a single existential query:

```sql
SELECT DISTINCT p.enrolid
FROM PHE p JOIN demo.DEMO d ON p.enrolid = d.enrolid
WHERE p.phecode IN ('EM_202', 'EM_202.1', 'EM_202.2')
  AND CAST(2003 + p.intday / 365.25 - d.dobyr AS INTEGER) >= 20
```

Read it row by row. It admits a patient if **any single** diabetes code falls at
age 20 or older, so a patient diagnosed at 12 who still has diabetes codes at 20,
which is every type 1 patient surviving into adulthood, qualifies. `DISTINCT`
then collapses them to one row.

The rule is existential where it should concern the earliest event. The cause is
structural rather than a typo: a test inside `WHERE` is evaluated per row, so it
can only ever ask whether *some* row qualifies. Making it a statement about onset
requires the aggregation to happen first and the age test to happen after, which
is the `MIN(intday) ... GROUP BY enrolid` form above.

An audit applying the age test after aggregation, with no filter inside the
query, found **8,872 patients (0.08%)** of 10,506,954 whose earliest diabetes code
predates age 20:

| Age at earliest diabetes code | Patients |
| --- | --- |
| under 5 | 16 |
| 5 to 9 | 256 |
| 10 to 14 | 1,415 |
| 15 to 19 | 7,185 |

Median 17. Small in proportion, but these are the patients with the longest
disease duration, and the Methods state they are not in the cohort.

The corrected criterion was checked against the prediction that it should remove
exactly those patients and no others, by set identity rather than by count:

```
previous cohort            10,506,954
pediatric-onset audit list      8,872
previous minus pediatric   10,498,082
corrected criterion 1      10,498,082
in new only: 0   in expected only: 0
```

The corrected form is also eleven times faster, 95 s against 1,108 s, because the
existential form had to join to `DEMO`, evaluate an age expression per row, and
then deduplicate a large intermediate.

## Extraction

For each eligible patient, one aggregated query driven from the sorted id list:

```sql
SELECT phecode, MIN(intday), MAX(intday), COUNT(*), GROUP_CONCAT(intday)
FROM PHE
WHERE enrolid = ? AND intday >= ?
GROUP BY phecode
```

`WHERE enrolid = ?` is a seek into `ind_enrolid`, so each patient is one short
index range; running the ids in sorted order walks those ranges front to back.
All four aggregates are computed as SQLite traverses the range.

The second bound deserves a note. The age rule is a statement about an event's
age, but age increases monotonically with `intday` for a fixed birth year, so
"age at least 20" is equivalent to "`intday` at or above some threshold". That
threshold is found by walking to the boundary of the original expression rather
than rearranging it algebraically, so floating point cannot shift it by a day:

```python
def min_intday_for_age20(dobyr):
    """Smallest intday d such that int(2003 + d / 365.25 - dobyr) >= 20."""
    t = max(0, int((dobyr - 1983) * 365.25))
    while t > 0 and int(2003 + (t - 1) / 365.25 - dobyr) >= 20:
        t -= 1
    while int(2003 + t / 365.25 - dobyr) < 20:
        t += 1
    return t
```

A per-event filter therefore becomes a narrowing of the index range: the rows are
never visited rather than fetched and rejected.

This replaced a two-stage path that ran `SELECT enrolid, intday, phecode FROM PHE`
with no `WHERE` clause, pulled all ~10 billion rows into Python, discarded the
95% whose `enrolid` was not eligible, built a nested dictionary of patient to
phecode to day list, and then hand-wrote the `GROUP BY` as a double loop over it.
The rewrite was checked by slicing the original function out of the superseded
script and executing it, rather than reimplementing it, over the same
5,000-patient subset: 193,416 `(enrolid, phecode)` pairs from both paths, zero
differences in first day, last day, count or day set, including the 413 events the
age rule removes.

## Sequences and vocabulary

Each patient's `(phecode, day)` pairs are expanded to one token per occurrence
and sorted chronologically. Events sharing a calendar day form one visit; visits
are separated by a `SEP` token, which also terminates the record. Each token
carries the patient's age at that event. The two hip-fracture phenotypes
(`MS_745.11`, `MS_745.9`) are excluded from the input for cases and controls
alike.

The vocabulary is fitted **once**, over the whole aggregate in row order, and
every consumer reads that file. Fitting it per stage is how two files come to
disagree about which integer means which phenotype; the two vocabulary files
written by the pipeline are checked for byte equality.

Records are truncated to the last 511 tokens (`MAX_LEN = 512`). On the November
2026 corpus that affects 26.3% of patients and discards 29.9% of tokens, oldest
first. The truncation is not uniform across arms: landmark records are cut at
diabetes index + 180 days and almost never reach the limit, while cut-at-fracture
controls carry a lifetime of codes.

## Labels

`days_to_hip_frac` is the earliest hip-fracture day minus the earliest diabetes
day, both taken from the aggregate's `MIN(intday)`. The class boundaries are in
`src/config/risk_classes.py`, which is the only place they are defined.

Deriving the index dates by exploding the aggregate into one row per event,
computing an age per event and filtering, is equivalent to taking the stored
minimum, and was verified as such on a 50,000-patient sample: 50,000 of 50,000
diabetes index dates and 686 of 686 hip-fracture index dates identical, with zero
events removed by the per-event age filter. The per-event filter is redundant
once criterion 1 enforces adult onset at the patient level, and a patient-level
assertion replaces it.

## Resulting cohort, November 2026

| Criterion | Patients | Time |
| --- | --- | --- |
| 1. earliest diabetes code at age >= 20 | 12,847,699 | 95 s |
| 2. aged >= 20 at enrolment | 12,805,279 | 297 s |
| 3. continuous enrollment over one year | 10,498,083 | 143 s |
| 4. consistent demographics | **10,498,082** | |

Criterion 1 begins from the 15,564,377 patients carrying any diabetes code, then
drops 282,895 for pediatric onset and 2,433,783 for having no recorded birth
year.

Extraction yields 401,404,225 rows over 3,308 distinct phecodes; 9,877,218 rows
carry a phecode outside the PhecodeX map and are dropped, leaving 391,527,007.
The vocabulary is 3,310 tokens: 5 special plus 3,305 phenotypes.

## A note on reproducing any of this

These counts come from `PHE_2024.db` as rebuilt on 13 November 2026. An earlier
build of the same source was corrupted by a `pandas.read_sas` 32-bit row-offset
overflow above 2^28 rows, which wrote the numeric columns of later rows to the
wrong positions while text columns survived: diagnosis codes stayed valid but
attached to the wrong patients on the wrong dates, for the 2021 to 2023 extracts.
Any cohort built from that earlier database is roughly 12.7% too large, because
one diabetes code admits a patient and no single code evicts one, so corruption
inflates a one-hit criterion about elevenfold.
