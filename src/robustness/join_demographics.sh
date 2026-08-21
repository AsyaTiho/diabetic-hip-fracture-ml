#!/usr/bin/env bash
# Step 1b of the matched-cohort pipeline (see src/robustness/README.md).
# Read-only with respect to the source databases; writes only into the matching
# work directory.
#
# Joins the Delta=180 cohort enrolids to their demographics:
#   sex, dobyr           <- DEMO_2024.db
#   total_visibility     <- visibility_years2024.db  (YEARS; days = x 365.25)
#
# DEMO_2024.db and visibility_years2024.db are proprietary databases derived from
# licensed MarketScan data that you must supply yourself (they are not shipped):
# sex and birth year come from DEMO_2024.db, enrollment visibility from
# visibility_years2024.db. Point HIPFX_MARKETSCAN_DEMO_DIR at the folder holding
# them.
#
# Uses an indexed lookup from the cohort id table rather than scanning the DEMO
# file end to end. DEMO's enrolid index is NOT unique, so this aggregates per
# enrolid and reports any patient carrying conflicting sex or dobyr rather than
# letting a fan-out join silently duplicate patients.
set -euo pipefail

DATA_DIR="${HIPFX_BEHRT_DIR:-data/behrt}"
DEMO_DIR="${HIPFX_MARKETSCAN_DEMO_DIR:-data/marketscan_demographics}"

M="$DATA_DIR/matching"
IDS="$M/cohort_180_enrolids.txt"
DB="$M/match_ids.db"
OUT="$M/cohort_180_demographics.csv"
mkdir -p "$M"

echo "=== building id table  $(date '+%F %T') ==="
rm -f "$DB"
sqlite3 "$DB" "CREATE TABLE ids(enrolid INTEGER PRIMARY KEY);"
sqlite3 "$DB" ".mode csv" ".import '$IDS' ids"
echo "  ids loaded: $(sqlite3 "$DB" 'SELECT COUNT(*) FROM ids;')"

echo
echo "=== duplicate / conflict check in DEMO  $(date '+%F %T') ==="
sqlite3 "$DB" "
ATTACH '$DEMO_DIR/DEMO_2024.db' AS d;
SELECT 'rows_per_enrolid>1', COUNT(*) FROM (
  SELECT i.enrolid FROM ids i JOIN d.DEMO m ON m.enrolid = i.enrolid
  GROUP BY i.enrolid HAVING COUNT(*) > 1
);
SELECT 'conflicting_sex_or_dobyr', COUNT(*) FROM (
  SELECT i.enrolid FROM ids i JOIN d.DEMO m ON m.enrolid = i.enrolid
  GROUP BY i.enrolid
  HAVING COUNT(DISTINCT m.sex) > 1 OR COUNT(DISTINCT m.dobyr) > 1
);
"

echo
echo "=== joining  $(date '+%F %T') ==="
sqlite3 -header -csv "$DB" "
ATTACH '$DEMO_DIR/DEMO_2024.db' AS d;
ATTACH '$DEMO_DIR/visibility_years2024.db' AS v;
SELECT i.enrolid,
       (SELECT MIN(m.sex)   FROM d.DEMO m WHERE m.enrolid = i.enrolid) AS sex,
       (SELECT MIN(m.dobyr) FROM d.DEMO m WHERE m.enrolid = i.enrolid) AS dobyr,
       (SELECT MIN(r.total_visibility) FROM v.visibility_results r
         WHERE r.enrolid = i.enrolid) AS total_visibility
FROM ids i;
" > "$OUT"

echo
echo "=== done  $(date '+%F %T') ==="
wc -l < "$OUT"
ls -la "$OUT"
