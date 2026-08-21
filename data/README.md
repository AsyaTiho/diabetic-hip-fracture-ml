# data/

Place input datasets here. **No patient data is shipped in this repository.**

The scripts read from absolute paths set near the top of each file / in the
`config.py` modules; point those at the copies you place here.

| Dataset | How to obtain |
| --- | --- |
| MarketScan (Merative/IBM Commercial Claims) | Licensed; via Merative under a data-use agreement. |
| All of Us | All of Us Researcher Workbench (controlled tier), credentialed access. |
| UChicago cohort | Institutional EHR; available only under the governing IRB / DUA. |

Public PheCode / PheCodeX mapping tables needed to tokenize ICD codes are
included separately under `../reference/phecode_maps/`.

Everything in this folder except this README is git-ignored.
