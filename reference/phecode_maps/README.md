# Phecode and ICD Mapping Reference Files

## Purpose
Static reference files used across all trials for mapping ICD-9/ICD-10 diagnosis codes
to Phecodes. These files are not specific to any one experiment and are shared across
the data preparation pipeline.

## Files

| File | Description |
|------|-------------|
| `phecode_definitions1.2.csv` | Official Phecode v1.2 definitions: phecode number, description, and category for every phecode. Used to look up what a given phecode code means. |
| `orig_phecode_map.csv` | Original ICD-9 → Phecode v1.2 mapping table. Maps individual ICD-9 codes to one or more phecodes. (~3.4M rows) |
| `Phecode_mapX_filtered.csv` | Filtered version of the Phecode X (extended) mapping table, covering both ICD-9 and ICD-10 codes. Used in the Dec 15 trial for a secondary mapping attempt. |
| `for_feat_analysis.csv` | Processed feature table used for feature importance and clinical interpretation analysis. |
| `scrap_book_for_data.ipynb` | Exploratory notebook for data investigation: loading `merged_data.npz`, checking sparsity, converting to sparse CSR format, and verifying `balanced_merged_data.npz` for duplicate feature names. |
