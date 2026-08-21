import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── 1. Load ────────────────────────────────────────────────────────────────
sequences = pd.read_pickle('patient_sequences.pkl')
# columns: enrolid | code (list of str) | age (list of str)

labels = pd.read_pickle('hip_frac_labels_CANONICAL.pkl')  # switched 2026-06-07 from stale hip_frac_labels.pkl
# columns: enrolid | risk_class | ...

# Paste the exact strings confirmed in Part 2
HIP_FRACTURE_PHENOTYPES = {'Fracture of femur', 'Pathological fracture'}

# ── 2. Build label lookup ──────────────────────────────────────────────────
label_dict = dict(zip(labels['enrolid'], labels['risk_class']))

# ── 3. Truncation function ─────────────────────────────────────────────────
def truncate_before_hip(code_seq, age_seq):
    """
    Find first occurrence of any hip fracture phenotype.
    Return everything strictly before that index.
    If not found, return sequences unchanged.
    """
    for i, token in enumerate(code_seq):
        if token in HIP_FRACTURE_PHENOTYPES:
            return code_seq[:i], age_seq[:i]
    return code_seq, age_seq

# ── 4. Apply truncation ────────────────────────────────────────────────────
truncated_codes = []
truncated_ages  = []

for _, row in tqdm(sequences.iterrows(), total=len(sequences)):
    c, a = truncate_before_hip(row['code'], row['age'])
    truncated_codes.append(c)
    truncated_ages.append(a)

sequences['code'] = truncated_codes
sequences['age']  = truncated_ages

# ── 5. Attach labels ───────────────────────────────────────────────────────
sequences['risk_class'] = sequences['enrolid'].map(label_dict)

# Patients not in labels (no diabetes) should not exist per your clarification
# but drop them defensively
missing_label = sequences['risk_class'].isna().sum()
if missing_label > 0:
    print(f"WARNING: {missing_label} patients have no label — dropping")
    sequences = sequences[sequences['risk_class'].notna()]

sequences['risk_class'] = sequences['risk_class'].astype(np.int8)

# ── 6. Drop empty sequences ────────────────────────────────────────────────
before = len(sequences)
sequences = sequences[sequences['code'].apply(len) > 0].reset_index(drop=True)
dropped = before - len(sequences)
print(f"Dropped {dropped} patients with empty post-truncation sequences")

# ── 7. Sanity checks ───────────────────────────────────────────────────────
print("\nFinal class distribution:")
print(sequences['risk_class'].value_counts().sort_index())

# Verify no hip fracture codes remain anywhere in the sequences
print("\nVerifying no hip fracture codes remain...")
leaked = sequences['code'].apply(
    lambda seq: any(t in HIP_FRACTURE_PHENOTYPES for t in seq)
).sum()
print(f"Sequences still containing hip fracture codes: {leaked}")  # must be 0

# ── 8. Save ────────────────────────────────────────────────────────────────
sequences.to_pickle('patient_sequences_finetune.pkl')
print(f"\nSaved {len(sequences):,} patients")
# Final schema: enrolid | code (list) | age (list) | risk_class (0, 1, or 2)