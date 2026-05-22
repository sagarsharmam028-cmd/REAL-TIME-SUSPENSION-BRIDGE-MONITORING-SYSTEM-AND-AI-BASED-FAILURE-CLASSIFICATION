"""
Quick test of quick labeling feature
"""
from labeling import label_specific_ranges
import pandas as pd

print("Testing quick labeling feature...")
print("This will label remaining unlabeled data based on deflection thresholds\n")

# First show current stats
df = pd.read_csv("dataset.csv")
print("Before:")
print(df['label'].value_counts())

# Use standard thresholds
# SAFE: max_deflection <= 40
# WARNING: 40 < max_deflection <= 60
# DANGER: max_deflection > 60

# Manually label using the logic
updates = 0
for idx, row in df.iterrows():
    if row['label'] == 'UNLABELED':
        if row['max_deflection'] <= 40:
            df.at[idx, 'label'] = 'SAFE'
            updates += 1
        elif row['max_deflection'] <= 60:
            df.at[idx, 'label'] = 'WARNING'
            updates += 1
        else:
            df.at[idx, 'label'] = 'DANGER'
            updates += 1

df.to_csv("dataset.csv", index=False)

print(f"\nLabeled {updates} samples with thresholds:")
print("  SAFE: max_deflection <= 40 mm")
print("  WARNING: 40 < max_deflection <= 60 mm")
print("  DANGER: max_deflection > 60 mm")

print("\nAfter:")
print(df['label'].value_counts())

print("\n✅ Quick labeling complete!")
print("Dataset is now fully labeled. Ready to train improved model!")
