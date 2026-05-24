#!/usr/bin/env python3
"""
Clean dataset - remove unlabeled rows, keep only labeled training data
"""
import pandas as pd
import os

print("\n" + "="*70)
print("🧹 CLEANING DATASET")
print("="*70)

dataset_path = "dataset.csv"

# Read current dataset
print(f"\n📖 Reading: {dataset_path}")
df = pd.read_csv(dataset_path)

print(f"   Total rows: {len(df)}")
print(f"\n📊 Current class distribution:")
print(df['label'].value_counts())

# Remove unlabeled and empty labels
print(f"\n🔍 Removing unlabeled rows...")
df = df[df["label"] != "UNLABELED"]
df = df[(df["label"].notna()) & (df["label"] != "")]

print(f"   Rows after cleaning: {len(df)}")

# Keep only valid labels
valid_labels = ['SAFE', 'WARNING', 'DANGER']
df = df[df['label'].isin(valid_labels)]

print(f"   Final labeled rows: {len(df)}")

# Create backup
backup_path = dataset_path.replace('.csv', f'_backup_full.csv')
df_full = pd.read_csv(dataset_path)
df_full.to_csv(backup_path, index=False)
print(f"\n💾 Backup saved: {backup_path} ({len(df_full)} rows)")

# Save cleaned dataset
df.to_csv(dataset_path, index=False)
print(f"\n✅ Cleaned dataset saved: {dataset_path}")

print(f"\n📊 FINAL CLASS DISTRIBUTION:")
print(df['label'].value_counts())

print(f"\n📈 SUMMARY:")
print(f"   Total samples: {len(df)}")
print(f"   SAFE:    {len(df[df['label'] == 'SAFE']):5d} ({len(df[df['label'] == 'SAFE'])/len(df)*100:5.1f}%)")
print(f"   WARNING: {len(df[df['label'] == 'WARNING']):5d} ({len(df[df['label'] == 'WARNING'])/len(df)*100:5.1f}%)")
print(f"   DANGER:  {len(df[df['label'] == 'DANGER']):5d} ({len(df[df['label'] == 'DANGER'])/len(df)*100:5.1f}%)")

print("\n" + "="*70)
print("✅ DATASET CLEANING COMPLETE")
print("="*70)
print("\n✅ Training data is now clean and locked!")
print("✅ All data is fully labeled")
print("✅ Ready for model training/evaluation")
