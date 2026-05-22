"""
Data Labeling Interface - Label unlabeled or mislabeled data
"""
import pandas as pd
import os
from datetime import datetime

LABELS = {
    '1': 'SAFE',
    '2': 'WARNING',
    '3': 'DANGER'
}

LABEL_DESCRIPTIONS = {
    'SAFE': '✅ No structural issues detected. Bridge is operating normally.',
    'WARNING': '🟡 Moderate deflection/vibration. Monitor closely for changes.',
    'DANGER': '🔴 Excessive deflection/vibration. Immediate inspection required.'
}


def display_data_sample(row, index, total):
    """Display a data sample with context"""
    print("\n" + "="*60)
    print(f"📊 Sample {index + 1}/{total}")
    print("="*60)
    
    print(f"\n📍 Sensor Readings:")
    print(f"   Mean Deflection:    {row['mean_deflection']:.2f} mm")
    print(f"   Max Deflection:     {row['max_deflection']:.2f} mm")
    print(f"   Vibration Std Dev:  {row['vibration_std']:.4f} g")
    print(f"   Max Vibration:      {row['vibration_max']:.4f} g")
    print(f"   Deflection Rate:    {row['deflection_rate']:.2f} mm/sample")
    
    current_label = row.get('label', 'UNLABELED')
    print(f"\n🏷️  Current Label: {current_label}")


def display_label_options():
    """Show labeling options"""
    print("\n📋 Choose Label:")
    print("   1️⃣  SAFE    - " + LABEL_DESCRIPTIONS['SAFE'])
    print("   2️⃣  WARNING - " + LABEL_DESCRIPTIONS['WARNING'])
    print("   3️⃣  DANGER  - " + LABEL_DESCRIPTIONS['DANGER'])
    print("   4️⃣  SKIP    - Skip this sample")
    print("   0️⃣  EXIT    - Save and exit")


def get_label_input():
    """Get user input for label"""
    while True:
        user_input = input("\n👉 Enter choice (0-4): ").strip()
        if user_input in ['0', '1', '2', '3', '4']:
            return user_input
        print("❌ Invalid choice. Please enter 0-4.")


def label_data(csv_path="dataset.csv", label_only_unlabeled=True):
    """Interactive labeling interface"""
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    # Read dataset
    df = pd.read_csv(csv_path)
    
    # Filter based on option
    if label_only_unlabeled:
        data_to_label = df[df["label"] == "UNLABELED"].copy()
        print(f"\n🔍 Found {len(data_to_label)} unlabeled samples")
    else:
        data_to_label = df.copy()
        print(f"\n🔍 Total {len(data_to_label)} samples")
    
    if len(data_to_label) == 0:
        print("✅ No samples to label!")
        return
    
    # Reset index for proper iteration
    data_to_label = data_to_label.reset_index(drop=True)
    original_df = df.copy()
    
    labeled_count = 0
    skipped_count = 0
    
    print("\n" + "="*60)
    print("🏷️  DATA LABELING INTERFACE")
    print("="*60)
    print("Label data to improve model training")
    print("Focus on DANGER and SAFE samples for class balance\n")
    
    for idx, (i, row) in enumerate(data_to_label.iterrows()):
        display_data_sample(row, idx, len(data_to_label))
        display_label_options()
        
        choice = get_label_input()
        
        if choice == '0':  # Exit
            print("\n💾 Saving labeled data...")
            break
        elif choice == '4':  # Skip
            skipped_count += 1
            print("⏭️  Skipped")
            continue
        else:  # Label selected
            new_label = LABELS[choice]
            # Update in both dataframes
            original_df.loc[original_df.index[i], 'label'] = new_label
            data_to_label.loc[i, 'label'] = new_label
            labeled_count += 1
            print(f"✅ Labeled as: {new_label}")
    
    # Save updated dataset
    if labeled_count > 0 or skipped_count > 0:
        # Create backup
        backup_path = csv_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(backup_path, index=False)
        print(f"   Backup saved: {backup_path}")
        
        # Save updated dataset
        original_df.to_csv(csv_path, index=False)
        print(f"   Updated dataset: {csv_path}")
        
        print(f"\n📈 Labeling Summary:")
        print(f"   Labeled: {labeled_count} samples")
        print(f"   Skipped: {skipped_count} samples")
        
        # Show class distribution
        print(f"\n📊 Updated Class Distribution:")
        print(original_df['label'].value_counts())
    else:
        print("\n⚠️ No changes made")


def review_dataset_stats(csv_path="dataset.csv"):
    """Show dataset statistics"""
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    
    print(f"\n📈 Total Samples: {len(df)}")
    
    print(f"\n🏷️  Label Distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        bar = "█" * int(percentage / 5)
        print(f"   {label:10} : {count:5} ({percentage:5.1f}%) {bar}")
    
    print(f"\n📊 Feature Statistics:")
    numeric_cols = ['mean_deflection', 'max_deflection', 'vibration_std', 'vibration_max', 'deflection_rate']
    for col in numeric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            max_val = df[col].max()
            min_val = df[col].min()
            print(f"   {col:20} | Min: {min_val:8.2f} | Mean: {mean_val:8.2f} | Max: {max_val:8.2f}")
    
    # Imbalance analysis
    unlabeled = len(df[df['label'] == 'UNLABELED'])
    print(f"\n⚠️  Unlabeled Samples: {unlabeled}")
    
    if len(df[df['label'] == 'DANGER']) < 50:
        print(f"⚠️  WARNING: Only {len(df[df['label'] == 'DANGER'])} DANGER samples. Need more for robust model!")
    if len(df[df['label'] == 'SAFE']) < 100:
        print(f"⚠️  WARNING: Only {len(df[df['label'] == 'SAFE'])} SAFE samples. Need more for balanced model!")
    
    print("\n" + "="*60)


def label_specific_ranges(csv_path="dataset.csv"):
    """Label data by deflection ranges (quick labeling helper)"""
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*60)
    print("⚡ QUICK LABELING BY DEFLECTION RANGES")
    print("="*60)
    
    print("\nDefine thresholds for automatic labeling:")
    
    try:
        safe_threshold = float(input("Max deflection for SAFE (mm, e.g., 30): "))
        warning_threshold = float(input("Max deflection for WARNING (mm, e.g., 50): "))
        
        # Apply automatic labeling
        updates = 0
        for idx, row in df.iterrows():
            if row['label'] == 'UNLABELED' or row['label'] not in ['SAFE', 'WARNING', 'DANGER']:
                if row['max_deflection'] <= safe_threshold:
                    df.at[idx, 'label'] = 'SAFE'
                    updates += 1
                elif row['max_deflection'] <= warning_threshold:
                    df.at[idx, 'label'] = 'WARNING'
                    updates += 1
                else:
                    df.at[idx, 'label'] = 'DANGER'
                    updates += 1
        
        if updates > 0:
            df.to_csv(csv_path, index=False)
            print(f"\n✅ Updated {updates} samples")
            print(f"\nUpdated Class Distribution:")
            print(df['label'].value_counts())
        else:
            print("\n⚠️ No samples to update")
    
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")


if __name__ == "__main__":
    import sys
    
    print("\n🏷️  DATA LABELING TOOL")
    print("="*60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "review":
            review_dataset_stats()
        elif sys.argv[1] == "quick":
            label_specific_ranges()
        else:
            label_data()
    else:
        print("Options:")
        print("  python labeling.py         - Interactive labeling")
        print("  python labeling.py review  - Show dataset statistics")
        print("  python labeling.py quick   - Quick labeling by deflection ranges\n")
        
        choice = input("Choose option (1=interactive, 2=review, 3=quick): ").strip()
        
        if choice == "1":
            label_data()
        elif choice == "2":
            review_dataset_stats()
        elif choice == "3":
            label_specific_ranges()
        else:
            print("Invalid choice")
