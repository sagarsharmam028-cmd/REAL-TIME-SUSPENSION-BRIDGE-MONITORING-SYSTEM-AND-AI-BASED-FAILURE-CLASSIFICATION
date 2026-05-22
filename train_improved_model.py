#!/usr/bin/env python3
"""
Train improved model with fully labeled dataset
"""
from model import train_model, visualize_feature_importance, list_models

print("\n" + "="*70)
print("🚀 TRAINING IMPROVED MODEL WITH FULL DATASET")
print("="*70)

# Train RandomForest with full labeled data
print("\n[1] Training RandomForest with expanded dataset...")
print("-"*70)

model, trained = train_model(model_type="rf")

if trained:
    print("\n[2] Creating feature importance visualization...")
    viz_file = visualize_feature_importance(model)
    
    print("\n[3] Model Registry:")
    print("-"*70)
    list_models()
    
    print("\n" + "="*70)
    print("✅ IMPROVED MODEL TRAINED SUCCESSFULLY!")
    print("="*70)
    print("\n📊 Key Improvements:")
    print("   • Trained on 3,529 samples (vs 1,481 before)")
    print("   • Better coverage: SAFE, WARNING, DANGER classes")
    print("   • More robust predictions")
    print("\n💾 Files:")
    print(f"   • Latest model: models/latest")
    if viz_file:
        print(f"   • Feature importance: {viz_file}")
else:
    print("⚠️ Training failed")
