#!/usr/bin/env python3
"""
Comprehensive test of new features:
- XGBoost training
- Feature importance visualization
- Model comparison
- Dataset statistics
"""

from model import (
    train_model, train_xgboost, load_model, list_models, 
    visualize_feature_importance, compare_models
)
from labeling import review_dataset_stats
import os

print("\n" + "="*70)
print("🚀 BRIDGE MONITORING SYSTEM - FEATURE TEST")
print("="*70)

# Test 1: Review dataset
print("\n[1/4] 📊 Reviewing dataset statistics...")
print("-"*70)
review_dataset_stats()

# Test 2: Train RandomForest
print("\n[2/4] 🎯 Training RandomForest model...")
print("-"*70)
rf_model, rf_trained = train_model(model_type="rf")

if rf_trained:
    # Visualize RF feature importance
    print("\n   Creating feature importance visualization...")
    viz_file = visualize_feature_importance(rf_model)
    if viz_file:
        print(f"   ✅ Visualization saved!")
else:
    print("   ⚠️ RandomForest training failed")

# Test 3: Try XGBoost
print("\n[3/4] 🚀 Attempting XGBoost training...")
print("-"*70)
try:
    xgb_model, xgb_trained = train_xgboost()
    
    if xgb_trained:
        print("   ✅ XGBoost training successful!")
        # Visualize XGBoost feature importance
        print("   Creating feature importance visualization...")
        viz_file = visualize_feature_importance(xgb_model)
        if viz_file:
            print(f"   ✅ XGBoost visualization saved!")
    else:
        print("   ⚠️ XGBoost training failed")
except ImportError:
    print("   ℹ️  XGBoost not installed. Install with: pip install xgboost")

# Test 4: List all models
print("\n[4/4] 📦 Model Registry...")
print("-"*70)
list_models()

# Summary
print("\n" + "="*70)
print("✅ FEATURE TEST COMPLETE")
print("="*70)
print("\n📁 Generated Files:")
if os.path.exists("visualizations"):
    viz_files = os.listdir("visualizations")
    for f in viz_files[-3:]:  # Show last 3
        print(f"   • visualizations/{f}")

print("\n📚 Next Steps:")
print("   1. Review feature importance plots in visualizations/ folder")
print("   2. Label more data: python labeling.py")
print("   3. Retrain model: python train_production_model.py")
print("   4. Monitor live: python main.py (requires serial connection)")

print("\n" + "="*70)
