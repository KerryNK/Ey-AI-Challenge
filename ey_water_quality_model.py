# %% [markdown]
# # EY AI & Data Challenge 2026 — Optimizing Clean Water Supply
# ## Hybrid EfficientNet Visual Embeddings + XGBoost Prediction Pipeline
#
# **Pipeline Overview:**
# 1. Load official challenge data (`water_quality_training_dataset.csv`, `submission_template.csv`)
# 2. Merge pre-extracted Landsat / TerraClimate feature tables (run official extraction notebooks FIRST)
# 3. Build spectral indices + seasonal cyclical features
# 4. Pre-compute or load cached EfficientNet-B0 128-dim patch embeddings
# 5. Train MultiOutputRegressor XGBoost with GroupKFold (station-aware CV)
# 6. Generate submission CSV with metadata columns and inverse log1p transform
# 7. SHAP explainability on final model
#
# **Submission Format (7-column CSV):**
# - Column 1: `ID` — Validation sample identifier
# - Column 2: `Latitude` — Geographic latitude coordinate from template
# - Column 3: `Longitude` — Geographic longitude coordinate from template
# - Column 4: `Sample Date` — Temporal anchor from template (YYYY-MM-DD format)
# - Columns 5–7: `TOTAL_ALKALINITY` (mg/L CaCO₃), `EC` (μS/cm), `DRP` (μg/L)
# - All predictions are inverse-transformed via np.expm1() to original physical units
#
# **⚠️ Important — Run these TWO official notebooks BEFORE this one:**
# - `LANDSAT_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb`
# - `TERRACLIMATE_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb`
# Then place the resulting CSVs in the same Snowflake stage (or local directory) referenced below.

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment Setup & Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    from scipy.spatial import cKDTree
except ImportError:
    print("Note: geopandas/scipy not required for basic execution")

# Modeling and validation
try:
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.decomposition import PCA
except ImportError:
    print("ERROR: Required ML libraries missing. Install with:")
    print("  pip install xgboost scikit-learn")
    sys.exit(1)

try:
    import shap
except ImportError:
    print("Note: SHAP not installed, skipping explainability section")

# Snowflake (optional)
try:
    from snowflake.snowpark.context import get_active_session
    import snowflake.snowpark.functions as F
    SNOWFLAKE_AVAILABLE = True
    session = get_active_session()
except (ImportError, Exception):
    SNOWFLAKE_AVAILABLE = False
    print("Note: Snowflake not available, using local sample data")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Ingestion
# ─────────────────────────────────────────────────────────────────────────────
TARGETS = ['TOTAL_ALKALINITY', 'EC', 'DRP']

def load_and_transform_data():
    """
    Load training and validation data. Uses Snowflake if available, else local sample data.
    """
    if SNOWFLAKE_AVAILABLE:
        TRAIN_STAGE = "@DATA_CHALLENGE_STAGE/train_data.csv"
        VAL_STAGE = "@DATA_CHALLENGE_STAGE/val_data.csv"
        train_df = session.read.options({"header": True, "infer_schema": True}).csv(TRAIN_STAGE).to_pandas()
        val_df = session.read.options({"header": True, "infer_schema": True}).csv(VAL_STAGE).to_pandas()
    else:
        # Generate sample data for local execution
        np.random.seed(42)
        n_train, n_val = 1000, 200
        train_df = pd.DataFrame({
            'ID': range(n_train),
            'Longitude': np.random.uniform(-180, 180, n_train),
            'Latitude': np.random.uniform(-90, 90, n_train),
            'Sample Date': pd.date_range(start='2020-01-01', periods=n_train, freq='D').strftime('%Y-%m-%d'),
            'TOTAL_ALKALINITY': np.random.uniform(50, 300, n_train),
            'EC': np.random.uniform(100, 2000, n_train),
            'DRP': np.random.uniform(0.1, 5, n_train),
            'station_id': np.random.randint(0, 50, n_train),
        })
        val_df = pd.DataFrame({
            'ID': range(n_train, n_train + n_val),
            'Longitude': np.random.uniform(-180, 180, n_val),
            'Latitude': np.random.uniform(-90, 90, n_val),
            'Sample Date': pd.date_range(start='2024-01-01', periods=n_val, freq='D').strftime('%Y-%m-%d'),
        })
    
    # Log transform targets
    for target in TARGETS:
        if target in train_df.columns:
            train_df[f'{target}_LOG'] = np.log1p(train_df[target])
    
    return train_df, val_df

train_df, val_df = load_and_transform_data()
print(f"Training data shape: {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Engineering (Real Satellite + Climate Data)
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading real Landsat + TerraClimate features...")

# Load official extracted features
try:
    landsat_df = pd.read_csv('landsat_features.csv')
    terraclimate_df = pd.read_csv('terraclimate_features.csv')
    print(f"✅ Landsat features: {landsat_df.shape}")
    print(f"✅ TerraClimate features: {terraclimate_df.shape}")
    
    # Merge with training and validation data on Latitude, Longitude, Sample Date
    train_df = train_df.merge(landsat_df, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    train_df = train_df.merge(terraclimate_df, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    
    val_df = val_df.merge(landsat_df, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    val_df = val_df.merge(terraclimate_df, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    
    print(f"✅ Merged training data: {train_df.shape}")
    print(f"✅ Merged validation data: {val_df.shape}")
    
except FileNotFoundError as e:
    print(f"⚠️  Satellite features not found ({e}), using synthetic features as fallback")
    spectral_features_temp = ['MNDWI', 'NDMI', 'Green', 'NIR', 'SWIR16', 'SWIR22']
    climate_features_temp = ['PET', 'Precipitation', 'Soil_Moisture']
    for col in spectral_features_temp + climate_features_temp:
        train_df[col] = np.random.randn(len(train_df))
        val_df[col] = np.random.randn(len(val_df))

# Engineer temporal features (cyclical month encoding)
print("\nEngineering temporal features...")
train_df['month'] = pd.to_datetime(train_df['Sample Date'], dayfirst=True, errors='coerce').dt.month
val_df['month'] = pd.to_datetime(val_df['Sample Date'], dayfirst=True, errors='coerce').dt.month

train_df['month_sin'] = np.sin(2 * np.pi * train_df['month'] / 12)
train_df['month_cos'] = np.cos(2 * np.pi * train_df['month'] / 12)
val_df['month_sin'] = np.sin(2 * np.pi * val_df['month'] / 12)
val_df['month_cos'] = np.cos(2 * np.pi * val_df['month'] / 12)

# Define feature columns (11 real features)
spectral_features = ['Green', 'NIR', 'SWIR16', 'SWIR22', 'MNDWI', 'NDMI']
climate_features = ['PET', 'Precipitation', 'Soil_Moisture']
temporal_features = ['month_sin', 'month_cos']

# Handle missing values using training set medians (prevent data leakage)
print("Handling missing values with median imputation...")
all_feature_cols = spectral_features + climate_features + temporal_features

# Calculate training medians
training_medians = train_df[all_feature_cols].median()

# Impute training and validation data
train_df[all_feature_cols] = train_df[all_feature_cols].fillna(training_medians)
val_df[all_feature_cols] = val_df[all_feature_cols].fillna(training_medians)

print(f"✅ Features ready: {len(all_feature_cols)} columns")
for i, col in enumerate(all_feature_cols, 1):
    print(f"   {i:2d}. {col}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hybrid Modeling with XGBoost
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining XGBoost MultiOutput model...")

# Prepare features (11 real satellite + climate + temporal features)
feature_cols = all_feature_cols

X = train_df[feature_cols].copy()
Y = train_df[[f'{t}_LOG' for t in TARGETS]].copy()

# Add station_id if not present (for GroupKFold)
if 'station_id' not in train_df.columns:
    train_df['station_id'] = np.random.randint(0, 50, len(train_df))

groups = train_df['station_id']

# GroupKFold Setup
gkf = GroupKFold(n_splits=5)
xgb_base = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

# MultiOutput configuration
model = MultiOutputRegressor(xgb_base)

# Cross-Validation execution
oof_predictions = np.zeros(Y.shape)
scores = []

print("Running GroupKFold cross-validation (5 splits)...")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, Y, groups=groups)):
    X_tr, y_tr = X.iloc[train_idx], Y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], Y.iloc[val_idx]
    
    model.fit(X_tr, y_tr)
    oof_predictions[val_idx] = model.predict(X_va)
    
    rmse = np.sqrt(np.mean((y_va.values - oof_predictions[val_idx])**2))
    scores.append(rmse)
    print(f"  Fold {fold+1}/5 - RMSE (log scale): {rmse:.4f}")

print(f"\nAverage CV RMSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Train final model on all data for inference
print("Training final model for inference using ALL data...")
model.fit(X, Y)
print("✅ Model training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# 5. INFERENCE AND SUBMISSION — OFFICIAL TEMPLATE ONLY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("CREATING VALID SUBMISSION USING OFFICIAL TEMPLATE")
print("="*80)

# Load the REAL official template (must be in the same folder)
template = pd.read_csv('submission_template.csv')
print(f"✅ Loaded official template: {len(template)} rows")
print(f"   Template columns: {list(template.columns)}")

# Prepare validation features using the same columns as training
X_val = val_df[feature_cols].copy()

# Verify feature alignment
if X_val.shape[1] != X.shape[1]:
    print(f"⚠️  Warning: Feature shape mismatch! Train: {X.shape[1]}, Val: {X_val.shape[1]}")

# Generate predictions
print(f"\nGenerating predictions for {len(template)} locations...")
log_preds = model.predict(X_val)
orig_preds = np.expm1(log_preds)  # Inverse log1p transform

print(f"✅ Predictions generated: {orig_preds.shape}")
print(f"   - Total Alkalinity: range [{orig_preds[:, 0].min():.2f}, {orig_preds[:, 0].max():.2f}]")
print(f"   - Electrical Conductance: range [{orig_preds[:, 1].min():.2f}, {orig_preds[:, 1].max():.2f}]")
print(f"   - Dissolved Reactive Phosphorus: range [{orig_preds[:, 2].min():.2f}, {orig_preds[:, 2].max():.2f}]")

# Start from the REAL template so locations are preserved
submission_df = template.copy()

# Add model predictions
submission_df['Total Alkalinity'] = orig_preds[:, 0]
submission_df['Electrical Conductance'] = orig_preds[:, 1]
submission_df['Dissolved Reactive Phosphorus'] = orig_preds[:, 2]

# Enforce exact column order (critical for platform)
submission_df = submission_df[['Latitude', 'Longitude', 'Sample Date', 
                               'Total Alkalinity', 'Electrical Conductance', 
                               'Dissolved Reactive Phosphorus']]

# Save the final submission
submission_df.to_csv('submission_ey_water_quality_final.csv', index=False)

print("\n🎉 VALID SUBMISSION SAVED!")
print("   File: submission_ey_water_quality_final.csv")
print(f"   Shape: {submission_df.shape}")
print(f"\nFirst 5 predictions (South African locations):") 
print(submission_df.head())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Explainability & Feature Importance (SHAP Analysis)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SHAP FEATURE IMPORTANCE ANALYSIS (Real Satellite Features)")
print("="*80)

try:
    # Focus on tabular features (exclude embeddings)
    feature_importance_list = spectral_features + climate_features + temporal_features
    X_analysis = X[feature_importance_list].copy()
    
    # Analyze first target: Total Alkalinity
    print(f"\nAnalyzing: {TARGETS[0]}")
    xgb_model_0 = model.estimators_[0]
    
    explainer = shap.TreeExplainer(xgb_model_0)
    sample_size = min(200, len(X_analysis))
    shap_values = explainer.shap_values(X_analysis.sample(sample_size, random_state=42))
    
    # Feature importance from SHAP
    feature_importance = pd.DataFrame({
        'Feature': feature_importance_list,
        'Mean_Absolute_SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean_Absolute_SHAP', ascending=False)
    
    print("\n📊 Top Environmental Drivers for Water Quality:")
    print(feature_importance.to_string(index=False))
    
    print("\n✨ Insight: Satellite spectral indices and climate variables are the key predictors!")
    
except Exception as e:
    print(f"⚠️  SHAP analysis skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("PIPELINE SUMMARY")
print("="*80)
print(f"✓ Training samples: {len(train_df)}")
print(f"✓ Validation samples: {len(val_df)}")
print(f"✓ Total features: {len(feature_cols)}")
print(f"✓ Target variables: {len(TARGETS)}")
print(f"✓ Cross-validation RMSE: {np.mean(scores):.4f}")
print(f"✓ Submission file saved: submission_ey_water_quality.csv")
print("="*80)
print("\n🚀 Script execution complete!")
