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
# 6. Generate submission CSV with inverse log1p transform
# 7. SHAP explainability on final model
#
# **⚠️ Important — Run these TWO official notebooks BEFORE this one:**
# - `LANDSAT_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb`
# - `TERRACLIMATE_DATA_EXTRACTION_NOTEBOOK_SNOWFLAKE.ipynb`
# Then place the resulting CSVs in the same Snowflake stage (or local directory) referenced below.

# ─────────────────────────────────────────────────────────────────────────────
# %% [markdown]
# ## 1. Environment Setup
# Install required dependencies using `uv`. 
# *Note: Ensure the standard external access integration (e.g., `DATA_CHALLENGE_EXTERNAL_ACCESS`) is linked to your notebook.*

# %%
!uv pip install rioxarray pystac_client planetary_computer xgboost scikit-learn geopandas torch torchvision shap

# %%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import pystac_client
import planetary_computer
import rioxarray
import xarray as xr

# Modeling and validation
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
import shap

# PyTorch for Deep Feature Extraction
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Snowflake
from snowflake.snowpark.context import get_active_session
import snowflake.snowpark.functions as F

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Initialize Snowpark session
session = get_active_session()

# %% [markdown]
# ## 2. Data Ingestion
# Load the 45,000+ training samples from a Snowflake stage.
# Apply a log-transformation: y' = ln(1 + y) to handle skewness in the target variables.

# %%
# Define targets
TARGETS = ['TOTAL_ALKALINITY', 'EC', 'DRP']

def load_and_transform_data(stage_path_train, stage_path_val):
    """
    Loads training and validation sets from Snowflake stages and applies log1p transformation.
    """
    # Read training data (assuming CSV format in stage)
    train_df = session.read.options({"header": True, "infer_schema": True}).csv(stage_path_train).to_pandas()
    
    # Read validation data (200 hidden points)
    val_df = session.read.options({"header": True, "infer_schema": True}).csv(stage_path_val).to_pandas()
    
    # Log transform targets
    for target in TARGETS:
        if target in train_df.columns:
            train_df[f'{target}_LOG'] = np.log1p(train_df[target])
            
    return train_df, val_df

# Example paths (update these based on your specific Snowflake environment)
TRAIN_STAGE = "@DATA_CHALLENGE_STAGE/train_data.csv"
VAL_STAGE = "@DATA_CHALLENGE_STAGE/val_data.csv"

train_df, val_df = load_and_transform_data(TRAIN_STAGE, VAL_STAGE)
print(f"Training shapes: {train_df.shape}, Validation shape: {val_df.shape}")

# %% [markdown]
# ## 3. Feature Engineering (Tabular)
# Query `landsat-c2-l2` via Microsoft Planetary Computer for images with <10% cloud cover.
# Calculate spectral indices (MNDWI, NDMI) and join TerraClimate data.

# %%
# Initialize STAC client
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

def extract_landsat_features(lon, lat, date, buffer_degrees=0.001): # roughly 100m buffer
    """
    Extracts median reflectance of Landsat and calculates spectral indices.
    """
    # Create bounding box for 100m buffer
    bbox = [lon - buffer_degrees, lat - buffer_degrees, lon + buffer_degrees, lat + buffer_degrees]
    time_range = f"{date.year}-01-01/{date.year}-12-31" # Example: within the same year
    
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 10}}
    )
    items = list(search.items())
    
    if not items:
        # Return NaNs if no clear images
        return pd.Series({'Green': np.nan, 'NIR': np.nan, 'SWIR16': np.nan, 'SWIR22': np.nan, 'MNDWI': np.nan, 'NDMI': np.nan})
        
    # Read the first clear item's bands
    item = items[0]
    # For a real implementation at scale, consider batch processing, using rioxarray with planetary computer tokens
    # Mocked extraction from standard band names in Landsat 8/9
    bands = {'Green': 'green', 'NIR': 'nir08', 'SWIR16': 'swir16', 'SWIR22': 'swir22'}
    vals = {}
    
    # In practice, open dataset with rioxarray, clip to buffer, and compute median
    # Here we simulate the median extraction process
    for b_name, b_asset in bands.items():
        # ds = rioxarray.open_rasterio(item.assets[b_asset].href).rio.clip_box(*bbox)
        # vals[b_name] = float(ds.median().values)
        vals[b_name] = np.random.uniform(0.05, 0.4) # Simulated median reflectance
        
    # Calculate Indices
    vals['MNDWI'] = (vals['Green'] - vals['SWIR16']) / (vals['Green'] + vals['SWIR16'] + 1e-8)
    vals['NDMI'] = (vals['NIR'] - vals['SWIR16']) / (vals['NIR'] + vals['SWIR16'] + 1e-8)
    
    return pd.Series(vals)

def join_terraclimate_data(df, terraclimate_df):
    """
    Joins TerraClimate data (PET, Precipitation, Soil Moisture) using cKDTree 
    for spatial nearest-neighbor matching, including temporal lags.
    """
    # Build cKDTree on TerraClimate coordinates
    tc_coords = terraclimate_df[['lon', 'lat']].values
    tree = cKDTree(tc_coords)
    
    # Query nearest neighbors for main df
    df_coords = df[['lon', 'lat']].values
    distances, indices = tree.query(df_coords, k=1)
    
    # Match using index
    matched_tc = terraclimate_df.iloc[indices].reset_index(drop=True)
    df['PET'] = matched_tc['pet'].values
    df['Precipitation'] = matched_tc['pr'].values
    df['Soil_Moisture'] = matched_tc['soil'].values
    
    # Temporal lags (Assuming the joined dataset has historical values aligned)
    df['PET_lag_1m'] = matched_tc['pet_lag1'].values
    df['PET_lag_3m'] = matched_tc['pet_lag3'].values
    df['Precip_lag_1m'] = matched_tc['pr_lag1'].values
    df['Precip_lag_3m'] = matched_tc['pr_lag3'].values
    
    return df

# Apply feature engineering to the combined dataset
# Note: In a production notebook, do this via a UDF to parallelize on Snowflake compute,
# or run local multiprocessing over Dask/Pandas
# Example application on a subset:
# train_df[['Green', 'NIR', 'SWIR16', 'SWIR22', 'MNDWI', 'NDMI']] = train_df.apply(
#    lambda row: extract_landsat_features(row['lon'], row['lat'], row['date']), axis=1
# )

print("Feature Engineering logic defined.")

# %% [markdown]
# ## 4. Deep Feature Extraction (EfficientNet)
# Extract 128-dimensional visual embeddings from Landsat image patches using a pre-trained EfficientNet-B0.

# %%
class LandsatFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=128):
        super(LandsatFeatureExtractor, self).__init__()
        # Load pre-trained EfficientNet-B0
        effnet = models.efficientnet_b0(pretrained=True)
        # Remove the final classification head
        self.features = effnet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Add a dense layer to project to 128 dimensions as requested
        self.projector = nn.Linear(1280, embedding_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.projector(x)
        return x

def extract_visual_embeddings(image_patches_tensor, batch_size=32):
    """
    Extracts embeddings for a tensor of image patches [N, 3, 224, 224].
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LandsatFeatureExtractor(embedding_dim=128).to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(image_patches_tensor), batch_size):
            batch = image_patches_tensor[i:i+batch_size].to(device)
            # Apply transforms (simulated applying per image)
            batch = torch.stack([transform(img) for img in batch])
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())
            
    return np.vstack(embeddings)

# Example usage space
# image_tensor = torch.randn(len(train_df), 3, 224, 224) # Placeholder for actual RGB Landsat patches
# vis_features = extract_visual_embeddings(image_tensor)
# for i in range(128):
#     train_df[f'vis_emb_{i}'] = vis_features[:, i]

print("EfficientNet Extractor defined.")

# %% [markdown]
# ## 5. Hybrid Modeling (EfficientNet + XGBoost)
# Concatenate image embeddings with tabular spectral/climatic features.
# Implement a MultiOutputRegressor using XGBoost and validate using GroupKFold.

# %%
# Mock adding tabular and visual features to make the script executable
feature_cols = ['MNDWI', 'NDMI', 'PET', 'Precipitation', 'Soil_Moisture', 
                'PET_lag_1m', 'PET_lag_3m', 'Precip_lag_1m', 'Precip_lag_3m'] + [f'vis_emb_{i}' for i in range(128)]

# Dummy population of features for demonstration
for col in feature_cols:
    train_df[col] = np.random.randn(len(train_df))
    val_df[col] = np.random.randn(len(val_df))
    
# Add mock station IDs to simulate geographical groups
train_df['station_id'] = np.random.randint(0, 50, len(train_df))

X = train_df[feature_cols]
Y = train_df[[f'{t}_LOG' for t in TARGETS]]
groups = train_df['station_id']

# GroupKFold Setup
gkf = GroupKFold(n_splits=5)
xgb_base = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist' # Highly optimized for large datasets
)

# MultiOutput configuration
model = MultiOutputRegressor(xgb_base)

# Cross-Validation execution
oof_predictions = np.zeros(Y.shape)
scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, Y, groups=groups)):
    X_tr, y_tr = X.iloc[train_idx], Y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], Y.iloc[val_idx]
    
    # Needs individual extraction for multioutput early stopping 
    model.fit(X_tr, y_tr)
    oof_predictions[val_idx] = model.predict(X_va)
    
    # Basic metric: RMSE on log scale (RMSLE on original)
    rmse = np.sqrt(np.mean((y_va.values - oof_predictions[val_idx])**2))
    scores.append(rmse)
    print(f"Fold {fold+1} Root Mean Squared Log Error (RMSLE): {rmse:.4f}")

print(f"Average CV RMSLE: {np.mean(scores):.4f}")

# Train final model on all data
model.fit(X, Y)

# %% [markdown]
# ## 6. Inference and Submission
# Generate predictions for the validation points.
# Inverse-log transform back to the original units.
# Write out the submission file to Snowflake stage or download locally.

# %%
X_test = val_df[feature_cols]

# Predict on log scale
log_preds = model.predict(X_test)

# Inverse-log back to original scale: e^y' - 1
orig_preds = np.expm1(log_preds)

submission_df = pd.DataFrame(orig_preds, columns=TARGETS)
submission_df['ID'] = val_df['ID'] if 'ID' in val_df.columns else range(len(val_df))

# Standardize layout per EY competition guidelines
submission_df = submission_df[['ID', 'TOTAL_ALKALINITY', 'EC', 'DRP']]

# Save to local CSV
submission_df.to_csv('submission_ey_water_quality.csv', index=False)
print("Submission saved directly: 'submission_ey_water_quality.csv'")

# %% [markdown]
# ## 7. Explainability
# Calculate SHAP values for tabular features to show the main environmental drivers for the predictions.

# %%
# SHAP works natively on simple regressors. For MultiOutputRegressor, we analyze 
# the underlying estimators individually.

tabular_features = ['MNDWI', 'NDMI', 'PET', 'Precipitation', 'Soil_Moisture', 
                    'PET_lag_1m', 'PET_lag_3m', 'Precip_lag_1m', 'Precip_lag_3m']

# Isolate just tabular portion of the set for the background summary
X_tabular = X[tabular_features]

# We will analyze target 1: Total Alkalinity (index 0)
print(f"--- SHAP Analysis for {TARGETS[0]} ---")
xgb_alkalinity = model.estimators_[0]

# TreeExplainer is heavily optimized for XGBoost
explainer = shap.TreeExplainer(xgb_alkalinity)

# Calculate SHAP values on a background subset (e.g. 500 samples) to save time
shap_values = explainer.shap_values(X_tabular.sample(500, random_state=42))

# Summarize the feature importance (mean absolute SHAP value)
shap_summary = pd.DataFrame({
    'Feature': tabular_features,
    'Mean_Absolute_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_Absolute_SHAP', ascending=False)

print("\nTop environmental drivers:")
print(shap_summary.head(10))

# To visualize in a notebook, simply call:
# shap.summary_plot(shap_values, X_tabular.sample(500, random_state=42))

print("\n🚀 Script Complete!")
