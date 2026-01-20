
---

## 3️⃣ Python Code  
📄 `src/preprocessing_pipeline.py`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Dataset
# -----------------------------
# Replace with actual dataset path
data = pd.read_csv("data/hrv_raw.csv")

print("Initial Dataset Shape:", data.shape)

# -----------------------------
# Handling Missing Values
# -----------------------------
# Interpolation for continuous sensor readings
data_interpolated = data.interpolate(method='linear')

# Median imputation for any remaining missing values
data_filled = data_interpolated.fillna(data_interpolated.median())

# -----------------------------
# Outlier Detection (Isolation Forest)
# -----------------------------
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(data_filled)

# Keep only inliers
data_no_outliers = data_filled[outliers == 1]

print("Shape after outlier removal:", data_no_outliers.shape)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = PowerTransformer(method='yeo-johnson')
scaled_features = scaler.fit_transform(data_no_outliers)

scaled_df = pd.DataFrame(scaled_features, columns=data_no_outliers.columns)

# -----------------------------
# Noise Reduction (Rolling Mean)
# -----------------------------
smoothed_df = scaled_df.rolling(window=3, min_periods=1).mean()

# -----------------------------
# Dimensionality Reduction (PCA)
# -----------------------------
pca = PCA(n_components=0.95)
pca_features = pca.fit_transform(smoothed_df)

print("Reduced dimensions after PCA:", pca_features.shape[1])

# Plot PCA variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Variance Retention")
plt.savefig("outputs/pca_variance.png")

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test = train_test_split(
    pca_features,
    test_size=0.2,
    random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("Preprocessing completed successfully.")
