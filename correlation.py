import joblib
import xarray as xr
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Load datasets
ds_ndvi = xr.open_dataset('sample_data/wekeo-ndvi/train.nc4')
ds_ducmass = xr.open_dataset('sample_data/nasa-ducmass/train.nc4')

# Convert datasets to dataframes
df_ducmass = (ds_ducmass
      .to_dataframe()
      .reset_index()
      .loc[:,['year_month', 'lat', 'lon', 'DUCMASS', 'RDUCMASS', 'LDUCMASS', 'TDUCMASS', 'BDUCMASS', 'OLD_DUCMASS']]
      )

df_ndvi = (ds_ndvi
      .to_dataframe()
      .reset_index()
      .loc[:,['year_month', 'lat', 'lon', 'NDVI', 'RNDVI', 'LNDVI', 'TNDVI', 'BNDVI', 'OLD_NDVI']]
      .interpolate(method='linear', limit_direction='both', subset=['NDVI'])  # Interpolate NaN values in NDVI
      )

# Trim dataframes to have the same length
min_length = min(len(df_ducmass), len(df_ndvi))
df_ducmass_trimmed = df_ducmass.iloc[:min_length]
df_ndvi_trimmed = df_ndvi.iloc[:min_length]

# Extract features and target variables
x_ndvi = df_ndvi_trimmed[['year_month', 'lat', 'lon', 'RNDVI', 'LNDVI', 'TNDVI', 'BNDVI', 'OLD_NDVI']]
y_ndvi = df_ndvi_trimmed['NDVI']
x_ducmass = df_ducmass_trimmed[['year_month', 'lat', 'lon', 'RDUCMASS', 'LDUCMASS', 'TDUCMASS', 'BDUCMASS', 'OLD_DUCMASS']]
y_ducmass = df_ducmass_trimmed['DUCMASS']

# Split the data into training and testing sets
X_train_ndvi, X_test_ndvi, y_train_ndvi, y_test_ndvi = train_test_split(x_ndvi, y_ndvi, test_size=0.2, shuffle=False)
X_train_ducmass, X_test_ducmass, y_train_ducmass, y_test_ducmass = train_test_split(x_ducmass, y_ducmass, test_size=0.2, shuffle=False)

# Load models
rf_ndvi = joblib.load('wekeo_ndvi.plk')
rf_ducmass = joblib.load('nasa-ducmass.plk')

# Make predictions
y_pred_ndvi = rf_ndvi.predict(X_train_ndvi)
y_pred_ducmass = rf_ducmass.predict(X_train_ducmass)

# Calculate correlation
corr_train, p_value_train = pearsonr(y_ducmass, y_ndvi)
print(f'Коефіцієнт кореляції: {corr_train}')
corr_percent = corr_train * 100
print(f'Коефіцієнт кореляції у відсотках: {corr_percent:.2f}%')

corr_pred, p_value_pred = pearsonr(y_pred_ndvi, y_pred_ducmass)
print(f'Коефіцієнт кореляції: {corr_pred}')
corr_percent = corr_pred * 100
print(f'Коефіцієнт кореляції у відсотках: {corr_percent:.2f}%')
