import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, \
      mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import joblib

ds = xr.open_dataset('sample_data/nasa-ducmass/train.nc4')

df = (ds
      # convert to dataframe
      .to_dataframe()
      # convert time and lon/lat to columns
      .reset_index()
      # only select what you want, in case there are bnds etc. in the data
      .loc[:,['year_month', 'lat', 'lon', 'DUCMASS', 'RDUCMASS', 'LDUCMASS', 'TDUCMASS', 'BDUCMASS', 'OLD_DUCMASS']]
      )

# Prepare the data for Random Forest Regression
X = df[['year_month', 'lat', 'lon', 'RDUCMASS', 'LDUCMASS', 'TDUCMASS', 'BDUCMASS', 'OLD_DUCMASS']]
y = df['DUCMASS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

rf = RandomForestRegressor(n_estimators=100, max_depth=20)
rf.fit(X_train, y_train)

joblib.dump(rf, 'nasa_ducmass.plk')

# rf = joblib.load('nasa_ducmass.plk')

y_pred = rf.predict(X_train)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
ev = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Explained Variance (EV): {ev}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")