import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df = pd.read_csv("Chronic_Kidney_Dsease_data_Regression.csv")

X = df.drop(columns=['HbA1c'])
y = df['HbA1c']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gbr = GradientBoostingRegressor(
    n_estimators=250,
    learning_rate=0.250,
    max_depth=3,
    random_state=42
)

gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Gradient Boosting R²: {r2:.4f}")
print(f"Gradient Boosting RMSE: {rmse:.4f}")
