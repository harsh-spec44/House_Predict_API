# Readymade dataset for training a model, contains data about California housing prices and features.
from sklearn.datasets import fetch_california_housing

# Provides different types of high performance models for regression and classification tasks.
from sklearn.ensemble import RandomForestRegressor

# Prepare data before train, split into two parts, one for training and one for testing the model.
# Split arrays or matrices into random train and test subsets.
from sklearn.model_selection import train_test_split

# Measure  accuracy of model answer 1) average absolute difference between predicted and actual values, 
# 2) average squared difference between predicted and actual values, 3) proportion of variance explained by the model.
# Evaluate the performance of a model by calculating the mean absolute error, mean squared error, and R-squared score.
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

# Save file in hardware
import joblib

print("loading datasets")
data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"total records: {X.shape[0]}")

X_train,X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42) 
# test size means 20% data will go for testing and rest for training

#training a model :-
model = RandomForestRegressor (
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"average error: ${mae*10000:,.0f}")

joblib.dump(model, "house_model.joblib")
joblib.dump(list(X.columns), "house_features.joblib")