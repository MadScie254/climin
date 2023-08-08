from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Create an SVR model
svr_model = SVR()

# Define the hyperparameter grid for tuning
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2]
}

# Perform grid search to find the best hyperparameters using cross-validation
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best SVR model from the grid search
best_svr_model = grid_search.best_estimator_

# Predict the GDP growth on the testing data
y_pred = best_svr_model.predict(X_test)

# Evaluate the model's performance using Mean Squared Error (MSE) and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Support Vector Machine Regression (SVR) Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print("Best Hyperparameters:")
print(grid_search.best_params_)
