import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def train_static(X, y, look_back, depthOfTree):
    """
    Static model:
    - Trains on the first look_back rows
    - Predicts all remaining rows
    - Returns predictions and average RMSE
    """
    # Training window
    X_train = X.iloc[:look_back]
    y_train = y.iloc[:look_back]

    # Test window (everything left)
    X_test = X.iloc[look_back:]
    y_test = y.iloc[look_back:]

    # Train model
    model = DecisionTreeRegressor(max_depth=depthOfTree)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Compute RMSE
    avg_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return predictions, avg_rmse