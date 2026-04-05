import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def train_static(X, y, look_back, look_ahead, depthOfTree):
    """
    Static model with look-ahead:
    - Trains on the first look_back rows
    - Predicts the next look_ahead rows
    - Returns predictions and RMSE
    """

    # Training window
    X_train = X.iloc[:look_back]
    y_train = y.iloc[:look_back]

    # Define prediction window
    start = look_back
    end = len(X)-look_back

    # Handle edge case (avoid going out of bounds)
    if end > len(X):
        end = len(X)

    X_test = X.iloc[start:end]
    y_test = y.iloc[start:end]

    # Train model
    model = DecisionTreeRegressor(max_depth=depthOfTree)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Compute RMSE
    avg_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return avg_rmse