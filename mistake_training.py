import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
def train_mistake_driven(X, y, look_back, look_ahead, depthOfTree, threshold):
    """
    Mistake-Driven Decision Tree:
    - Trains on the first look_back rows
    - Predicts in chunks of look_ahead
    - Retrains ONLY if RMSE of last prediction > threshold
    """
    errors = []
    predictions_list = []

    n_rows = X.shape[0]
    
    # Initialize model with initial training window
    model = DecisionTreeRegressor(max_depth=depthOfTree)
    X_train = X.iloc[:look_back]
    y_train = y.iloc[:look_back]
    model.fit(X_train, y_train)
    
    start_idx = look_back
    
    while start_idx + look_ahead <= n_rows:
        # Slice test window
        X_test = X.iloc[start_idx:start_idx + look_ahead]
        y_test = y.iloc[start_idx:start_idx + look_ahead]
        
        # Predict
        preds = model.predict(X_test)
        predictions_list.extend(preds)
        
        # Compute RMSE for this window
        error = np.sqrt(mean_squared_error(y_test, preds))
        errors.append(error)
        
        # Retrain if error exceeds threshold
        if error > threshold:
            train_start = max(0, start_idx - look_back)
            X_train = X.iloc[train_start:start_idx]
            y_train = y.iloc[train_start:start_idx]
            model.fit(X_train, y_train)
        
        # Move forward
        start_idx += look_ahead
    
    avg_error = np.mean(errors)
    return predictions_list, avg_error