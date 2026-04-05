import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def train_progressive(X, y, look_back, look_ahead, retrain_step, depthOfTree):
    """
    Progressive (rolling) model:
    - Trains on the first look_back rows
    - Retrains every retrain_step rows
    - Predicts look_ahead rows at each step
    """
    errors = []
    n_rows = X.shape[0]
    
    # Initialize model and first training window
    model = DecisionTreeRegressor(max_depth=depthOfTree)
    X_train = X.iloc[:look_back]
    y_train = y.iloc[:look_back]
    model.fit(X_train, y_train)
    
    # Start predicting from the end of initial training
    start_idx = look_back
    
    while start_idx + look_ahead <= n_rows:
        # Slice test window
        X_test = X.iloc[start_idx:start_idx + look_ahead]
        y_test = y.iloc[start_idx:start_idx + look_ahead]
        
        # Predict
        predictions = model.predict(X_test)
        
        # Compute error
        error = np.sqrt(mean_squared_error(y_test, predictions))
        errors.append(error)
        
        # Move to next step
        start_idx += look_ahead
        
        # Retrain if passed a retrain_step
        if start_idx % retrain_step == 0:
            # Use most recent look_back rows for retraining
            train_start = max(0, start_idx - look_back)
            X_train = X.iloc[train_start:start_idx]
            y_train = y.iloc[train_start:start_idx]
            model.fit(X_train, y_train)
    
    # Average RMSE across all predictions
    avg_error = np.mean(errors)
    return avg_error