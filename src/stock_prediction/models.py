import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def prepare_features(self, data):
        """Prepare features for ML model"""
        df = pd.DataFrame(data)
        
        # Convert to numeric
        numeric_cols = ['close', 'volume', 'ma_10', 'ma_30', 'rsi']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create lag features
        df['close_lag1'] = df['close'].shift(1)
        df['close_lag2'] = df['close'].shift(2)
        df['volume_ma'] = df['volume'].rolling(window=5).mean()
        
        # Price change features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=10).std()
        
        # Drop NaN and create feature matrix
        df = df.dropna()
        
        feature_cols = ['close_lag1', 'close_lag2', 'volume', 'ma_10', 'ma_30', 'rsi', 'volume_ma', 'volatility']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].values
        y = df['close'].values
        
        return X, y, available_cols
    
    def train(self, data):
        """Train the prediction model"""
        X, y, feature_cols = self.prepare_features(data)
        
        if len(X) < 10:
            return {"error": "Not enough data for training"}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        return {
            "mae": mae,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "training_samples": len(X),
            "features_used": feature_cols
        }
    
    def predict(self, data):
        """Make predictions"""
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        X, _, _ = self.prepare_features(data)
        if len(X) == 0:
            return {"error": "No valid data for prediction"}
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions[-10:].tolist()  # Return last 10 predictions
