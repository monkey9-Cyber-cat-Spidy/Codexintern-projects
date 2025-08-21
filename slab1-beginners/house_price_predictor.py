importportport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = 'price'
        
    def load_data(self, file_path=None):
        """Load house price dataset"""
        if file_path and Path(file_path).exists():
            print(f"Loading data from {file_path}")
            self.df = pd.read_csv(file_path)
        else:
            print("Creating synthetic house price dataset...")
            self.df = self.create_synthetic_dataset()
            self.df.to_csv('house_prices.csv', index=False)
            print("✓ Synthetic dataset saved as 'house_prices.csv'")
        
        print(f"Dataset loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def create_synthetic_dataset(self, n_samples=1000):
        """Create a realistic synthetic house price dataset"""
        np.random.seed(42)
        
        # Location factors
        locations = ['Downtown', 'Suburb', 'Rural', 'Waterfront', 'Urban']
        location_multipliers = {'Downtown': 1.5, 'Suburb': 1.2, 'Rural': 0.8, 'Waterfront': 2.0, 'Urban': 1.3}
        
        # Generate features
        data = {
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'sqft_living': np.random.randint(800, 4000, n_samples),
            'sqft_lot': np.random.randint(2000, 15000, n_samples),
            'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
            'age': np.random.randint(0, 50, n_samples),
            'location': np.random.choice(locations, n_samples),
            'garage': np.random.randint(0, 4, n_samples),
            'pool': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'fireplace': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic price based on features
        base_price = 100000
        
        # Price calculation with realistic relationships
        price = (
            base_price +
            df['bedrooms'] * 15000 +
            df['bathrooms'] * 12000 +
            df['sqft_living'] * 150 +
            df['sqft_lot'] * 5 +
            df['floors'] * 8000 +
            (50 - df['age']) * 1000 +  # Newer houses cost more
            df['garage'] * 10000 +
            df['pool'] * 25000 +
            df['fireplace'] * 8000
        )
        
        # Apply location multipliers
        for i, location in enumerate(df['location']):
            price.iloc[i] *= location_multipliers[location]
        
        # Add some noise and ensure positive prices
        price += np.random.normal(0, 20000, n_samples)
        price = np.maximum(price, 50000)  # Minimum price
        
        df['price'] = price.round(0).astype(int)
        
        return df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic info
        print("\nDataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        print(f"\nTarget variable ({self.target_name}) statistics:")
        target_stats = self.df[self.target_name].describe()
        print(target_stats)
        
        # Correlation analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        print(f"\nTop correlations with {self.target_name}:")
        price_corr = correlation_matrix[self.target_name].abs().sort_values(ascending=False)
        for feature, corr in price_corr.head(6).items():  # Top 5 + target itself
            if feature != self.target_name:
                print(f"  {feature}: {corr:.3f}")
        
        return correlation_matrix
    
    def visualize_data(self, save_plots=True):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        if save_plots:
            Path('plots').mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)
        
        # 1. Price distribution
        plt.figure(figsize=fig_size)
        plt.hist(self.df[self.target_name], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('House Price Distribution', fontsize=16)
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        if save_plots:
            plt.savefig('plots/price_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=fig_size)
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        if save_plots:
            plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Feature relationships with price
        important_features = ['sqft_living', 'bedrooms', 'bathrooms', 'age']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(important_features):
            axes[i].scatter(self.df[feature], self.df[self.target_name], alpha=0.6, color='coral')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Price ($)')
            axes[i].set_title(f'Price vs {feature}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('plots/feature_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Price by location
        plt.figure(figsize=fig_size)
        location_prices = self.df.groupby('location')[self.target_name].mean().sort_values(ascending=False)
        bars = plt.bar(location_prices.index, location_prices.values, color='lightgreen', edgecolor='darkgreen')
        plt.title('Average House Price by Location', fontsize=16)
        plt.xlabel('Location')
        plt.ylabel('Average Price ($)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom')
        
        if save_plots:
            plt.savefig('plots/price_by_location.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizations created and saved to 'plots/' folder")
    
    def preprocess_data(self):
        """Preprocess data for machine learning"""
        print("\n" + "="*60)
        print("PREPROCESSING DATA")
        print("="*60)
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Handle categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        print(f"Encoding categorical columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        # Feature engineering
        print("Creating new features...")
        
        # Price per square foot
        df_processed['price_per_sqft'] = df_processed['sqft_living'] / (df_processed['sqft_living'] + 1)
        
        # Total rooms
        df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['bathrooms']
        
        # House age categories
        df_processed['age_category'] = pd.cut(df_processed['age'], 
                                            bins=[0, 5, 15, 30, 50], 
                                            labels=['New', 'Recent', 'Mature', 'Old'])
        df_processed['age_category'] = LabelEncoder().fit_transform(df_processed['age_category'])
        
        # Separate features and target
        X = df_processed.drop(columns=[self.target_name])
        y = df_processed[self.target_name]
        
        self.feature_names = X.columns.tolist()
        print(f"Features used: {self.feature_names}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print("✓ Data preprocessing completed")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple models and compare performance"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Train Linear Regression
        print("Training Linear Regression model...")
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Train Random Forest (for comparison)
        print("Training Random Forest model...")
        self.rf_model.fit(self.X_train, self.y_train)  # RF doesn't need scaling
        
        print("✓ Models trained successfully")
    
    def evaluate_models(self):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Linear Regression predictions
        y_pred_lr = self.model.predict(self.X_test_scaled)
        
        # Random Forest predictions
        y_pred_rf = self.rf_model.predict(self.X_test)
        
        # Calculate metrics for both models
        models = {
            'Linear Regression': y_pred_lr,
            'Random Forest': y_pred_rf
        }
        
        results = {}
        
        for model_name, predictions in models.items():
            mse = mean_squared_error(self.y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            
            results[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
            
            print(f"\n{model_name} Performance:")
            print(f"  Mean Squared Error: ${mse:,.0f}")
            print(f"  Root Mean Squared Error: ${rmse:,.0f}")
            print(f"  Mean Absolute Error: ${mae:,.0f}")
            print(f"  R² Score: {r2:.4f}")
        
        # Feature importance for Linear Regression
        print(f"\nLinear Regression Feature Importance:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        })
        feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        for _, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['coefficient']:,.2f}")
        
        return results, y_pred_lr, y_pred_rf
    
    def visualize_predictions(self, y_pred_lr, y_pred_rf, save_plots=True):
        """Visualize model predictions"""
        print("\n" + "="*60)
        print("VISUALIZING PREDICTIONS")
        print("="*60)
        
        if save_plots:
            Path('plots').mkdir(exist_ok=True)
        
        # Prediction vs Actual plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear Regression
        axes[0].scatter(self.y_test, y_pred_lr, alpha=0.6, color='blue')
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title('Linear Regression: Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Random Forest
        axes[1].scatter(self.y_test, y_pred_rf, alpha=0.6, color='green')
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Price')
        axes[1].set_ylabel('Predicted Price')
        axes[1].set_title('Random Forest: Actual vs Predicted')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('plots/prediction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Residual plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear Regression residuals
        residuals_lr = self.y_test - y_pred_lr
        axes[0].scatter(y_pred_lr, residuals_lr, alpha=0.6, color='blue')
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Price')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Linear Regression: Residual Plot')
        axes[0].grid(True, alpha=0.3)
        
        # Random Forest residuals
        residuals_rf = self.y_test - y_pred_rf
        axes[1].scatter(y_pred_rf, residuals_rf, alpha=0.6, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Price')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Random Forest: Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('plots/residual_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Prediction visualizations created")
    
    def predict_price(self, house_features):
        """Predict price for a new house"""
        if isinstance(house_features, dict):
            # Convert dict to DataFrame
            house_df = pd.DataFrame([house_features])
        else:
            house_df = house_features
        
        # Apply same preprocessing
        for col in self.label_encoders:
            if col in house_df.columns:
                house_df[col] = self.label_encoders[col].transform(house_df[col])
        
        # Add engineered features if they exist
        if 'sqft_living' in house_df.columns and 'price_per_sqft' in self.feature_names:
            house_df['price_per_sqft'] = house_df['sqft_living'] / (house_df['sqft_living'] + 1)
        
        if 'bedrooms' in house_df.columns and 'bathrooms' in house_df.columns and 'total_rooms' in self.feature_names:
            house_df['total_rooms'] = house_df['bedrooms'] + house_df['bathrooms']
        
        if 'age' in house_df.columns and 'age_category' in self.feature_names:
            age_category = pd.cut(house_df['age'], bins=[0, 5, 15, 30, 50], labels=[0, 1, 2, 3])
            house_df['age_category'] = age_category
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in house_df.columns:
                house_df[feature] = 0  # Default value
        
        # Select and order features
        house_features_ordered = house_df[self.feature_names]
        
        # Scale features
        house_features_scaled = self.scaler.transform(house_features_ordered)
        
        # Predict
        predicted_price = self.model.predict(house_features_scaled)[0]
        
        return predicted_price
    
    def save_model(self, filename='house_price_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved as {filename}")
    
    def load_model(self, filename='house_price_model.pkl'):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        
        print(f"✓ Model loaded from {filename}")

def main():
    """Main function to run the house price prediction analysis"""
    print("🏠 HOUSE PRICE PREDICTION SYSTEM")
    print("="*60)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Load data
    predictor.load_data()
    
    # Explore data
    correlation_matrix = predictor.explore_data()
    
    # Create visualizations
    predictor.visualize_data()
    
    # Preprocess data
    predictor.preprocess_data()
    
    # Train models
    predictor.train_models()
    
    # Evaluate models
    results, y_pred_lr, y_pred_rf = predictor.evaluate_models()
    
    # Visualize predictions
    predictor.visualize_predictions(y_pred_lr, y_pred_rf)
    
    # Save model
    predictor.save_model()
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    
    example_house = {
        'bedrooms': 3,
        'bathrooms': 2,
        'sqft_living': 2000,
        'sqft_lot': 8000,
        'floors': 2,
        'age': 10,
        'location': 'Suburb',
        'garage': 2,
        'pool': 0,
        'fireplace': 1
    }
    
    predicted_price = predictor.predict_price(example_house)
    
    print("House features:")
    for feature, value in example_house.items():
        print(f"  {feature}: {value}")
    
    print(f"\nPredicted price: ${predicted_price:,.0f}")
    
    print("\n✓ Analysis complete! Check the 'plots' folder for visualizations.")

if __name__ == "__main__":
    main()
