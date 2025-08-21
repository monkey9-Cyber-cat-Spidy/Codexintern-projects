import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.insights = []
        
    def load_csv(self, file_path):
        """Load CSV file and perform initial inspection"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"✓ Successfully loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"✗ Error loading CSV: {e}")
            return False
    
    def basic_info(self):
        """Display basic information about the dataset"""
        if self.df is None:
            print("No data loaded!")
            return
        
        print("\n" + "="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nData types:")
        print(self.df.dtypes)
        
        print("\nMissing values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        
        print("\nNumerical columns statistics:")
        print(self.df.describe())
    
    def analyze_column(self, column_name):
        """Perform detailed analysis on a specific column"""
        if self.df is None or column_name not in self.df.columns:
            print(f"Column '{column_name}' not found!")
            return
        
        col_data = self.df[column_name]
        
        print(f"\n" + "="*50)
        print(f"ANALYSIS FOR COLUMN: {column_name}")
        print("="*50)
        
        if pd.api.types.is_numeric_dtype(col_data):
            print(f"Average (Mean): {col_data.mean():.2f}")
            print(f"Median: {col_data.median():.2f}")
            print(f"Standard Deviation: {col_data.std():.2f}")
            print(f"Min: {col_data.min()}")
            print(f"Max: {col_data.max()}")
            print(f"Range: {col_data.max() - col_data.min()}")
            
            # Add insight
            if col_data.std() / col_data.mean() > 0.5:
                self.insights.append(f"{column_name} shows high variability (CV > 0.5)")
            
        else:
            print(f"Data type: {col_data.dtype}")
            print(f"Unique values: {col_data.nunique()}")
            print(f"Most common value: {col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A'}")
            print(f"Value counts (top 10):")
            print(col_data.value_counts().head(10))
    
    def create_visualizations(self, save_path="plots"):
        """Create various visualizations"""
        if self.df is None:
            print("No data loaded!")
            return
        
        Path(save_path).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            # 1. Bar chart for numeric columns (means)
            plt.figure(figsize=(12, 6))
            means = self.df[numeric_cols].mean()
            plt.bar(means.index, means.values)
            plt.title('Average Values of Numeric Columns')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{save_path}/bar_chart_means.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # 2. Correlation heatmap
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = self.df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.2f')
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                plt.savefig(f"{save_path}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
                plt.show()
                
                # Add correlation insights
                high_corr = correlation_matrix.abs() > 0.7
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        if high_corr.iloc[i, j]:
                            high_corr_pairs.append(f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}")
                
                if high_corr_pairs:
                    self.insights.append(f"Strong correlations found: {', '.join(high_corr_pairs)}")
            
            # 3. Scatter plot (first two numeric columns)
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 6))
                plt.scatter(self.df[numeric_cols[0]], self.df[numeric_cols[1]], alpha=0.6)
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
                plt.title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
                plt.tight_layout()
                plt.savefig(f"{save_path}/scatter_plot.png", dpi=300, bbox_inches='tight')
                plt.show()
        
        # 4. Distribution plots for numeric columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols[:4]):
                if i < 4:
                    axes[i].hist(self.df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_cols), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/distributions.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"✓ Visualizations saved to '{save_path}' folder")
    
    def generate_insights(self):
        """Generate and display insights based on analysis"""
        if self.df is None:
            print("No data loaded!")
            return
        
        print("\n" + "="*50)
        print("KEY INSIGHTS AND OBSERVATIONS")
        print("="*50)
        
        # Data quality insights
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        if missing_percentage.max() > 10:
            self.insights.append(f"High missing data: {missing_percentage.idxmax()} has {missing_percentage.max():.1f}% missing values")
        
        # Data size insights
        if len(self.df) > 100000:
            self.insights.append("Large dataset (>100K rows) - suitable for machine learning")
        elif len(self.df) < 100:
            self.insights.append("Small dataset (<100 rows) - limited statistical power")
        
        # Numeric columns insights
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            skewness = self.df[col].skew()
            if abs(skewness) > 2:
                self.insights.append(f"{col} is highly skewed (skewness: {skewness:.2f})")
        
        # Display all insights
        if self.insights:
            for i, insight in enumerate(self.insights, 1):
                print(f"{i}. {insight}")
        else:
            print("No specific insights generated from automatic analysis.")
        
        print(f"\n{len(self.insights)} insights generated in total.")

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'education_years': np.random.randint(8, 20, 1000),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 1000),
        'satisfaction_score': np.random.uniform(1, 10, 1000)
    }
    
    # Add some correlations
    data['income'] = data['income'] + data['education_years'] * 2000 + np.random.normal(0, 5000, 1000)
    data['satisfaction_score'] = 5 + (data['income'] / 10000) * 0.5 + np.random.normal(0, 1, 1000)
    
    df = pd.DataFrame(data)
    df['income'] = np.maximum(df['income'], 20000)  # Minimum income
    df['satisfaction_score'] = np.clip(df['satisfaction_score'], 1, 10)  # Clip satisfaction
    
    return df

def main():
    """Main function to run the data analysis"""
    analyzer = DataAnalyzer()
    
    print("🐼 PANDAS DATA ANALYSIS TOOL")
    print("="*50)
    
    # Try to load existing CSV or create sample data
    csv_file = "sample_data.csv"
    
    if not Path(csv_file).exists():
        print("Creating sample dataset for demonstration...")
        sample_df = create_sample_dataset()
        sample_df.to_csv(csv_file, index=False)
        print(f"✓ Sample dataset saved as '{csv_file}'")
    
    # Load and analyze data
    if analyzer.load_csv(csv_file):
        analyzer.basic_info()
        
        # Analyze specific columns
        numeric_cols = analyzer.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analyzer.analyze_column(numeric_cols[0])  # Analyze first numeric column
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Generate insights
        analyzer.generate_insights()
        
        print("\n✓ Analysis complete! Check the 'plots' folder for visualizations.")

if __name__ == "__main__":
    main()