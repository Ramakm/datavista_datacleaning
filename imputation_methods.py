import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class DataImputer:
    """Advanced imputation methods using only pandas and numpy"""
    
    def __init__(self):
        self.imputation_report = []
        self.dropped_columns = []
        self.dropped_rows_count = 0
        
    def analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data patterns and provide recommendations"""
        missing_info = {}
        total_rows = len(df)
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100
            
            missing_info[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'data_type': str(df[col].dtype),
                'unique_values': df[col].nunique(),
                'recommendation': self._get_recommendation(missing_percentage, df[col])
            }
        
        return missing_info
    
    def _get_recommendation(self, missing_percentage: float, series: pd.Series) -> str:
        """Get recommendation based on missing percentage and data type"""
        if missing_percentage == 0:
            return "No action needed"
        elif missing_percentage > 40:
            return "Consider dropping column"
        elif missing_percentage > 30:
            return "High missing - use advanced imputation or drop"
        elif missing_percentage < 5:
            return "Consider dropping rows or simple imputation"
        elif missing_percentage < 15:
            return "Good candidate for imputation"
        else:
            return "Moderate missing - use robust imputation"
    
    def detect_outliers_iqr(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using IQR method"""
        outliers_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers = df[outlier_mask]
            
            outliers_info[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'outlier_indices': outliers.index.tolist()
            }
        
        return outliers_info
    
    def smart_drop_strategy(self, df: pd.DataFrame, 
                           column_threshold: float = 40.0,
                           row_threshold: float = 5.0) -> pd.DataFrame:
        """Smart dropping strategy based on thresholds"""
        df_processed = df.copy()
        original_shape = df_processed.shape
        
        # 1. Drop columns with high missing percentage
        columns_to_drop = []
        for col in df_processed.columns:
            missing_percentage = (df_processed[col].isnull().sum() / len(df_processed)) * 100
            if missing_percentage > column_threshold:
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df_processed = df_processed.drop(columns=columns_to_drop)
            self.dropped_columns.extend(columns_to_drop)
            self.imputation_report.append(f"Dropped {len(columns_to_drop)} columns with >{column_threshold}% missing: {columns_to_drop}")
        
        # 2. Drop rows with low missing percentage
        total_cols = len(df_processed.columns)
        rows_to_drop = []
        
        for idx, row in df_processed.iterrows():
            missing_in_row = row.isnull().sum()
            missing_percentage = (missing_in_row / total_cols) * 100
            
            if missing_percentage > 0 and missing_percentage <= row_threshold:
                rows_to_drop.append(idx)
        
        if rows_to_drop:
            df_processed = df_processed.drop(index=rows_to_drop)
            self.dropped_rows_count = len(rows_to_drop)
            self.imputation_report.append(f"Dropped {len(rows_to_drop)} rows with <={row_threshold}% missing values")
        
        if df_processed.shape != original_shape:
            self.imputation_report.append(f"Shape after smart dropping: {original_shape} → {df_processed.shape}")
        
        return df_processed
    
    def remove_outliers_before_imputation(self, df: pd.DataFrame, 
                                        outlier_threshold: float = 10.0) -> pd.DataFrame:
        """Remove outliers before imputation to avoid biased imputation"""
        df_processed = df.copy()
        outliers_info = self.detect_outliers_iqr(df_processed)
        
        rows_to_remove = set()
        
        for col, info in outliers_info.items():
            if info['outlier_percentage'] < outlier_threshold:  # Only remove if not too many outliers
                rows_to_remove.update(info['outlier_indices'])
        
        if rows_to_remove:
            df_processed = df_processed.drop(index=list(rows_to_remove))
            self.imputation_report.append(f"Removed {len(rows_to_remove)} outlier rows before imputation")
        
        return df_processed
    
    def simple_imputation(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """Simple imputation with automatic strategy selection"""
        df_processed = df.copy()
        
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['object', 'category']:
                    # Categorical data - use mode
                    mode_val = df_processed[col].mode()
                    if len(mode_val) > 0:
                        df_processed[col].fillna(mode_val[0], inplace=True)
                        self.imputation_report.append(f"Filled '{col}' missing values with mode: {mode_val[0]}")
                else:
                    # Numerical data
                    if strategy == 'auto' or strategy == 'median':
                        # Use median for numerical data (more robust to outliers)
                        median_val = df_processed[col].median()
                        df_processed[col].fillna(median_val, inplace=True)
                        self.imputation_report.append(f"Filled '{col}' missing values with median: {median_val:.2f}")
                    elif strategy == 'mean':
                        mean_val = df_processed[col].mean()
                        df_processed[col].fillna(mean_val, inplace=True)
                        self.imputation_report.append(f"Filled '{col}' missing values with mean: {mean_val:.2f}")
        
        return df_processed
    
    def forward_fill_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward fill imputation for time series data"""
        df_processed = df.copy()
        
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                original_missing = df_processed[col].isnull().sum()
                df_processed[col].fillna(method='ffill', inplace=True)
                
                # If still missing values, use backward fill
                if df_processed[col].isnull().sum() > 0:
                    df_processed[col].fillna(method='bfill', inplace=True)
                
                remaining_missing = df_processed[col].isnull().sum()
                filled_count = original_missing - remaining_missing
                
                if filled_count > 0:
                    self.imputation_report.append(f"Forward/backward filled {filled_count} values in '{col}'")
        
        return df_processed
    
    def interpolation_imputation(self, df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """Interpolation imputation for numeric data"""
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                original_missing = df_processed[col].isnull().sum()
                df_processed[col] = df_processed[col].interpolate(method=method)
                remaining_missing = df_processed[col].isnull().sum()
                filled_count = original_missing - remaining_missing
                
                if filled_count > 0:
                    self.imputation_report.append(f"Interpolated {filled_count} values in '{col}' using {method} method")
        
        return df_processed
    
    def advanced_imputation_pipeline(self, df: pd.DataFrame,
                                   column_drop_threshold: float = 40.0,
                                   row_drop_threshold: float = 5.0,
                                   remove_outliers: bool = True,
                                   imputation_method: str = 'simple',
                                   **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        """Complete advanced imputation pipeline without sklearn"""
        
        self.imputation_report = []
        self.dropped_columns = []
        self.dropped_rows_count = 0
        
        df_processed = df.copy()
        original_shape = df_processed.shape
        
        # Step 1: Smart dropping strategy
        df_processed = self.smart_drop_strategy(df_processed, column_drop_threshold, row_drop_threshold)
        
        # Step 2: Remove outliers before imputation (optional)
        if remove_outliers and df_processed.select_dtypes(include=[np.number]).shape[1] > 0:
            df_processed = self.remove_outliers_before_imputation(df_processed)
        
        # Step 3: Apply selected imputation method
        if df_processed.isnull().sum().sum() > 0:  # Still have missing values
            if imputation_method == 'simple':
                df_processed = self.simple_imputation(df_processed, kwargs.get('strategy', 'auto'))
            elif imputation_method == 'forward_fill':
                df_processed = self.forward_fill_imputation(df_processed)
            elif imputation_method == 'interpolation':
                df_processed = self.interpolation_imputation(df_processed, kwargs.get('method', 'linear'))
        
        # Final report
        final_missing = df_processed.isnull().sum().sum()
        if final_missing == 0:
            self.imputation_report.append("✅ All missing values successfully handled!")
        else:
            self.imputation_report.append(f"⚠️ {final_missing} missing values remain")
        
        self.imputation_report.append(f"Final shape: {original_shape} → {df_processed.shape}")
        
        return df_processed, self.imputation_report
    
    def get_imputation_summary(self) -> Dict:
        """Get summary of imputation operations"""
        return {
            'dropped_columns': self.dropped_columns,
            'dropped_rows_count': self.dropped_rows_count,
            'imputation_report': self.imputation_report
        }