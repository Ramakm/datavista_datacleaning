import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Optional, Tuple
from imputation_methods import DataImputer

def detect_file_type(file) -> str:
    """Detect file type based on file extension"""
    if file.name.endswith('.csv'):
        return 'csv'
    elif file.name.endswith(('.xlsx', '.xls')):
        return 'excel'
    elif file.name.endswith('.json'):
        return 'json'
    elif file.name.endswith('.txt'):
        return 'txt'
    else:
        return 'unknown'

def read_file(file, file_type: str) -> Optional[pd.DataFrame]:
    """Read different file types and return DataFrame"""
    try:
        if file_type == 'csv':
            # Try different encodings and separators
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                for sep in [',', ';', '\t', '|']:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, encoding=encoding, sep=sep)
                        if len(df.columns) > 1:  # Good separator found
                            return df
                    except:
                        continue
            # Fallback to default
            file.seek(0)
            return pd.read_csv(file)
            
        elif file_type == 'excel':
            return pd.read_excel(file, engine='openpyxl')
            
        elif file_type == 'json':
            file.seek(0)
            return pd.read_json(file)
            
        elif file_type == 'txt':
            file.seek(0)
            # Try tab-separated first, then other separators
            for sep in ['\t', ',', ';', '|']:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, sep=sep)
                    if len(df.columns) > 1:
                        return df
                except:
                    continue
            # Fallback to single column
            file.seek(0)
            content = file.read().decode('utf-8')
            lines = content.strip().split('\n')
            return pd.DataFrame({'data': lines})
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Clean the dataframe and return cleaned df with cleaning report"""
    cleaning_report = []
    original_shape = df.shape
    
    # 1. Remove completely empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    if df.shape != original_shape:
        cleaning_report.append(f"Removed empty rows/columns. Shape: {original_shape} → {df.shape}")
    
    # 2. Clean column names
    original_columns = df.columns.tolist()
    df.columns = df.columns.astype(str)
    df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
    if df.columns.tolist() != original_columns:
        cleaning_report.append("Cleaned column names (removed spaces, special chars)")
    
    # 3. Handle duplicate columns
    cols = df.columns.tolist()
    duplicated_cols = set([x for x in cols if cols.count(x) > 1])
    if duplicated_cols:
        df.columns = pd.io.common.dedup_names(df.columns, is_potential_multiindex=False)
        cleaning_report.append(f"Renamed duplicate columns: {duplicated_cols}")
    
    # 4. Remove duplicate rows
    original_rows = len(df)
    df = df.drop_duplicates()
    if len(df) != original_rows:
        cleaning_report.append(f"Removed {original_rows - len(df)} duplicate rows")
    
    # 5. Basic data type optimization
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if not numeric_series.isna().all():
                df[col] = numeric_series
                cleaning_report.append(f"Converted '{col}' to numeric")
    
    # 6. Handle missing values info
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        cleaning_report.append(f"Found missing values in columns: {missing_info[missing_info > 0].to_dict()}")
    
    return df, cleaning_report

def display_imputation_analysis(df: pd.DataFrame, imputer: DataImputer):
    """Display missing data analysis and recommendations with improved styling"""
    st.subheader("🔍 Missing Data Analysis")
    
    missing_analysis = imputer.analyze_missing_data(df)
    
    # Create analysis dataframe
    analysis_data = []
    for col, info in missing_analysis.items():
        analysis_data.append({
            'Column': col,
            'Missing Count': info['missing_count'],
            'Missing %': f"{info['missing_percentage']:.1f}%",
            'Data Type': info['data_type'],
            'Unique Values': info['unique_values'],
            'Recommendation': info['recommendation']
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # FIXED: Improved styling with better contrast
    def row_style(row):
        """Apply row-based styling with better contrast"""
        missing_pct = float(row['Missing %'].replace('%', ''))
        
        if missing_pct > 40:
            return ['background-color: #d32f2f; color: white; font-weight: bold;'] * len(row)
        elif missing_pct > 30:
            return ['background-color: #f57c00; color: white; font-weight: bold;'] * len(row)
        elif missing_pct > 15:
            return ['background-color: #fbc02d; color: black; font-weight: bold;'] * len(row)
        elif missing_pct > 0:
            return ['background-color: #388e3c; color: white; font-weight: bold;'] * len(row)
        else:
            return ['background-color: #e8eaf6; color: black;'] * len(row)
    
    # Apply row-based styling
    styled_df = analysis_df.style.apply(row_style, axis=1)
    
    # Add specific styling for recommendations
    def style_recommendations(val):
        if 'drop' in val.lower():
            return 'background-color: #ffebee; color: #c62828; font-weight: bold; padding: 5px'
        elif 'imputation' in val.lower():
            return 'background-color: #e8f5e8; color: #2e7d32; font-weight: bold; padding: 5px'
        elif 'no action' in val.lower():
            return 'background-color: #f3e5f5; color: #7b1fa2; font-weight: bold; padding: 5px'
        return 'color: black; padding: 5px'
    
    styled_df = styled_df.applymap(style_recommendations, subset=['Recommendation'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Create a legend for missing data colors
    st.markdown("### 📊 Missing Data Legend:")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div style="background-color: #d32f2f; color: white; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold;">Critical (>40%)</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="background-color: #f57c00; color: white; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold;">High (30-40%)</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="background-color: #fbc02d; color: black; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold;">Moderate (15-30%)</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div style="background-color: #388e3c; color: white; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold;">Low (0-15%)</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div style="background-color: #e8eaf6; color: black; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold;">Complete (0%)</div>', unsafe_allow_html=True)
    
    # Show outlier analysis with improved styling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.subheader("📊 Outlier Analysis")
        outliers_info = imputer.detect_outliers_iqr(df)
        
        outlier_data = []
        for col, info in outliers_info.items():
            outlier_data.append({
                'Column': col,
                'Outlier Count': info['outlier_count'],
                'Outlier %': f"{info['outlier_percentage']:.1f}%"
            })
        
        if outlier_data:
            outlier_df = pd.DataFrame(outlier_data)
            
            def style_outlier_data(val):
                """Style outlier percentages"""
                if isinstance(val, str) and val.endswith('%'):
                    percentage = float(val.replace('%', ''))
                    if percentage > 20:
                        return 'background-color: #d32f2f; color: white; font-weight: bold; padding: 5px'
                    elif percentage > 10:
                        return 'background-color: #f57c00; color: white; font-weight: bold; padding: 5px'
                    elif percentage > 5:
                        return 'background-color: #fbc02d; color: black; font-weight: bold; padding: 5px'
                    elif percentage > 0:
                        return 'background-color: #388e3c; color: white; font-weight: bold; padding: 5px'
                    else:
                        return 'background-color: #e8eaf6; color: black; padding: 5px'
                return 'color: black; padding: 5px'
            
            styled_outlier_df = outlier_df.style.applymap(style_outlier_data, subset=['Outlier %'])
            
            # Apply consistent styling to other columns
            def style_other_columns(val):
                return 'color: black; padding: 5px'
            
            styled_outlier_df = styled_outlier_df.applymap(style_other_columns, subset=['Column', 'Outlier Count'])
            st.dataframe(styled_outlier_df, use_container_width=True)
            
            # Outlier legend
            st.markdown("### 🎯 Outlier Legend:")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown('<div style="background-color: #d32f2f; color: white; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold;">Very High (>20%)</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div style="background-color: #f57c00; color: white; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold;">High (10-20%)</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div style="background-color: #fbc02d; color: black; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold;">Moderate (5-10%)</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div style="background-color: #388e3c; color: white; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold;">Low (0-5%)</div>', unsafe_allow_html=True)
            with col5:
                st.markdown('<div style="background-color: #e8eaf6; color: black; padding: 10px; text-align: center; border-radius: 5px; font-weight: bold;">None (0%)</div>', unsafe_allow_html=True)

def display_data_comparison(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, rows_to_show):
    """Display original and cleaned data side by side"""
    
    # Create comparison tabs
    tab1, tab2, tab3 = st.tabs(["📊 Side-by-Side Comparison", "📋 Original Data", "✨ Cleaned Data"])
    
    with tab1:
        st.subheader("Data Comparison")
        
        # Summary comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Original Data**")
            st.info(f"Shape: {original_df.shape[0]:,} rows × {original_df.shape[1]} columns")
            
            # Show first few rows of original
            if rows_to_show == "All":
                display_original = original_df
            else:
                display_original = original_df.head(rows_to_show)
            
            st.dataframe(display_original, use_container_width=True, key="original_comparison")
            
        with col2:
            st.markdown("**✨ Cleaned Data**")
            st.success(f"Shape: {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns")
            
            # Show first few rows of cleaned
            if rows_to_show == "All":
                display_cleaned = cleaned_df
            else:
                display_cleaned = cleaned_df.head(rows_to_show)
            
            st.dataframe(display_cleaned, use_container_width=True, key="cleaned_comparison")
        
        # Show differences summary
        st.subheader("📈 Comparison Summary")
        
        comparison_data = {
            'Metric': ['Rows', 'Columns', 'Memory Usage (KB)', 'Missing Values', 'Duplicate Rows'],
            'Original': [
                f"{original_df.shape[0]:,}",
                f"{original_df.shape[1]}",
                f"{original_df.memory_usage(deep=True).sum() / 1024:.1f}",
                f"{original_df.isnull().sum().sum():,}",
                f"{original_df.duplicated().sum():,}"
            ],
            'Cleaned': [
                f"{cleaned_df.shape[0]:,}",
                f"{cleaned_df.shape[1]}",
                f"{cleaned_df.memory_usage(deep=True).sum() / 1024:.1f}",
                f"{cleaned_df.isnull().sum().sum():,}",
                f"{cleaned_df.duplicated().sum():,}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("📋 Original Data")
        
        # Original data metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{original_df.shape[0]:,}")
        with col2:
            st.metric("Columns", original_df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{original_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Display original data
        if rows_to_show == "All":
            display_original = original_df
        else:
            display_original = original_df.head(rows_to_show)
        
        st.dataframe(display_original, use_container_width=True, key="original_full")
        
        # Original data column info
        with st.expander("📋 Original Data Column Information"):
            original_col_info = pd.DataFrame({
                'Column': original_df.columns,
                'Data Type': original_df.dtypes.astype(str),
                'Non-Null Count': original_df.count(),
                'Missing Values': original_df.isnull().sum(),
                'Missing %': (original_df.isnull().sum() / len(original_df) * 100).round(2)
            })
            st.dataframe(original_col_info, use_container_width=True)
    
    with tab3:
        st.subheader("✨ Cleaned Data")
        
        # Cleaned data metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{cleaned_df.shape[0]:,}")
        with col2:
            st.metric("Columns", cleaned_df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{cleaned_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Display cleaned data
        if rows_to_show == "All":
            display_cleaned = cleaned_df
        else:
            display_cleaned = cleaned_df.head(rows_to_show)
        
        st.dataframe(display_cleaned, use_container_width=True, key="cleaned_full")
        
        # Cleaned data column info
        with st.expander("📋 Cleaned Data Column Information"):
            cleaned_col_info = pd.DataFrame({
                'Column': cleaned_df.columns,
                'Data Type': cleaned_df.dtypes.astype(str),
                'Non-Null Count': cleaned_df.count(),
                'Missing Values': cleaned_df.isnull().sum(),
                'Missing %': (cleaned_df.isnull().sum() / len(cleaned_df) * 100).round(2)
            })
            st.dataframe(cleaned_col_info, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Advanced Data Viewer & Cleaner",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Advanced Data Viewer & Cleaner")
    st.markdown("Upload any data file to view, clean, and apply advanced imputation methods")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json', 'txt'],
        help="Supported formats: CSV, Excel (.xlsx, .xls), JSON, TXT"
    )
    
    if uploaded_file is not None:
        # Show file info
        st.info(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size:,} bytes")
        
        # Detect and read file
        file_type = detect_file_type(uploaded_file)
        
        with st.spinner("Reading and processing file..."):
            original_df = read_file(uploaded_file, file_type)
        
        if original_df is not None:
            # Show original data info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📋 File Information")
                st.write(f"**Shape:** {original_df.shape[0]:,} rows × {original_df.shape[1]} columns")
                st.write(f"**File Type:** {file_type.upper()}")
                
            # Data cleaning and imputation options
            with col2:
                st.subheader("🧹 Processing Options")
                basic_clean = st.checkbox("Basic cleaning", value=True)
                advanced_imputation = st.checkbox("Advanced imputation", value=False)
            
            # Initialize imputer
            imputer = DataImputer()
            
            # Process data
            cleaned_df = None
            cleaning_report = []
            imputation_report = []
            
            # Basic cleaning
            if basic_clean:
                with st.spinner("Applying basic cleaning..."):
                    cleaned_df, cleaning_report = clean_data(original_df.copy())
            else:
                cleaned_df = original_df.copy()
            
            # FIXED: Show missing data analysis on cleaned data
            if cleaned_df.isnull().sum().sum() > 0:
                display_imputation_analysis(cleaned_df, imputer)
            
                # Advanced imputation settings
                if advanced_imputation:
                    st.subheader("⚙️ Advanced Imputation Settings")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        column_drop_threshold = st.slider(
                            "Column drop threshold (%)", 
                            min_value=20, max_value=60, value=40,
                            help="Drop columns with missing values above this percentage"
                        )
                    
                    with col2:
                        row_drop_threshold = st.slider(
                            "Row drop threshold (%)", 
                            min_value=1, max_value=20, value=5,
                            help="Drop rows with missing values below this percentage"
                        )
                    
                    with col3:
                        remove_outliers = st.checkbox(
                            "Remove outliers before imputation", 
                            value=True,
                            help="Remove outliers to avoid biased imputation"
                        )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        imputation_method = st.selectbox(
                            "Imputation method",
                            ['simple', 'forward_fill', 'interpolation'],
                            help="Choose imputation strategy"
                        )
                    
                    with col2:
                        if imputation_method == 'simple':
                            strategy = st.selectbox("Strategy", ['auto', 'mean', 'median'], index=0)
                        elif imputation_method == 'interpolation':
                            interpolation_method = st.selectbox("Method", ['linear', 'polynomial', 'spline'], index=0)
                        else:
                            strategy = 'auto'  # Default for forward_fill
                            interpolation_method = 'linear'  # Default
                    
                    # Apply advanced imputation
                    with st.spinner("Applying advanced imputation..."):
                        kwargs = {}
                        if imputation_method == 'simple':
                            kwargs['strategy'] = strategy
                        elif imputation_method == 'interpolation':
                            kwargs['method'] = interpolation_method
                        
                        cleaned_df, imputation_report = imputer.advanced_imputation_pipeline(
                            cleaned_df,
                            column_drop_threshold=column_drop_threshold,
                            row_drop_threshold=row_drop_threshold,
                            remove_outliers=remove_outliers,
                            imputation_method=imputation_method,
                            **kwargs
                        )
                    
                    # Show imputation report
                    st.subheader("🔧 Advanced Imputation Report")
                    for report in imputation_report:
                        st.write(f"✅ {report}")
            
            # Show basic cleaning report
            if cleaning_report:
                st.subheader("🧹 Basic Cleaning Report")
                for report in cleaning_report:
                    st.write(f"✅ {report}")
            
            # Display options
            st.subheader("🔍 Display Options")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                rows_to_show = st.selectbox("Rows to display", [10, 25, 50, 100, "All"], index=1)
            with col2:
                show_stats = st.checkbox("Show statistics", value=False)
            with col3:
                comparison_mode = st.checkbox("Show comparison view", value=True)
            
            # Display data
            if comparison_mode:
                display_data_comparison(original_df, cleaned_df, rows_to_show)
                current_df = cleaned_df
            else:
                st.subheader("📊 Data Preview")
                
                # Data summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{cleaned_df.shape[0]:,}")
                with col2:
                    st.metric("Columns", cleaned_df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{cleaned_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                # Display the dataframe
                if rows_to_show == "All":
                    display_rows = cleaned_df
                else:
                    display_rows = cleaned_df.head(rows_to_show)
                
                st.dataframe(display_rows, use_container_width=True)
                current_df = cleaned_df
            
            # Show basic statistics if requested
            if show_stats:
                st.subheader("📈 Statistical Summary")
                numeric_cols = current_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(current_df[numeric_cols].describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found for statistical summary")
            
            # Download section
            st.subheader("💾 Download Data")
            
            # Create download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Download original data
                csv_buffer_original = io.StringIO()
                original_df.to_csv(csv_buffer_original, index=False)
                csv_string_original = csv_buffer_original.getvalue()
                
                st.download_button(
                    label="📥 Download Original CSV",
                    data=csv_string_original,
                    file_name=f"original_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                    mime="text/csv",
                    help="Download the original uploaded data as CSV"
                )
            
            with col2:
                # Download processed data
                csv_buffer_cleaned = io.StringIO()
                cleaned_df.to_csv(csv_buffer_cleaned, index=False)
                csv_string_cleaned = csv_buffer_cleaned.getvalue()
                
                st.download_button(
                    label="📥 Download Processed CSV",
                    data=csv_string_cleaned,
                    file_name=f"processed_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                    mime="text/csv",
                    help="Download the processed data as CSV"
                )
            
        else:
            st.error("❌ Could not read the file. Please check the file format and try again.")
    
    # Instructions
    with st.sidebar:
        st.subheader("📖 Instructions")
        st.markdown("""
        **How to use:**
        1. Upload your data file
        2. Review missing data analysis with color-coded table
        3. Configure imputation settings
        4. Apply processing and compare results
        5. Download processed data
        
        **Color Coding:**
        - 🔴 **Critical (>40% missing)**: Consider dropping
        - 🟠 **High (30-40% missing)**: Advanced imputation
        - 🟡 **Moderate (15-30% missing)**: Standard imputation
        - 🟢 **Low (0-15% missing)**: Simple imputation
        - ⚪ **Complete (0% missing)**: No action needed
        
        **Imputation Methods:**
        - **Simple**: Mean/median/mode imputation
        - **Forward Fill**: Use previous values
        - **Interpolation**: Linear/polynomial interpolation
        
        **Smart Rules:**
        - Drop columns with >30-40% missing
        - Drop rows with <5% missing
        - Remove outliers before imputation
        - Use robust methods for skewed data
        
        **Features:**
        - 🔍 Color-coded missing data analysis
        - 📊 Outlier detection with visual indicators
        - 🧹 Smart dropping strategies
        - 🤖 Advanced imputation methods
        - 📈 Before/after comparison
        - 📋 Interactive legends
        """)

if __name__ == "__main__":
    main()