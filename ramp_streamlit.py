"""
Sales Ramp Analysis Streamlit App

This Streamlit application provides an interactive interface for analyzing sales representative 
performance data, including ramp time analysis, performance metrics, and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import tempfile
from sampledata import generate_sales_data

# Set page config
st.set_page_config(
    page_title="Sales Ramp Analysis Tool",
    page_icon="📈",
    layout="wide"
)

# Helper functions
def linear_regression(x, a, b):
    """Linear regression model: y = ax + b"""
    return a * x + b

def logistic_function(x, L, k, x0):
    """Logistic growth function: y = L / (1 + exp(-k(x - x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))

def gompertz_function(x, a, b, c):
    """Gompertz growth model: y = a * exp(-b * exp(-cx))"""
    return a * np.exp(-b * np.exp(-c * x))

def get_model_function(model_type):
    """Return the appropriate model function based on type."""
    models = {
        'linear': linear_regression,
        'logistic': logistic_function,
        'gompertz': gompertz_function
    }
    return models.get(model_type)

def get_initial_params(model_type, y_data, x_data):
    """Determine initial parameters for model fitting."""
    if model_type == 'linear':
        return [(y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0]), y_data[0]]
    elif model_type == 'gompertz':
        return [max(y_data) * 1.2, 5, 0.2]
    return [max(y_data), 0.2, np.mean(x_data)]

def calculate_ramp_time(L, k, x0, ramp_pct):
    """Calculate time to reach specified percentage of maximum value."""
    p = ramp_pct / 100
    return x0 - np.log(1/p - 1)/k

def prepare_data(df, verbose=True):
    """Prepare and validate the sales dataset for analysis."""
    categorical_cols = {
        'EID': str,
        'Market': str,
        'Theater': str,
        'Region': str,
        'Segment': str,
        'Territory_Profile': str
    }
    month_cols = [f'Month{i}' for i in range(1, 37)]
    trailing_cols = [f'Trailing{i}' for i in range(1, 37)]
    
    df = df.copy()
    
    # Convert categorical columns
    for col, dtype in categorical_cols.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
            if verbose:
                st.write(f"{col}: {df[col].nunique()} unique values")
    
    # Convert dates
    if 'StartDate' in df.columns:
        df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
        if verbose:
            null_dates = df['StartDate'].isnull().sum()
            if null_dates > 0:
                st.warning(f"{null_dates} rows have invalid dates")
            st.write(f"Date range: {df['StartDate'].min()} to {df['StartDate'].max()}")
    
    # Convert numeric data
    for col in month_cols + trailing_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Remove zero-value rows
    numeric_cols = [col for col in month_cols + trailing_cols if col in df.columns]
    initial_rows = len(df)
    df = df[df[numeric_cols].any(axis=1)]
    
    if verbose:
        st.write(f"Removed {initial_rows - len(df)} rows with all zero values")
        st.write(f"Final dataset shape: {df.shape}")
        st.write(f"Average monthly booking: ${df[month_cols].mean().mean():,.2f}")
        st.write(f"Average trailing booking: ${df[trailing_cols].mean().mean():,.2f}")
    
    return df, month_cols, trailing_cols

def plot_performance(df, cols, plot_type='monthly', segment=None):
    """Generate performance visualizations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == 'monthly':
        for segment in df['Segment'].unique():
            segment_df = df[df['Segment'] == segment]
            monthly_mean = segment_df[cols].mean()
            plt.plot(range(1, len(cols) + 1), monthly_mean, label=segment)
        plt.title('Monthly Performance by Segment')
        plt.xlabel('Month of Tenure')
    else:  # distribution
        final_performance = df[cols[-1]]
        if segment:
            final_performance = df[df['Segment'] == segment][cols[-1]]
        sns.histplot(final_performance, bins=30)
        plt.title(f'Distribution of Final Performance{" - " + segment if segment else ""}')
        plt.xlabel('Trailing 12-Month Bookings ($)')
    
    plt.ylabel('Monthly Bookings ($)' if plot_type == 'monthly' else 'Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    return fig

def analyze_ramp(df, segments=None, markets=None, theaters=None, regions=None, 
               territories=None, date_range=(1, 36), ramp_target_pct=90, model_type='logistic'):
    """Analyze sales ramp data with specified filters."""
    month_cols = [f'Month{i}' for i in range(1, 37)]
    trailing_cols = [f'Trailing{i}' for i in range(1, 37)]
    selected_cols = trailing_cols[date_range[0]-1:date_range[1]]
    
    # Apply filters
    if segments:
        df = df[df['Segment'].isin(segments)]
    if markets:
        df = df[df['Market'].isin(markets)]
    if theaters:
        df = df[df['Theater'].isin(theaters)]
    if regions:
        df = df[df['Region'].isin(regions)]
    if territories:
        df = df[df['Territory_Profile'].isin(territories)]
    
    if len(df) == 0:
        st.error("No data available for the selected filters.")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = {}
    model_func = get_model_function(model_type)
    
    for segment in df['Segment'].unique():
        segment_df = df[df['Segment'] == segment]
        mean_performance = segment_df[selected_cols].mean()
        
        # Plot individual trajectories
        for _, row in segment_df.iterrows():
            plt.plot(range(date_range[0], date_range[1]+1), 
                    row[selected_cols], alpha=0.1)
        
        # Plot mean
        plt.plot(range(date_range[0], date_range[1]+1),
                mean_performance,
                linewidth=3,
                label=f'{segment} Mean')
        
        try:
            x_data = np.array(range(date_range[0], date_range[1]+1))
            y_data = mean_performance.values
            
            p0 = get_initial_params(model_type, y_data, x_data)
            popt, _ = curve_fit(model_func, x_data, y_data, p0=p0)
            
            y_pred = model_func(x_data, *popt)
            r2 = r2_score(y_data, y_pred)
            rmse = np.sqrt(mean_squared_error(y_data, y_pred))
            
            x_smooth = np.linspace(date_range[0], date_range[1], 100)
            y_smooth = model_func(x_smooth, *popt)
            plt.plot(x_smooth, y_smooth, '--',
                    linewidth=2,
                    label=f'{segment} {model_type.title()} Fit')
            
            if model_type == 'logistic':
                ramp_time = calculate_ramp_time(popt[0], popt[1], popt[2], ramp_target_pct)
            elif model_type == 'gompertz':
                a, b, c = popt
                target = (ramp_target_pct/100) * a
                ramp_time = (-1/c) * np.log(-np.log(target/a)/b)
            else:  # linear
                slope, intercept = popt
                target = (ramp_target_pct/100) * max(y_data)
                ramp_time = (target - intercept) / slope
            
            ramp_value = model_func(ramp_time, *popt)
            metrics[segment] = {
                'Ramp Time': f"{ramp_time:.1f} months",
                'Target Value': f"${ramp_value:,.2f}",
                'R²': f"{r2:.3f}",
                'RMSE': f"${rmse:,.2f}"
            }
            
            plt.plot(ramp_time, ramp_value, 'o',
                    alpha=0.8, markersize=10)
            plt.hlines(y=ramp_value, xmin=ramp_time, xmax=date_range[1],
                      linestyles=':', alpha=0.5)
            
        except Exception as e:
            st.warning(f"Could not fit {model_type} model for {segment} segment: {str(e)}")
    
    plt.title(f'Sales Rep Trailing Performance by Segment\n{model_type.title()} Regression')
    plt.xlabel('Month of Tenure')
    plt.ylabel('Trailing 12-Month Bookings ($)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    
    return fig, metrics

def generate_sample_data(num_reps, segment_params, seed=42, noise_level=0.50):
    """Generate sample data with custom parameters"""
    df = generate_sales_data(
        num_reps=num_reps,
        segment_params=segment_params,
        seed=seed,
        noise_level=noise_level
    )
    
    # Create a temporary file to store the generated data
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        return tmp_file.name

def create_sample_template():
    """Create a sample CSV template with correct columns."""
    sample_data = {
        'EID': ['SR001', 'SR002'],
        'StartDate': ['2023-01-01', '2023-01-01'],
        'Market': ['SampleMarket', 'SampleMarket'],
        'Theater': ['SampleTheater', 'SampleTheater'],
        'Region': ['SampleRegion', 'SampleRegion'],
        'Segment': ['Commercial', 'Enterprise'],
        'Territory_Profile': ['Acquisition', 'Expansion']
    }
    
    # Add Month1-Month36 columns
    for i in range(1, 37):
        sample_data[f'Month{i}'] = [0, 0]
    
    # Add Trailing1-Trailing36 columns
    for i in range(1, 37):
        sample_data[f'Trailing{i}'] = [0, 0]
    
    return pd.DataFrame(sample_data)

def main():
    st.title("Sales Ramp Analysis Tool")
    
    # Add main tabs
    tab1, tab2, tab3 = st.tabs(["Documentation", "Analysis", "Generate Sample Data"])
    
    with tab1:
        st.header("Documentation")
        
        # Add table of contents
        st.markdown("""
        ## Table of Contents
        1. [Overview](#overview)
        2. [Getting Started](#getting-started)
        3. [Data Requirements](#data-requirements)
        4. [Features](#features)
        5. [Analysis Methods](#analysis-methods)
        6. [Sample Data Generation](#sample-data-generation)
        7. [Troubleshooting](#troubleshooting)
        8. [Best Practices](#best-practices)
        """)
        
        # Create expandable sections for each major topic
        with st.expander("Overview", expanded=True):
            st.markdown("""
            The Sales Ramp Analysis Tool is an application designed to analyze and visualize 
            sales representative productivity data. It helps organizations understand ramp times, performance 
            patterns, and  effectiveness across different segments and regions.

            ### Key Functionality
            - Analyze rep ramp performance
            - Appy various regression models to determine ramp periods and target productivity
            - Generate sample data to experiment with the tool
            """)

        with st.expander("Getting Started"):
            st.markdown("""
            ### Prerequisites
            - Sales performance data in CSV format
            - Monthly booking values for each sales representative
            - Territory and segment classifications

            ### Quick Start
            1. Download the template CSV from the sidebar
            2. Format your data according to the template
            3. Upload your CSV file
            4. Select your analysis parameters
            5. Explore the visualizations and metrics
            """)

        with st.expander("Data Requirements"):
            st.markdown("""
            ### Required Columns
            - **EID**: Unique identifier for each sales representative
            - **StartDate**: Rep's start date (YYYY-MM-DD format)
            - **Market**: Geographic market designation
            - **Theater**: Sub-market designation
            - **Region**: Regional designation
            - **Segment**: Business segment (e.g., Commercial, Enterprise, Majors)
            - **Territory_Profile**: Territory type (e.g., Acquisition, Expansion)
            - **Month1-Month36**: Monthly booking values
            - **Trailing1-Trailing36**: Trailing 12-month booking values

            ### Data Format Guidelines
            - All monetary values should be in the same currency
            - Dates must be in YYYY-MM-DD format
            - No commas in numeric values
            - Missing values should be left empty or marked as 0
            - Text fields should not contain special characters
            """)

        with st.expander("Analysis Methods"):
            st.markdown("""
            ### Growth Models

            #### Logistic Model
            - S-shaped curve modeling
            - Best for typical ramp patterns
            - Formula: y = L / (1 + exp(-k(x - x0)))
            - Parameters:
                - L: Maximum achievement level
                - k: Growth rate
                - x0: Midpoint of growth

            #### Gompertz Model
            - Asymmetric growth curve
            - Useful for accelerated ramp patterns
            - Formula: y = a * exp(-b * exp(-cx))
            - Parameters:
                - a: Asymptote
                - b: Displacement
                - c: Growth rate

            #### Linear Model
            - Simple linear progression
            - Best for steady growth patterns
            - Formula: y = ax + b
            - Parameters:
                - a: Growth rate
                - b: Initial value

            ### Metrics Calculation
            - Ramp time to target percentage
            - R-squared value for model fit
            - Root Mean Square Error (RMSE)
            - Target value achievement
            """)

        with st.expander("Sample Data Generation"):
            st.markdown("""
            ### Configuration Options
            - Number of sales representatives
            - Segment parameters
                - Annual targets
                - Ramp periods
            - Noise level for variation
            - Random seed for reproducibility

            ### Segment Parameters
            #### Commercial
            - Default annual target: $900,000
            - Typical ramp period: 6 months
            - Faster ramp, lower target

            #### Enterprise
            - Default annual target: $1,500,000
            - Typical ramp period: 9 months
            - Moderate ramp, medium target

            #### Majors
            - Default annual target: $2,500,000
            - Typical ramp period: 12 months
            - Slower ramp, higher target
            """)

        with st.expander("Troubleshooting"):
            st.markdown("""
            ### Common Issues
            1. **Data Upload Errors**
                - Check CSV format
                - Verify column names match template
                - Ensure date format is correct
                - Remove special characters

            2. **Analysis Errors**
                - Verify data completeness
                - Check for outliers
                - Ensure sufficient data points
                - Validate segment assignments

            3. **Visualization Issues**
                - Refresh browser
                - Clear cache
                - Reduce data size if too large
                - Check for missing values
            """)

        with st.expander("Best Practices"):
            st.markdown("""
            ### Data Preparation
            1. Clean and validate data before upload
            2. Use consistent naming conventions
            3. Remove inactive or incomplete records
            4. Verify territory assignments

            ### Analysis Configuration
            1. Start with default parameters
            2. Adjust target percentages based on business goals
            3. Compare multiple growth models
            4. Use appropriate date ranges

            ### Interpretation Guidelines
            1. Consider market conditions
            2. Account for seasonality
            3. Compare similar segments
            4. Look for patterns across territories

            For additional support or feature requests, please contact your system administrator 
            or the development team.
            """)

    with tab2:
        # Sidebar for analysis tab
        st.sidebar.header("📊 Sales Ramp Analysis")
        
        # Template download
        st.sidebar.markdown("### 📥 Get Started")
        template_df = create_sample_template()
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            "Download Template CSV",
            template_csv,
            "sales_ramp_template.csv",
            "text/csv",
            help="Download a sample CSV template with the correct columns"
        )
        
        # File upload
        st.sidebar.markdown("### 📤 Upload Data")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your sales data CSV", 
            type=['csv'],
            help="Upload a CSV file following the template format",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Load and prepare data
            df = pd.read_csv(uploaded_file)
            
            # Data validation in collapsible section
            with st.expander("📋 Data Validation Results", expanded=False):
                processed_df, month_cols, trailing_cols = prepare_data(df)
                
                # Add summary after data preparation
                st.markdown("---")
                st.markdown("### Quick Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reps", len(processed_df))
                with col2:
                    st.metric("Segments", len(processed_df['Segment'].unique()))
                with col3:
                    st.metric("Markets", len(processed_df['Market'].unique()))
            
            # Analysis options
            st.write("### Analysis Options")
            
            # Create two columns for filters
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                st.write("#### Data Filters")
                selected_markets = st.multiselect(
                    "Markets",
                    options=processed_df['Market'].unique(),
                    default=None
                )
                
                selected_theaters = st.multiselect(
                    "Theaters",
                    options=processed_df['Theater'].unique(),
                    default=None
                )
                
                selected_regions = st.multiselect(
                    "Regions",
                    options=processed_df['Region'].unique(),
                    default=None
                )
                
                selected_segments = st.multiselect(
                    "Segments",
                    options=processed_df['Segment'].unique(),
                    default=processed_df['Segment'].unique()
                )

            with filter_col2:
                st.write("#### Analysis Parameters")
                model_type = st.selectbox(
                    "Model Type",
                    options=['logistic', 'gompertz', 'linear'],
                    index=0,
                    help="Select the type of growth model to fit"
                )
                
                ramp_target = st.slider(
                    "Ramp Target Percentage",
                    min_value=50,
                    max_value=100,
                    value=90,
                    step=5,
                    help="Target percentage of steady-state performance"
                )
                
                date_range = st.slider(
                    "Analysis Time Range (Months)",
                    min_value=1,
                    max_value=36,
                    value=(1, 36),
                    step=1,
                    help="Select the month range for analysis"
                )
            
            # Performance visualizations
            st.write("### Performance Visualizations")
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Ramp Analysis", "Monthly Trends", "Performance Distribution"])
            
            with viz_tab1:
                fig_ramp, metrics = analyze_ramp(
                    processed_df,
                    segments=selected_segments if len(selected_segments) > 0 else None,
                    markets=selected_markets if len(selected_markets) > 0 else None,
                    theaters=selected_theaters if len(selected_theaters) > 0 else None,
                    regions=selected_regions if len(selected_regions) > 0 else None,
                    model_type=model_type,
                    ramp_target_pct=ramp_target,
                    date_range=date_range
                )
                st.pyplot(fig_ramp)
                
                # Display metrics
                st.write("### Ramp Metrics")
                metrics_df = pd.DataFrame(metrics).T
                st.dataframe(metrics_df)
                
                # Download metrics
                csv = metrics_df.to_csv().encode('utf-8')
                st.download_button(
                    "Download Metrics CSV",
                    csv,
                    "ramp_metrics.csv",
                    "text/csv",
                    key='download-metrics'
                )
            
            with viz_tab2:
                fig_monthly = plot_performance(processed_df, month_cols, 'monthly')
                st.pyplot(fig_monthly)
            
            with viz_tab3:
                fig_dist = plot_performance(processed_df, trailing_cols, 'distribution')
                st.pyplot(fig_dist)

    with tab3:
        st.header("Sample Data Generation")
        
        # Create two columns for general settings
        gen_col1, gen_col2 = st.columns(2)
        
        with gen_col1:
            # Number of reps slider
            num_reps = st.slider(
                "Number of Sales Representatives",
                min_value=10,
                max_value=1000,
                value=500,
                step=10
            )
            
            # Random seed input
            seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=99999,
                value=42,
                help="Set a seed for reproducible data generation"
            )
            
        with gen_col2:
            # Noise level slider
            noise_level = st.slider(
                "Noise Level",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Amount of random variation in the data (0=none, 1=maximum)"
            )
        
        # Segment configuration
        st.subheader("Segment Parameters")
        segments = ['Commercial', 'Enterprise', 'Majors']
        
        segment_params = {}
        for segment in segments:
            st.write(f"### {segment}")
            col1, col2 = st.columns(2)
            
            with col1:
                annual_target = st.number_input(
                    f"{segment} Annual Target ($)",
                    min_value=100000,
                    max_value=10000000,
                    value={'Commercial': 900000, 'Enterprise': 1500000, 'Majors': 2500000}[segment],
                    step=100000
                )
            
            with col2:
                ramp_period = st.slider(
                    f"{segment} Ramp Period (months)",
                    min_value=1,
                    max_value=24,
                    value={'Commercial': 6, 'Enterprise': 9, 'Majors': 12}[segment]
                )
            
            segment_params[segment] = {
                'annual_target': annual_target,
                'ramp_period': ramp_period
            }
        
        if st.button("Generate Sample Data"):
            try:
                with st.spinner("Generating sample data..."):
                    sample_file = generate_sample_data(num_reps, segment_params, seed=seed, noise_level=noise_level)
                    
                    # Read the generated file
                    with open(sample_file, 'rb') as f:
                        st.download_button(
                            "Download Generated Data",
                            f,
                            "generated_sales_data.csv",
                            "text/csv",
                            help="Download the generated sample data as CSV"
                        )
                    
                    st.success("Sample data generated successfully!")
                    
                    # Preview the data
                    df = pd.read_csv(sample_file)
                    st.write("### Preview of Generated Data")
                    st.dataframe(df.head())
                    
                    # Show summary statistics
                    st.write("### Summary Statistics")
                    for segment in segments:
                        segment_df = df[df['Segment'] == segment]
                        st.write(f"**{segment}**")
                        st.write(f"Number of Reps: {len(segment_df)}")
                        ramp_month = segment_params[segment]['ramp_period']
                        avg_booking = segment_df[f'Month{ramp_month}'].mean()
                        st.write(f"Average Monthly Booking at Ramp: ${avg_booking:,.2f}")
            
            except Exception as e:
                st.error(f"Error generating sample data: {str(e)}")
    

if __name__ == "__main__":
    main()