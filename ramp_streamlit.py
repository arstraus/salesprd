import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import tempfile
from sampledata import generate_sales_data

st.set_page_config(page_title="Sales Ramp Analysis Tool", page_icon="📈", layout="wide")

def linear_regression(x, a, b):
    return a * x + b

def logistic_function(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def gompertz_function(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))

def get_model_function(model_type):
    models = {'linear': linear_regression, 'logistic': logistic_function, 'gompertz': gompertz_function}
    return models.get(model_type)

def get_initial_params(model_type, y_data, x_data):
    if model_type == 'linear':
        return [(y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0]), y_data[0]]
    elif model_type == 'gompertz':
        return [max(y_data) * 1.2, 5, 0.2]
    return [max(y_data), 0.2, np.mean(x_data)]

def calculate_ramp_time(L, k, x0, ramp_pct):
    p = ramp_pct / 100
    return x0 - np.log(1/p - 1)/k

def prepare_data(df, verbose=True):
    categorical_cols = {'EID': str, 'Market': str, 'Theater': str, 'Region': str, 
                       'Segment': str, 'Territory_Profile': str}
    month_cols = [f'Month{i}' for i in range(1, 37)]
    trailing_cols = [f'Trailing{i}' for i in range(1, 37)]
    
    df = df.copy()
    
    for col, dtype in categorical_cols.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
            if verbose:
                st.write(f"{col}: {df[col].nunique()} unique values")
    
    if 'StartDate' in df.columns:
        df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
        if verbose:
            null_dates = df['StartDate'].isnull().sum()
            if null_dates > 0:
                st.warning(f"{null_dates} rows have invalid dates")
            st.write(f"Date range: {df['StartDate'].min()} to {df['StartDate'].max()}")
    
    for col in month_cols + trailing_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    numeric_cols = [col for col in month_cols + trailing_cols if col in df.columns]
    initial_rows = len(df)
    df = df[df[numeric_cols].any(axis=1)]
    
    if verbose:
        st.write(f"Removed {initial_rows - len(df)} rows with all zero values")
        st.write(f"Final dataset shape: {df.shape}")
        st.write(f"Average monthly booking: ${df[month_cols].mean().mean():,.2f}")
        st.write(f"Average trailing booking: ${df[trailing_cols].mean().mean():,.2f}")
    
    return df, month_cols, trailing_cols

def analyze_ramp(df, segments=None, markets=None, theaters=None, regions=None, 
               territories=None, date_range=(1, 36), ramp_target_pct=90, model_type='logistic'):
    month_cols = [f'Month{i}' for i in range(1, 37)]
    trailing_cols = [f'Trailing{i}' for i in range(1, 37)]
    selected_cols = trailing_cols[date_range[0]-1:date_range[1]]
    
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
        
        for _, row in segment_df.iterrows():
            plt.plot(range(date_range[0], date_range[1]+1), row[selected_cols], alpha=0.1)
        
        plt.plot(range(date_range[0], date_range[1]+1), mean_performance, 
                linewidth=3, label=f'{segment} Mean')
        
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
            plt.plot(x_smooth, y_smooth, '--', linewidth=2, 
                    label=f'{segment} {model_type.title()} Fit')
            
            if model_type == 'logistic':
                ramp_time = calculate_ramp_time(popt[0], popt[1], popt[2], ramp_target_pct)
            elif model_type == 'gompertz':
                a, b, c = popt
                target = (ramp_target_pct/100) * a
                ramp_time = (-1/c) * np.log(-np.log(target/a)/b)
            else:
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
            
            plt.plot(ramp_time, ramp_value, 'o', alpha=0.8, markersize=10)
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
    df = generate_sales_data(num_reps=num_reps, segment_params=segment_params, 
                           seed=seed, noise_level=noise_level)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        return tmp_file.name

def create_sample_template():
    sample_data = {
        'EID': ['SR001', 'SR002'],
        'StartDate': ['2023-01-01', '2023-01-01'],
        'Market': ['SampleMarket', 'SampleMarket'],
        'Theater': ['SampleTheater', 'SampleTheater'],
        'Region': ['SampleRegion', 'SampleRegion'],
        'Segment': ['Commercial', 'Enterprise'],
        'Territory_Profile': ['Acquisition', 'Expansion']
    }
    
    for i in range(1, 37):
        sample_data[f'Month{i}'] = [0, 0]
        sample_data[f'Trailing{i}'] = [0, 0]
    
    return pd.DataFrame(sample_data)

def plot_distribution(df, trailing_month, bin_size=50):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the trailing month data
    col_name = f'Trailing{trailing_month}'
    data = df[col_name]
    
    # Calculate number of bins based on data range and bin size
    data_range = data.max() - data.min()
    n_bins = int(data_range / bin_size)
    
    # Create histogram
    plt.hist(data, bins=n_bins, density=True, alpha=0.7, color='skyblue')
    plt.hist(data, bins=n_bins, density=True, histtype='step', color='navy')
    
    # Add labels and title
    plt.title(f'Distribution of Trailing {trailing_month}-Month Bookings\nBin Size: ${bin_size:,.0f}')
    plt.xlabel('Trailing Bookings ($)')
    plt.ylabel('Frequency')
    
    # Format x-axis to show currency
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def main():
    st.title("Sales Ramp Analysis Tool")
    tab1, tab2, tab3 = st.tabs(["Documentation", "Analysis", "Generate Sample Data"])
    
    with tab1:
        st.header("Documentation")
        
        st.markdown("""
        ## Overview
        The Sales Ramp Analysis Tool helps analyze and visualize sales representative 
        productivity data. It provides insights into ramp times, performance patterns, 
        and effectiveness across different segments and regions.

        ### Key Features
        - Multiple growth model analysis (Linear, Logistic, Gompertz)
        - Interactive data visualization 
        - Customizable analysis parameters
        - Automated data validation
        - Sample data generation
        
        ### Data Requirements
        Upload a CSV file with the following columns:
        - EID: Unique identifier for each sales representative
        - StartDate: Rep's start date (YYYY-MM-DD format)
        - Market: Geographic market designation
        - Theater: Sub-market designation
        - Region: Regional designation
        - Segment: Business segment (e.g., Commercial, Enterprise)
        - Territory_Profile: Territory type
        - Month1-Month36: Monthly booking values
        - Trailing1-Trailing36: Trailing 12-month booking values
        
        ### Analysis Methods
        The tool provides three types of growth models:
        1. **Logistic Model**: S-shaped curve modeling (best for typical ramp patterns)
        2. **Gompertz Model**: Asymmetric growth curve (useful for accelerated patterns)
        3. **Linear Model**: Simple linear progression (best for steady growth)
        
        ### Getting Started
        1. Download the template CSV from the sidebar
        2. Format your data according to the template
        3. Upload your CSV file
        4. Select your analysis parameters
        5. Explore the visualizations and metrics
        """)
    
    with tab2:
        st.sidebar.header("📊 Sales Ramp Analysis")
        if 'uploaded_file' not in locals():
            st.info("👈 Please upload your sales data CSV file using the sidebar to begin analysis")
        st.sidebar.markdown("### 📥 Get Started")
        template_df = create_sample_template()
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Download Template CSV", template_csv, 
                                 "sales_ramp_template.csv", "text/csv")
        
        st.sidebar.markdown("### 📤 Upload Data")
        uploaded_file = st.sidebar.file_uploader("Upload your sales data CSV", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            with st.expander("📋 Data Validation Results", expanded=False):
                processed_df, month_cols, trailing_cols = prepare_data(df)
                st.markdown("---")
                st.markdown("### Quick Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reps", len(processed_df))
                with col2:
                    st.metric("Segments", len(processed_df['Segment'].unique()))
                with col3:
                    st.metric("Markets", len(processed_df['Market'].unique()))
            
            st.write("### Analysis Options")
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                st.write("#### Data Filters")
                selected_markets = st.multiselect("Markets", options=processed_df['Market'].unique())
                selected_theaters = st.multiselect("Theaters", options=processed_df['Theater'].unique())
                selected_regions = st.multiselect("Regions", options=processed_df['Region'].unique())
                selected_segments = st.multiselect("Segments", options=processed_df['Segment'].unique(), 
                                                 default=processed_df['Segment'].unique())

            with filter_col2:
                st.write("#### Analysis Parameters")
                model_type = st.selectbox("Model Type", 
                                        options=['logistic', 'gompertz', 'linear'])
                ramp_target = st.slider("Ramp Target Percentage", 50, 100, 90, 5)
                date_range = st.slider("Analysis Time Range (Months)", 1, 36, (1, 36))
            
            viz_tab1, viz_tab2 = st.tabs(["Ramp Analysis", "Distribution Analysis"])
            
            with viz_tab1:
                st.write("### Performance Visualization")
                fig_ramp, metrics = analyze_ramp(processed_df, selected_segments, selected_markets,
                                               selected_theaters, selected_regions, 
                                               model_type=model_type, ramp_target_pct=ramp_target,
                                               date_range=date_range)
                if fig_ramp is not None:
                    st.pyplot(fig_ramp)
                    st.write("### Ramp Metrics")
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(metrics_df)
                    csv = metrics_df.to_csv().encode('utf-8')
                    st.download_button("Download Metrics CSV", csv, "ramp_metrics.csv", "text/csv")
            
            with viz_tab2:
                st.write("### Distribution Analysis")
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    trailing_month = st.slider("Select Trailing Month", 1, 36, 12)
                
                with dist_col2:
                    # Calculate a reasonable default bin size based on data range
                    col_name = f'Trailing{trailing_month}'
                    data_range = processed_df[col_name].max() - processed_df[col_name].min()
                    default_bin_size = int(data_range / 50)  # Default to 50 bins
                    bin_size = st.number_input("Bin Size ($)", 
                                             min_value=100,
                                             max_value=int(data_range),
                                             value=default_bin_size,
                                             step=100)
                
                # Apply filters
                filtered_df = processed_df.copy()
                if selected_markets:
                    filtered_df = filtered_df[filtered_df['Market'].isin(selected_markets)]
                if selected_theaters:
                    filtered_df = filtered_df[filtered_df['Theater'].isin(selected_theaters)]
                if selected_regions:
                    filtered_df = filtered_df[filtered_df['Region'].isin(selected_regions)]
                if selected_segments:
                    filtered_df = filtered_df[filtered_df['Segment'].isin(selected_segments)]
                
                if len(filtered_df) > 0:
                    fig_dist = plot_distribution(filtered_df, trailing_month, bin_size)
                    st.pyplot(fig_dist)
                    
                    # Add summary statistics
                    st.write("### Distribution Statistics")
                    col_name = f'Trailing{trailing_month}'
                    stats = {
                        'Mean': f"${filtered_df[col_name].mean():,.2f}",
                        'Median': f"${filtered_df[col_name].median():,.2f}",
                        'Standard Deviation': f"${filtered_df[col_name].std():,.2f}",
                        'Minimum': f"${filtered_df[col_name].min():,.2f}",
                        'Maximum': f"${filtered_df[col_name].max():,.2f}"
                    }
                    st.dataframe(pd.DataFrame([stats]).T.rename(columns={0: 'Value'}))

    with tab3:
        st.header("Sample Data Generation")
        gen_col1, gen_col2 = st.columns(2)
        
        with gen_col1:
            num_reps = st.slider("Number of Sales Representatives", 10, 1000, 500, 10)
            seed = st.number_input("Random Seed", 1, 99999, 42)
            
        with gen_col2:
            noise_level = st.slider("Noise Level", 0.0, 1.0, 0.5, 0.05)
        
        st.subheader("Segment Parameters")
        segments = ['Commercial', 'Enterprise', 'Majors']
        
        segment_params = {}
        for segment in segments:
            st.write(f"### {segment}")
            col1, col2 = st.columns(2)
            
            with col1:
                annual_target = st.number_input(
                    f"{segment} Annual Target ($)",
                    100000,
                    10000000,
                    {'Commercial': 900000, 'Enterprise': 1500000, 'Majors': 2500000}[segment],
                    100000
                )
            
            with col2:
                ramp_period = st.slider(
                    f"{segment} Ramp Period (months)",
                    1,
                    24,
                    {'Commercial': 6, 'Enterprise': 9, 'Majors': 12}[segment]
                )
            
            segment_params[segment] = {
                'annual_target': annual_target,
                'ramp_period': ramp_period
            }
        
        if st.button("Generate Sample Data"):
            try:
                with st.spinner("Generating sample data..."):
                    sample_file = generate_sample_data(num_reps, segment_params, seed, noise_level)
                    
                    with open(sample_file, 'rb') as f:
                        st.download_button(
                            "Download Generated Data",
                            f,
                            "generated_sales_data.csv",
                            "text/csv"
                        )
                    
                    st.success("Sample data generated successfully!")
                    
                    df = pd.read_csv(sample_file)
                    st.write("### Preview of Generated Data")
                    st.dataframe(df.head())
                    
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