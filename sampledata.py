import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def calculate_monthly_values(annual_target, ramp_period):
    """
    Calculate monthly values using a logistic function that plateaus at target rate
    """
    target_monthly = annual_target / 12
    months = np.arange(1, 37)
    
    # Logistic function parameters
    k = 6.0 / ramp_period  # Controls steepness
    x0 = ramp_period / 2   # Midpoint of the curve
    
    # Generate S-curve that goes from 0 to target_monthly
    monthly_values = target_monthly / (1 + np.exp(-k * (months - x0)))
    
    # Scale the values so that sum at ramp period equals prorated target
    prorated_target = (annual_target / 12) * ramp_period
    scale_factor = prorated_target / monthly_values[:ramp_period].sum()
    monthly_values = monthly_values * scale_factor
    
    # Ensure we don't exceed target_monthly in later months
    monthly_values[monthly_values > target_monthly] = target_monthly
    
    return monthly_values

def add_noise(value, noise_level=0.50):
    """Add random noise to a value"""
    return value * (1 + (random.random() - 0.5) * noise_level)

def generate_sales_data(num_reps=500, segment_params=None):
    """
    Generate sample sales data.
    
    Parameters:
    num_reps (int): Number of sales representatives to generate
    segment_params (dict): Dictionary containing parameters for each segment
        Format: {
            'segment_name': {
                'annual_target': float,
                'ramp_period': int
            }
        }
    """
    # Constants
    start_date = datetime(2022, 1, 1)
    markets = ['AMS', 'EMEA', 'APJ']
    theaters = [f'Market-{chr(65+i)}' for i in range(26)]  # Market-A through Market-Z
    regions = [f'Theater-{chr(65+i)}' for i in range(26)]  # Theater-A through Theater-Z
    territory_profiles = ['Acquisition', 'Expansion', 'Retention']  # Added territory profiles

    # Use default parameters if none provided
    if segment_params is None:
        segment_params = {
            'Commercial': {'annual_target': 900000, 'ramp_period': 6},
            'Enterprise': {'annual_target': 1500000, 'ramp_period': 9},
            'Majors': {'annual_target': 2500000, 'ramp_period': 12}
        }

    data = []
    
    for i in range(num_reps):
        segment = random.choice(list(segment_params.keys()))
        params = segment_params[segment]
        
        rep = {
            'EID': f'SR{str(i+1).zfill(4)}',
            'StartDate': (start_date + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
            'Market': random.choice(markets),
            'Theater': random.choice(theaters),
            'Region': random.choice(regions),
            'Segment': segment,
            'Territory_Profile': random.choice(territory_profiles)
        }

        # Get base monthly values for this segment
        monthly_values = calculate_monthly_values(
            params['annual_target'],
            params['ramp_period']
        )

        # Add noise and store values
        for month in range(36):
            value = max(0, round(add_noise(monthly_values[month])))
            rep[f'Month{month + 1}'] = value

        # Calculate trailing values without annualization
        for month in range(36):
            lookback = min(12, month + 1)
            lookback_values = [rep[f'Month{i+1}'] for i in range(max(0, month - 11), month + 1)]
            sum_value = sum(lookback_values)
            rep[f'Trailing{month + 1}'] = round(sum_value)

        data.append(rep)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    month_cols = [f'Month{i+1}' for i in range(36)]
    trailing_cols = [f'Trailing{i+1}' for i in range(36)]
    col_order = ['EID', 'StartDate', 'Market', 'Theater', 'Region', 'Segment', 'Territory_Profile'] + month_cols + trailing_cols
    df = df[col_order]
    
    return df

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate data for 500 reps
    df = generate_sales_data(num_reps=500)
    
    # Save to CSV
    df.to_csv('sales_data.csv', index=False)