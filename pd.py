import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib

# Set up Matplotlib to work with Streamlit
matplotlib.use('Agg')

def plot_pd_graph(years, pd_values, extrapolate_years):
    # Transform the years using the natural logarithm
    ln_years = np.log(years)

    # Add a constant to the independent variable matrix (for the intercept)
    X = sm.add_constant(ln_years)

    # Perform the regression
    model = sm.OLS(pd_values, X)
    results = model.fit()

    # Get the estimated coefficients
    intercept, slope = results.params
    r_squared = results.rsquared

    # Extrapolate PD values up to the specified extrapolation year
    extrapolate_years_array = np.arange(1, extrapolate_years + 1)
    ln_extrapolate_years = np.log(extrapolate_years_array)
    extrapolated_pd = intercept + slope * ln_extrapolate_years
    extrapolated_pd_percent = extrapolated_pd * 100

    # Create a DataFrame for the extrapolated PD values
    extrapolation_df = pd.DataFrame({
        'Years': extrapolate_years_array,
        'Extrapolated PD (%)': extrapolated_pd_percent
    })

    # Plot the results
    plt.figure(figsize=(14, 8))  # Increased figure size
    plt.style.use('seaborn-dark')  # Use a style with a dark grid background

    # Plot the data
    plt.plot(extrapolation_df['Years'], extrapolation_df['Extrapolated PD (%)'], marker='o', linestyle='-', color='b', label='Extrapolated PD', linewidth=2)
    plt.scatter(extrapolate_years, extrapolation_df[extrapolation_df['Years'] == extrapolate_years]['Extrapolated PD (%)'].values[0], color='red', zorder=5, s=100, edgecolor='k')  # Larger marker
    plt.text(extrapolate_years, extrapolation_df[extrapolation_df['Years'] == extrapolate_years]['Extrapolated PD (%)'].values[0], f'{extrapolation_df[extrapolation_df["Years"] == extrapolate_years]["Extrapolated PD (%)"].values[0]:.2f}%', fontsize=14, verticalalignment='bottom', horizontalalignment='right', color='red', weight='bold')

    # Labeling
    plt.xlabel('Years', fontsize=14)
    plt.ylabel('Extrapolated PD (%)', fontsize=14)
    plt.title(f'Extrapolated Probability of Default (PD) Over {extrapolate_years} Years', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add the regression equation and R-squared to the plot
    equation_text = f'PD = {slope:.4f} * ln(Years) + {intercept:.4f}\n$R^2$ = {r_squared:.4f}'
    plt.text(1, max(extrapolated_pd_percent) * 0.85, equation_text, fontsize=14, bbox=dict(facecolor='white', alpha=0.7))

    # Add a legend
    plt.legend(fontsize=12)

    # Save the plot to a BytesIO object
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return buf


# Streamlit app
st.title('PD% Extrapolation')

# Input fields for PD% values
pd_values_input = []
for year in range(1, 6):
    pd_value = st.number_input(f'Enter PD% for Year {year}', format="%.4f", min_value=0.0, max_value=100.0, step=0.01)
    pd_values_input.append(pd_value / 100.0)  # Convert to decimal format

# Input field for extrapolation year
extrapolation_year = st.number_input('Enter number of years to extrapolate', min_value=1, max_value=50, value=20)

# Generate the plot if inputs are provided
if len(pd_values_input) == 5 and any(pd_values_input):
    years = np.array([1, 2, 3, 4, 5])
    pd_values = np.array(pd_values_input)

    # Plot the graph
    buf = plot_pd_graph(years, pd_values, extrapolation_year)

    # Display the plot
    st.image(buf, use_column_width=True)
else:
    st.write('Please enter PD% values for all five years.')
