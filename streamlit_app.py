import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:',  # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file."""
    DATA_FILENAME = Path(__file__).parent / 'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    gdp_df = raw_gdp_df.melt(
        ['Country Code', 'Country Name'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website.
'''

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

# Dropdown for countries with names
country_mapping = gdp_df[['Country Code', 'Country Name']].drop_duplicates()
countries = country_mapping['Country Name'] + ' (' + country_mapping['Country Code'] + ')'

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['Germany (DEU)', 'France (FRA)', 'United Kingdom (GBR)']
)

selected_country_codes = [
    country.split('(')[-1].strip(')') for country in selected_countries
]

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_country_codes))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

# Line chart for GDP over time
if not filtered_gdp_df.empty:
    st.line_chart(
        filtered_gdp_df,
        x='Year',
        y='GDP',
        color='Country Code',
    )
else:
    st.warning('No data available for the selected filters.')

''
''

# Add a table view
if st.checkbox('Show raw data table'):
    st.dataframe(filtered_gdp_df)

# Add download button
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

if not filtered_gdp_df.empty:
    csv_data = convert_df_to_csv(filtered_gdp_df)

    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name='filtered_gdp_data.csv',
        mime='text/csv',
    )

''
''

# Display GDP metrics for the selected countries
first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

cols = st.columns(4)

for i, country in enumerate(selected_country_codes):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000 if not first_year.empty else None
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000 if not last_year.empty else None

        if first_gdp is None or math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B' if last_gdp else 'n/a',
            delta=growth,
            delta_color=delta_color
        )

''
''

# Show average GDP for selected years
avg_gdp = filtered_gdp_df.groupby('Country Code')['GDP'].mean().reset_index()
avg_gdp['GDP'] = avg_gdp['GDP'] / 1e9  # Convert to billions

if not avg_gdp.empty:
    st.subheader('Average GDP (in Billions)')
    st.dataframe(avg_gdp.rename(columns={'Country Code': 'Country', 'GDP': 'Average GDP (B)'}))

# Add a bar chart for GDP growth
if not first_year.empty and not last_year.empty:
    gdp_growth = last_year.set_index('Country Code')['GDP'] / first_year.set_index('Country Code')['GDP']
    gdp_growth.dropna(inplace=True)
    st.bar_chart(gdp_growth.rename('GDP Growth'))

''
''

# Add a comparison heatmap for GDP trends
import seaborn as sns
import matplotlib.pyplot as plt
if len(selected_country_codes) > 1:
    heatmap_data = filtered_gdp_df.pivot(index='Year', columns='Country Code', values='GDP')
    if not heatmap_data.empty:
        st.subheader('GDP Heatmap Comparison')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap='YlGnBu', cbar=True, ax=ax)
        st.pyplot(fig)

# Add GDP forecast using linear regression
from sklearn.linear_model import LinearRegression
import numpy as np
if len(selected_country_codes) == 1:
    st.subheader('GDP Forecast for Selected Country')
    country_code = selected_country_codes[0]
    country_data = filtered_gdp_df[filtered_gdp_df['Country Code'] == country_code]

    if not country_data.empty:
        model = LinearRegression()
        X = country_data['Year'].values.reshape(-1, 1)
        y = country_data['GDP'].values
        model.fit(X, y)

        future_years = np.arange(max_value + 1, max_value + 11).reshape(-1, 1)
        future_gdp = model.predict(future_years)

        forecast_df = pd.DataFrame({
            'Year': future_years.flatten(),
            'Forecasted GDP': future_gdp
        })

        st.line_chart(forecast_df, x='Year', y='Forecasted GDP')

# Add GDP percentile ranking
st.subheader('GDP Percentile Ranking for Selected Countries')
percentile_data = last_year[['Country Code', 'GDP']].dropna()
percentile_data['Percentile'] = percentile_data['GDP'].rank(pct=True) * 100

if not percentile_data.empty:
    st.dataframe(
        percentile_data[percentile_data['Country Code'].isin(selected_country_codes)]
        .sort_values(by='Percentile', ascending=False)
        .rename(columns={'Country Code': 'Country', 'Percentile': 'Percentile Rank (%)'})
    )
