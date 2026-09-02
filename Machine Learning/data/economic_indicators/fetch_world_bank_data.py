"""
Script to fetch economic indicators from the World Bank API

This script downloads key economic and social indicators for a selection of countries
to be used in PCA analysis and clustering for the CEU Machine Learning course.

The output CSV includes:
- Economic/social indicators (GDP, inflation, unemployment, etc.)
- Categorical features: Income_Level, Region (14 categories), Region_Broad (6 categories)
- Binary flags: is_OECD, is_EU, is_G7, is_G20, is_BRICS

CATEGORICAL FEATURES (mutually exclusive & complete - each country has exactly one value):
  
  Income_Level (4 categories):
    - High_Income
    - Upper_Middle_Income
    - Lower_Middle_Income
    - Low_Income
  
  Region (14 detailed categories):
    - Western_Europe, Northern_Europe, Southern_Europe, Eastern_Europe
    - North_America, Latin_America
    - East_Asia, Southeast_Asia, South_Asia, Central_Asia
    - Oceania, Middle_East, North_Africa, Sub_Saharan_Africa
  
  Region_Broad (6 broader categories):
    - Europe (combines Western, Northern, Southern, Eastern)
    - Americas (combines North America, Latin America)
    - Asia (combines East, Southeast, South, Central)
    - Africa (combines North Africa, Sub-Saharan)
    - Middle_East
    - Oceania

BINARY FLAGS (non-exclusive - countries can have multiple):
  - is_OECD, is_EU, is_G7, is_G20, is_BRICS

Example usage in notebooks:
    # Load data
    df = pd.read_csv('countries_wb_2020_2023.csv').set_index('Country')
    
    # Filter by categorical features
    high_income = df[df['Income_Level'] == 'High_Income']
    european = df[df['Region_Broad'] == 'Europe']
    asian_emerging = df[(df['Region_Broad'] == 'Asia') & (df['Income_Level'] == 'Upper_Middle_Income')]
    
    # Filter by binary flags
    oecd_countries = df[df['is_OECD'] == 1]
    
    # Use for clustering validation
    true_labels_income = df['Income_Level']
    true_labels_region = df['Region_Broad']
    
    # Get only numeric indicators for PCA/clustering
    indicator_cols = [col for col in df.columns if not col.startswith('is_') and 
                      col not in ['Income_Level', 'Region', 'Region_Broad']]
    X = df[indicator_cols]

Data source: World Bank Open Data (https://data.worldbank.org/)
API documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""

import pandas as pd
import requests
import time

# Define the indicators we want to fetch
# Format: 'indicator_code': 'readable_name'
INDICATORS = {
    'NY.GDP.PCAP.CD': 'GDP_per_capita',              # GDP per capita (current US$)
    'FP.CPI.TOTL.ZG': 'Inflation_rate',              # Inflation, consumer prices (annual %)
    'SL.UEM.TOTL.ZS': 'Unemployment_rate',           # Unemployment, total (% of labor force)
    'NE.TRD.GNFS.ZS': 'Trade_percent_GDP',           # Trade (% of GDP)
    'SH.XPD.CHEX.GD.ZS': 'Health_spending_pct_GDP',  # Current health expenditure (% of GDP)
    'SP.DYN.LE00.IN': 'Life_expectancy',             # Life expectancy at birth (years)
    'IT.NET.USER.ZS': 'Internet_users_pct',          # Individuals using the Internet (% of population)
    'EN.ATM.CO2E.PC': 'CO2_emissions_per_capita',    # CO2 emissions (metric tons per capita)
    'GB.XPD.RSDV.GD.ZS': 'RD_spending_pct_GDP',      # Research and development expenditure (% of GDP)
    'BX.KLT.DINV.WD.GD.ZS': 'FDI_pct_GDP',           # Foreign direct investment, net inflows (% of GDP)
    'SE.SEC.ENRR': 'School_enrollment_secondary_pct' # School enrollment, secondary (% gross)
}

# Country categories for filtering and clustering analysis
# 
# CATEGORICAL FEATURES (stored as text, mutually exclusive & complete):
# - Income_Level: High_Income, Upper_Middle_Income, Lower_Middle_Income, Low_Income
# - Region: 14 detailed geographic regions
# - Region_Broad: 6 broader geographic regions (Europe, Americas, Asia, Africa, Middle_East, Oceania)
#
# BINARY FLAGS (stored as 0/1, non-exclusive):
# - Political/Economic groups: OECD, EU, G7, G20, BRICS

# Income levels (mutually exclusive and complete - World Bank classification)
INCOME_LEVELS = {
    'High_Income': ['USA', 'CAN', 'GBR', 'DEU', 'FRA', 'ITA', 'ESP', 'NLD', 'BEL', 'AUT', 
                    'SWE', 'NOR', 'DNK', 'FIN', 'CHE', 'IRL', 'PRT', 'GRC', 'ISL', 'LUX',
                    'JPN', 'KOR', 'SGP', 'AUS', 'NZL', 'ISR', 'SAU', 'ARE', 'QAT', 
                    'OMN', 'URY', 'CHL', 'CRI', 'CYP', 'MLT'],
    'Upper_Middle_Income': ['BRA', 'MEX', 'ARG', 'THA', 'MYS', 'COL', 
                            'PER', 'ZAF', 'IRN', 'KAZ', 'AZE', 'GEO',
                            'POL', 'CZE', 'HUN', 'ROU', 'BGR', 'HRV', 'SVK', 'SVN', 'EST', 
                            'LVA', 'LTU', 'SRB', 'PRY'],
    'Lower_Middle_Income': ['IND', 'IDN', 'VNM', 'PAK', 'EGY', 'KEN', 
                            'UKR', 'UZB', 'GTM', 'LKA'],
    'Low_Income': [],
}

# Geographic regions (mutually exclusive and complete)
REGIONS = {
    'Western_Europe': ['GBR', 'DEU', 'FRA', 'NLD', 'BEL', 'AUT', 'CHE', 'IRL', 'LUX'],
    'Northern_Europe': ['SWE', 'NOR', 'DNK', 'FIN', 'ISL'],
    'Southern_Europe': ['ITA', 'ESP', 'PRT', 'GRC', 'CYP', 'MLT'],
    'Eastern_Europe': ['POL', 'CZE', 'HUN', 'ROU', 'BGR', 'HRV', 'SVK', 'SVN', 'EST', 
                       'LVA', 'LTU', 'SRB', 'UKR'],
    'North_America': ['USA', 'CAN'],
    'Latin_America': ['MEX', 'BRA', 'ARG', 'CHL', 'COL', 'PER', 'URY', 
                      'PRY', 'CRI', 'GTM'],
    'East_Asia': ['JPN', 'KOR'],
    'Southeast_Asia': ['SGP', 'IDN', 'THA', 'MYS', 'VNM'],
    'South_Asia': ['IND', 'PAK', 'LKA'],
    'Central_Asia': ['KAZ', 'UZB', 'AZE', 'GEO'],
    'Oceania': ['AUS', 'NZL'],
    'Middle_East': ['SAU', 'ARE', 'ISR', 'QAT', 'OMN', 'IRN'],
    'North_Africa': ['EGY'],
    'Sub_Saharan_Africa': ['ZAF', 'KEN'],
}

# Political and economic groupings (non-exclusive - countries can belong to multiple)
POLITICAL_ECONOMIC_GROUPS = {
    'OECD': ['USA', 'CAN', 'MEX', 'GBR', 'DEU', 'FRA', 'ITA', 'ESP', 'NLD', 'BEL', 'AUT', 
             'SWE', 'NOR', 'DNK', 'FIN', 'CHE', 'IRL', 'PRT', 'GRC', 'ISL', 'LUX', 
             'POL', 'CZE', 'HUN', 'SVK', 'EST', 'LVA', 'LTU', 'SVN',
             'JPN', 'KOR', 'AUS', 'NZL', 'ISR', 'CHL', 'CRI', 'COL'],
    'EU': ['AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 
           'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 
           'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE'],
    'G7': ['USA', 'CAN', 'GBR', 'DEU', 'FRA', 'ITA', 'JPN'],
    'G20': ['USA', 'CAN', 'GBR', 'DEU', 'FRA', 'ITA', 'ESP', 'JPN', 'KOR', 'AUS', 
            'IND', 'BRA', 'MEX', 'IDN', 'SAU', 'ZAF', 'ARG'],
    'BRICS': ['BRA', 'IND', 'ZAF'],
}

# Mapping from detailed regions to broad regions
REGION_TO_BROAD = {
    'Western_Europe': 'Europe',
    'Northern_Europe': 'Europe',
    'Southern_Europe': 'Europe',
    'Eastern_Europe': 'Europe',
    'North_America': 'Americas',
    'Latin_America': 'Americas',
    'East_Asia': 'Asia',
    'Southeast_Asia': 'Asia',
    'South_Asia': 'Asia',
    'Central_Asia': 'Asia',
    'Oceania': 'Oceania',
    'Middle_East': 'Middle_East',
    'North_Africa': 'Africa',
    'Sub_Saharan_Africa': 'Africa',
}

# Get all unique countries from categories
COUNTRIES = sorted(list(set([country for countries in list(INCOME_LEVELS.values()) + list(REGIONS.values()) for country in countries])))

def fetch_indicator_for_countries(indicator_code, countries, start_year=2020, end_year=2023):
    """
    Fetch a single indicator for all countries using World Bank API.
    
    Parameters:
    -----------
    indicator_code : str
        World Bank indicator code
    countries : list
        List of ISO 3-letter country codes
    start_year : int
        Start year for data range
    end_year : int
        End year for data range
    
    Returns:
    --------
    dict
        Dictionary mapping country codes to values
    """
    # Build API URL - fetch for all countries at once
    countries_str = ';'.join(countries)
    url = f"https://api.worldbank.org/v2/country/{countries_str}/indicator/{indicator_code}"
    
    params = {
        'date': f'{start_year}:{end_year}',
        'format': 'json',
        'per_page': 1000  # Get all results in one request
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # World Bank API returns [metadata, data]
        if len(data) < 2 or data[1] is None:
            return {}
        
        # Extract most recent value for each country
        result = {}
        for entry in data[1]:
            country_code = entry['countryiso3code']
            value = entry['value']
            
            # Only keep if value is not None and we haven't seen this country yet
            # (API returns most recent first, so first non-null value is what we want)
            if value is not None and country_code not in result:
                result[country_code] = value
        
        return result
        
    except Exception as e:
        print(f"  Error fetching {indicator_code}: {e}")
        return {}

def fetch_world_bank_data(countries, indicators, start_year=2020, end_year=2023):
    """
    Fetch all indicators for all countries.
    
    Parameters:
    -----------
    countries : list
        List of ISO 3-letter country codes
    indicators : dict
        Dictionary mapping indicator codes to readable names
    start_year : int
        Start year for data range
    end_year : int
        End year for data range
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with countries as rows and indicators as columns
    """
    print(f"Fetching data for {len(countries)} countries and {len(indicators)} indicators...")
    print(f"Date range: {start_year}-{end_year} (will use most recent available data)")
    print()
    
    all_data = []
    
    for i, (indicator_code, indicator_name) in enumerate(indicators.items(), 1):
        print(f"[{i}/{len(indicators)}] Fetching {indicator_name}...", end=' ')
        
        data = fetch_indicator_for_countries(indicator_code, countries, start_year, end_year)
        
        if data:
            print(f"✓ Got data for {len(data)} countries")
            all_data.append(pd.Series(data, name=indicator_name))
        else:
            print("✗ No data returned")
        
        # Be nice to the API - small delay between requests
        time.sleep(0.2)
    
    if not all_data:
        print("\nNo data was fetched successfully.")
        return None
    
    # Combine all series into a DataFrame
    df = pd.concat(all_data, axis=1)
    df.index.name = 'Country'
    df = df.reset_index()
    
    print(f"\nSuccessfully fetched data for {len(df)} countries")
    print(f"\nMissing values per indicator:")
    for col in df.columns:
        if col != 'Country':
            missing = df[col].isna().sum()
            print(f"  {col}: {missing} missing ({missing/len(df)*100:.1f}%)")
    
    return df

def add_country_categories(df, categories_dict):
    """
    Add binary columns for each country category.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with 'Country' column
    categories_dict : dict
        Dictionary mapping category names to lists of country codes
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added category columns
    """
    df = df.copy()
    
    for category_name, countries_in_category in categories_dict.items():
        df[f'is_{category_name}'] = df['Country'].isin(countries_in_category).astype(int)
    
    return df

def add_categorical_features(df, income_levels, regions, region_to_broad):
    """
    Add categorical features (Income_Level, Region, Region_Broad) to the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with 'Country' column
    income_levels : dict
        Dictionary mapping income levels to country lists
    regions : dict
        Dictionary mapping detailed regions to country lists
    region_to_broad : dict
        Dictionary mapping detailed regions to broad regions
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added categorical columns
    """
    df = df.copy()
    
    # Create country to income level mapping
    country_to_income = {}
    for income_level, countries in income_levels.items():
        for country in countries:
            country_to_income[country] = income_level
    
    # Create country to region mapping
    country_to_region = {}
    for region, countries in regions.items():
        for country in countries:
            country_to_region[country] = region
    
    # Add categorical columns
    df['Income_Level'] = df['Country'].map(country_to_income)
    df['Region'] = df['Country'].map(country_to_region)
    df['Region_Broad'] = df['Region'].map(region_to_broad)
    
    return df

def clean_data(df, max_missing_pct=0.4):
    """
    Clean the fetched data by handling missing values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data from World Bank
    max_missing_pct : float
        Maximum percentage of missing values allowed per country (default: 0.4 = 40%)
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned data
    """
    print(f"\nCleaning data (removing countries with >{max_missing_pct*100:.0f}% missing values)...")
    
    # Calculate missing percentage per country
    missing_pct = df.drop(columns=['Country']).isna().sum(axis=1) / (len(df.columns) - 1)
    
    # Keep countries with acceptable amount of missing data
    countries_to_keep = df[missing_pct <= max_missing_pct].copy()
    
    removed = len(df) - len(countries_to_keep)
    if removed > 0:
        print(f"Removed {removed} countries with >{max_missing_pct*100:.0f}% missing values")
        removed_countries = df[missing_pct > max_missing_pct]['Country'].tolist()
        print(f"Removed: {', '.join(removed_countries)}")
    
    print(f"\nFinal dataset: {len(countries_to_keep)} countries, {len(countries_to_keep.columns)-1} indicators")
    
    # Show remaining missing values
    if countries_to_keep.drop(columns=['Country']).isna().sum().sum() > 0:
        print("\nRemaining missing values will need to be handled during PCA (imputation or feature removal)")
    
    return countries_to_keep

def save_data(df, filename='countries_wb_2020_2023.csv'):
    """
    Save the cleaned data to CSV.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned data
    filename : str
        Output filename
    """
    filepath = f'data/economic_indicators/{filename}'
    
    # Reorder columns: Country, indicators, categorical features, then binary flags
    indicator_cols = [col for col in df.columns if not col.startswith('is_') and 
                     col not in ['Country', 'Income_Level', 'Region', 'Region_Broad']]
    categorical_cols = ['Income_Level', 'Region', 'Region_Broad']
    binary_cols = sorted([col for col in df.columns if col.startswith('is_')])
    column_order = ['Country'] + indicator_cols + categorical_cols + binary_cols
    df = df[column_order]
    
    df.to_csv(filepath, index=False)
    print(f"\n{'='*60}")
    print(f"Data saved to: {filepath}")
    print(f"Shape: {df.shape}")
    print(f"  - Economic indicators: {len(indicator_cols)}")
    print(f"  - Categorical features: {len(categorical_cols)}")
    print(f"  - Binary flags: {len(binary_cols)}")
    print(f"\nFirst few rows:")
    print(df[['Country'] + indicator_cols[:3] + categorical_cols].head(10))
    
    # Show categorical feature distributions
    print(f"\n{'='*60}")
    print("Categorical feature distributions:")
    for col in categorical_cols:
        print(f"\n{col}:")
        counts = df[col].value_counts().sort_index()
        for value, count in counts.items():
            print(f"  {value:.<35} {count:>3} countries")
    
    # Show binary flag membership
    print(f"\n{'='*60}")
    print("Binary flag membership:")
    for col in binary_cols:
        count = df[col].sum()
        flag_name = col.replace('is_', '')
        print(f"  {flag_name:.<30} {count:>3} countries")
    
    # Save summary statistics for indicators only
    print(f"\n{'='*60}")
    print("Summary statistics (economic indicators):")
    print(df[indicator_cols].describe().round(2))

if __name__ == "__main__":
    print("=" * 60)
    print("World Bank Economic Indicators Data Collection using World Bank API")
    print("=" * 60)
    print()
    
    # Fetch data
    df = fetch_world_bank_data(COUNTRIES, INDICATORS, start_year=2020, end_year=2023)
    
    if df is not None and len(df) > 0:
        # Add categorical features
        print(f"\n{'='*60}")
        print("Adding categorical features (Income_Level, Region, Region_Broad)...")
        df = add_categorical_features(df, INCOME_LEVELS, REGIONS, REGION_TO_BROAD)
        print(f"Added 3 categorical features")
        
        # Add binary flags for political/economic groups
        print(f"\nAdding binary flags for political/economic groups...")
        df = add_country_categories(df, POLITICAL_ECONOMIC_GROUPS)
        print(f"Added {len(POLITICAL_ECONOMIC_GROUPS)} binary flags")
        
        # Clean data
        df_clean = clean_data(df, max_missing_pct=0.4)
        
        if len(df_clean) > 0:
            # Save to CSV
            save_data(df_clean)
            
            print("\n" + "=" * 60)
            print("✓ Data collection complete!")
            print("=" * 60)
        else:
            print("\nNo countries left after cleaning. Try increasing max_missing_pct.")
    else:
        print("\nData collection failed. Please check your internet connection.")
