import streamlit as st
import pandas as pd
import joblib
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ============================================================================================
# PAGE CONFIGURATION
# ============================================================================================
st.set_page_config(
    page_title="AutoWise - Smart Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================================
# DATA LOADING AND CLEANING FUNCTION
# ============================================================================================
@st.cache_data
def load_data_and_model():
    # Load the raw CSV file
    try:
        df = pd.read_csv('car data.csv')
    except FileNotFoundError:
        st.error("Error: `car data.csv` not found. Please make sure it's in the same folder as the app.")
        return None, None

    # ** CRITICAL FIX: Standardize column names **
    df.rename(columns={
        'Selling_type': 'Seller_Type',
        'Driven_kms': 'Kms_Driven'
    }, inplace=True)
    
    # Feature Engineering
    df['brand'] = df['Car_Name'].apply(lambda x: x.split(' ')[0])
    df['car_age'] = datetime.datetime.now().year - df['Year']
    
    # Load the trained pipeline
    try:
        pipeline = joblib.load('xgboost_pipeline_car_data.pkl')
    except FileNotFoundError:
        st.error("Error: `xgboost_pipeline_car_data.pkl` not found. Please place the file in the correct directory.")
        return df, None
        
    return df, pipeline

df, pipeline = load_data_and_model()

# Load CSS (optional)
try:
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("CSS file not found. App will run with default styling.")

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ============================================================================================
# HELPER FUNCTIONS
# ============================================================================================
def format_price(price_lakhs):
    """Formats the price into a readable string."""
    return f"₹ {price_lakhs:.2f} Lakhs"

def add_trendline_without_statsmodels(df, x_col, y_col):
    """Add a simple trendline without using statsmodels"""
    # Calculate rolling average for trendline
    sorted_df = df.sort_values(by=x_col)
    sorted_df['trend'] = sorted_df[y_col].rolling(window=5, min_periods=1, center=True).mean()
    return sorted_df

# ============================================================================================
# DATA VISUALIZATION FUNCTIONS (UPDATED - NO STATSMODELS DEPENDENCY)
# ============================================================================================
def create_market_overview_charts(df):
    """Create comprehensive market overview charts without statsmodels"""
    
    # Chart 1: Price Distribution by Brand
    fig1 = px.box(df, x='brand', y='Selling_Price', 
                  title='📊 Selling Price Distribution by Brand',
                  color='brand',
                  labels={'Selling_Price': 'Selling Price (Lakhs)', 'brand': 'Brand'},
                  template='plotly_dark')
    fig1.update_layout(showlegend=False, height=400)
    
    # Chart 2: Fuel Type Impact on Price
    fuel_price_fig = px.violin(df, x='Fuel_Type', y='Selling_Price', color='Fuel_Type',
                              title='⛽ Fuel Type Impact on Selling Price',
                              labels={'Selling_Price': 'Selling Price (Lakhs)', 'Fuel_Type': 'Fuel Type'},
                              template='plotly_dark')
    fuel_price_fig.update_layout(showlegend=False, height=400)
    
    # Chart 3: Transmission Type Comparison
    trans_fig = px.box(df, x='Transmission', y='Selling_Price', color='Transmission',
                      title='⚙️ Transmission Type Price Comparison',
                      labels={'Selling_Price': 'Selling Price (Lakhs)', 'Transmission': 'Transmission Type'},
                      template='plotly_dark')
    trans_fig.update_layout(showlegend=False, height=400)
    
    # Chart 4: Car Age vs Selling Price (without LOWESS)
    age_fig = px.scatter(df, x='car_age', y='Selling_Price', color='Fuel_Type',
                        title='📈 Car Age vs Selling Price',
                        labels={'car_age': 'Car Age (Years)', 'Selling_Price': 'Selling Price (Lakhs)'},
                        template='plotly_dark')
    
    # Add a simple polynomial trendline
    try:
        # Calculate average price per age
        age_avg = df.groupby('car_age')['Selling_Price'].mean().reset_index()
        age_fig.add_traces(go.Scatter(x=age_avg['car_age'], y=age_avg['Selling_Price'],
                                    mode='lines', name='Trend',
                                    line=dict(color='white', width=3, dash='dash')))
    except:
        pass  # If trendline fails, just show scatter plot
    
    age_fig.update_layout(height=400)
    
    return fig1, fuel_price_fig, trans_fig, age_fig

def create_brand_analysis_charts(df):
    """Create detailed brand analysis charts"""
    
    # Brand performance metrics
    brand_stats = df.groupby('brand').agg({
        'Selling_Price': ['mean', 'median', 'count'],
        'Present_Price': 'mean',
        'car_age': 'mean',
        'Kms_Driven': 'mean'
    }).round(2)
    brand_stats.columns = ['Avg_Selling_Price', 'Median_Selling_Price', 'Count', 
                          'Avg_Showroom_Price', 'Avg_Car_Age', 'Avg_Kms_Driven']
    brand_stats = brand_stats.reset_index()
    
    # Chart 1: Top brands by average selling price
    top_brands = brand_stats.nlargest(10, 'Avg_Selling_Price')
    brand_price_fig = px.bar(top_brands, x='brand', y='Avg_Selling_Price',
                            title='🏆 Top 10 Brands by Average Selling Price',
                            color='Avg_Selling_Price',
                            color_continuous_scale='viridis',
                            labels={'Avg_Selling_Price': 'Average Selling Price (Lakhs)', 'brand': 'Brand'},
                            template='plotly_dark')
    brand_price_fig.update_layout(height=500)
    
    # Chart 2: Brand popularity (number of listings)
    popular_brands = brand_stats.nlargest(10, 'Count')
    popularity_fig = px.pie(popular_brands, values='Count', names='brand',
                           title='📊 Market Share by Brand (Top 10)',
                           template='plotly_dark')
    popularity_fig.update_layout(height=500)
    
    # Chart 3: Price depreciation analysis by brand
    brand_stats['Depreciation_Rate'] = ((brand_stats['Avg_Showroom_Price'] - brand_stats['Avg_Selling_Price']) / 
                                       brand_stats['Avg_Showroom_Price']) * 100
    depreciation_fig = px.scatter(brand_stats, x='Avg_Car_Age', y='Depreciation_Rate', 
                                 size='Avg_Selling_Price', color='brand',
                                 title='💸 Depreciation Analysis by Brand',
                                 labels={'Avg_Car_Age': 'Average Car Age (Years)', 
                                        'Depreciation_Rate': 'Depreciation Rate (%)',
                                        'Avg_Selling_Price': 'Average Selling Price'},
                                 template='plotly_dark',
                                 hover_data=['brand', 'Avg_Selling_Price', 'Depreciation_Rate'])
    depreciation_fig.update_layout(height=500)
    
    return brand_price_fig, popularity_fig, depreciation_fig, brand_stats

def create_feature_impact_charts(df):
    """Create charts showing impact of different features on car prices"""
    
    # Chart 1: Owner count impact
    owner_fig = px.box(df, x='Owner', y='Selling_Price', color='Owner',
                      title='👤 Impact of Previous Owners on Selling Price',
                      labels={'Selling_Price': 'Selling Price (Lakhs)', 'Owner': 'Number of Previous Owners'},
                      template='plotly_dark')
    owner_fig.update_layout(showlegend=False, height=400)
    
    # Chart 2: Kilometer driven impact (without LOWESS)
    km_fig = px.scatter(df, x='Kms_Driven', y='Selling_Price', color='Fuel_Type',
                       title='🛣️ Kilometers Driven vs Selling Price',
                       labels={'Kms_Driven': 'Kilometers Driven', 'Selling_Price': 'Selling Price (Lakhs)'},
                       template='plotly_dark')
    
    # Add average line for kilometers
    try:
        km_bins = pd.cut(df['Kms_Driven'], bins=10)
        km_avg = df.groupby(km_bins)['Selling_Price'].mean().reset_index()
        km_avg['Kms_mid'] = km_avg['Kms_Driven'].apply(lambda x: x.mid)
        km_fig.add_trace(go.Scatter(x=km_avg['Kms_mid'], y=km_avg['Selling_Price'],
                                  mode='lines', name='Average Trend',
                                  line=dict(color='white', width=3)))
    except:
        pass
    
    km_fig.update_layout(height=400)
    
    # Chart 3: Seller type comparison
    seller_fig = px.violin(df, x='Seller_Type', y='Selling_Price', color='Seller_Type',
                          title='🏪 Seller Type Price Distribution',
                          labels={'Selling_Price': 'Selling Price (Lakhs)', 'Seller_Type': 'Seller Type'},
                          template='plotly_dark')
    seller_fig.update_layout(showlegend=False, height=400)
    
    # Chart 4: Year-wise price trends
    year_trend = df.groupby('Year')['Selling_Price'].mean().reset_index()
    year_fig = px.line(year_trend, x='Year', y='Selling_Price',
                      title='📅 Year-wise Average Selling Price Trend',
                      labels={'Selling_Price': 'Average Selling Price (Lakhs)', 'Year': 'Manufacturing Year'},
                      template='plotly_dark',
                      markers=True)
    year_fig.update_layout(height=400)
    
    return owner_fig, km_fig, seller_fig, year_fig

def create_interactive_comparison_charts(df):
    """Create interactive comparison charts"""
    
    # Price comparison by multiple features
    fig = px.sunburst(df, path=['brand', 'Fuel_Type', 'Transmission'], 
                     values='Selling_Price', color='Selling_Price',
                     title='🔍 Price Distribution: Brand → Fuel Type → Transmission',
                     color_continuous_scale='viridis',
                     template='plotly_dark')
    fig.update_layout(height=600)
    
    return fig

# ============================================================================================
# SIDEBAR - INPUT FORM
# ============================================================================================
st.sidebar.header('🚗 AutoWise Predictor')
st.sidebar.markdown("Enter car details to estimate the selling price.")

if df is not None:
    with st.sidebar.form(key='prediction_form'):
        car_brand = st.selectbox('Brand', options=sorted(df['brand'].unique()))
        car_year = st.slider('Manufacturing Year', int(df['Year'].min()), int(df['Year'].max()), 2017)
        present_price = st.number_input('Current Showroom Price (Lakhs)', min_value=0.5, value=10.0, step=0.5)
        km_driven = st.number_input('Kilometers Driven', min_value=500, value=40000, step=1000)

        col1, col2 = st.columns(2)
        with col1:
            fuel_type = st.selectbox('Fuel Type', options=df['Fuel_Type'].unique())
            seller_type = st.selectbox('Seller Type', options=df['Seller_Type'].unique())
        with col2:
            transmission_type = st.selectbox('Transmission', options=df['Transmission'].unique())
            owner_type = st.selectbox('Number of Previous Owners', options=sorted(df['Owner'].unique()))

        submit_button = st.form_submit_button(label='Predict Selling Price')
else:
    st.sidebar.error("Data could not be loaded. Please check file names.")
    submit_button = False

# ============================================================================================
# MAIN CONTENT
# ============================================================================================
st.title('🚗 Smart Car Price Valuation')
st.markdown("An advanced AI tool for precise price estimations with comprehensive market analysis.")

# Prediction Section
if submit_button and pipeline is not None:
    car_age = datetime.datetime.now().year - car_year
    input_data = pd.DataFrame({
        'Present_Price': [present_price], 'Kms_Driven': [km_driven], 'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type], 'Transmission': [transmission_type], 'Owner': [owner_type],
        'brand': [car_brand], 'car_age': [car_age], 'Year': [car_year] 
    })
    
    predicted_price = pipeline.predict(input_data)[0]
    predicted_price = max(0.1, predicted_price)
    
    st.markdown(f"""
    <div class="result-card"><h2>Predicted Selling Price</h2><p class="price">{format_price(predicted_price)}</p></div>
    """, unsafe_allow_html=True)
    
    # Display insights and analysis
    st.subheader("💡 Key Insights & Analysis")
    total_depreciation = present_price - predicted_price
    annual_depreciation = total_depreciation / car_age if car_age > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card"><h4>Total Depreciation</h4><p>{format_price(max(0, total_depreciation))}</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><h4>Annual Depreciation</h4><p>{format_price(max(0, annual_depreciation))}/yr</p></div>""", unsafe_allow_html=True)
    with col3:
        if present_price > 0:
            value_retention = (predicted_price / present_price) * 100
        else:
            value_retention = 0
        st.markdown(f"""<div class="metric-card"><h4>Value Retention</h4><p>{value_retention:.1f}%</p></div>""", unsafe_allow_html=True)

    # Add to history
    prediction_record = {"Brand": car_brand, "Year": car_year, "Showroom Price": format_price(present_price), "Predicted Price": format_price(predicted_price)}
    st.session_state.prediction_history.insert(0, prediction_record)
    st.session_state.prediction_history = st.session_state.prediction_history[:10]

elif df is not None:
    st.info("Fill out the details in the sidebar to predict a car's price.")

# ============================================================================================
# DATA VISUALIZATION SECTIONS (UPDATED)
# ============================================================================================
if df is not None:
    # Section 1: Market Overview
    st.markdown("---")
    st.header("📈 Market Overview Analysis")
    
    st.subheader("Price Trends & Distributions")
    fig1, fuel_price_fig, trans_fig, age_fig = create_market_overview_charts(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(trans_fig, use_container_width=True)
    with col2:
        st.plotly_chart(fuel_price_fig, use_container_width=True)
        st.plotly_chart(age_fig, use_container_width=True)
    
    # Section 2: Brand Analysis
    st.markdown("---")
    st.header("🏭 Brand Performance Analysis")
    
    brand_price_fig, popularity_fig, depreciation_fig, brand_stats = create_brand_analysis_charts(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(brand_price_fig, use_container_width=True)
    with col2:
        st.plotly_chart(popularity_fig, use_container_width=True)
    
    st.plotly_chart(depreciation_fig, use_container_width=True)
    
    # Section 3: Feature Impact Analysis
    st.markdown("---")
    st.header("🔍 Feature Impact Analysis")
    
    owner_fig, km_fig, seller_fig, year_fig = create_feature_impact_charts(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(owner_fig, use_container_width=True)
        st.plotly_chart(km_fig, use_container_width=True)
    with col2:
        st.plotly_chart(seller_fig, use_container_width=True)
        st.plotly_chart(year_fig, use_container_width=True)
    
    # Section 4: Interactive Charts
    st.markdown("---")
    st.header("🔄 Interactive Analysis")
    
    sunburst_fig = create_interactive_comparison_charts(df)
    st.plotly_chart(sunburst_fig, use_container_width=True)
    
    # Section 5: Correlation Analysis (Simplified)
    st.markdown("---")
    st.header("📊 Feature Correlation Analysis")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        correlation_matrix = numeric_df.corr()
        
        corr_fig = px.imshow(correlation_matrix,
                            title="Heatmap: Feature Correlations",
                            color_continuous_scale="RdBu_r",
                            aspect="auto",
                            template='plotly_dark')
        corr_fig.update_layout(height=600)
        st.plotly_chart(corr_fig, use_container_width=True)
    
    # Section 6: Quick Statistics
    st.markdown("---")
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cars", len(df))
    with col2:
        st.metric("Unique Brands", df['brand'].nunique())
    with col3:
        st.metric("Average Selling Price", format_price(df['Selling_Price'].mean()))
    with col4:
        st.metric("Data Range", f"{df['Year'].min()}-{df['Year'].max()}")
    
    # Additional stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Most Common Fuel", df['Fuel_Type'].mode().iloc[0] if not df['Fuel_Type'].empty else "N/A")
    with col2:
        st.metric("Avg Kilometers", f"{df['Kms_Driven'].mean():,.0f} km")
    with col3:
        st.metric("Manual Cars", f"{(df['Transmission'] == 'Manual').sum()}")
    with col4:
        st.metric("Avg Car Age", f"{df['car_age'].mean():.1f} years")

# --- Display Prediction History ---
if st.session_state.prediction_history:
    st.markdown("---")
    st.subheader("📋 Recent Predictions")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)
