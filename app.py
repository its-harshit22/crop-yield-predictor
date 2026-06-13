import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')
columns = joblib.load('columns.pkl')

st.title("🌾 Crop Yield Predictor")
st.write("Predict crop yield (hg/ha) based on region, crop type, and environmental factors.")

areas = ['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon', 'Canada', 'Central African Republic', 'Chile', 'Colombia', 'Croatia', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Eritrea', 'Estonia', 'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon', 'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway', 'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom', 'Uruguay', 'Zambia', 'Zimbabwe']
crops = ['Cassava', 'Maize', 'Plantains and others', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Sweet potatoes', 'Wheat', 'Yams']

col1, col2 = st.columns(2)
with col1:
    area = st.selectbox("Country/Area", areas, index=areas.index("India"))
    crop = st.selectbox("Crop", crops, index=crops.index("Wheat"))
    year = st.number_input("Year", 1990, 2030, 2024)

with col2:
    rainfall = st.number_input("Avg Rainfall (mm/year)", 0, 5000, 1200)
    pesticides = st.number_input("Pesticides (tonnes)", 0, 50000, 5000)
    temp = st.number_input("Avg Temperature (°C)", -10.0, 50.0, 22.0)

if st.button("Predict Yield"):
    input_data = pd.DataFrame([{col: 0 for col in columns}])
    input_data['Year'] = year
    input_data['average_rain_fall_mm_per_year'] = rainfall
    input_data['pesticides_tonnes'] = pesticides
    input_data['avg_temp'] = temp

    area_col = f'Area_{area}'
    crop_col = f'Item_{crop}'
    if area_col in input_data.columns:
        input_data[area_col] = 1
    if crop_col in input_data.columns:
        input_data[crop_col] = 1

    pred = model.predict(input_data)[0]
    st.success(f"Predicted Yield: **{pred:.2f} hg/ha** ({pred/10000:.2f} tons/ha)")
    # Compare with historical average for this crop
    df_orig = pd.read_csv('yield_df.csv')
    avg_yield = df_orig[df_orig['Item'] == crop]['hg/ha_yield'].mean()

    comparison = pd.DataFrame({
        'Yield (hg/ha)': [pred, avg_yield]
    }, index=['Your Prediction', f'Historical Avg ({crop})'])

    st.bar_chart(comparison)

st.markdown("---")
st.subheader("Model Performance")
st.write("R² Score: 0.9875 | MAE: 3461.19")

st.subheader("Feature Importance")
st.image("feature_importance.png")

st.subheader("Actual vs Predicted (Test Set)")
st.image("actual_vs_predicted.png")
