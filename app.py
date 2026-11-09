import pickle
from datetime import date, timedelta
 
import pandas as pd
import requests
import streamlit as st
 
# ---- Settings ----------------------------------------------------------
CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Miami": (25.76, -80.19),
    "Los Angeles": (34.05, -118.24),
}
LOW_TEMP, HIGH_TEMP = 50, 100        # the temperatures (F) the model learned from
LOW_SALES, HIGH_SALES = 200, 300     # below LOW_SALES is a slow day, above HIGH_SALES is a busy one
URL = "https://api.open-meteo.com/v1/forecast"
# ------------------------------------------------------------------------
 
# Load the trained model (we trained it once, in Colab, and saved it to a file)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
 
st.set_page_config(page_title="Ice Cream Sales Predictor", page_icon="🍦")
st.title("🍦 Ice Cream Sales Predictor")
st.caption("Pick a city and a date. The app gets the weather forecast (high temperature "
           "and chance of rain) and predicts how many cones you will sell.")
 
city = st.selectbox("City", list(CITIES))
selected_date = st.date_input(
    "Date",
    value=date.today() + timedelta(days=1),
    min_value=date.today(),
    max_value=date.today() + timedelta(days=15),    # the forecast reaches about 16 days ahead
)
 
if st.button("Predict sales", type="primary"):
    # 1. Ask the weather API for the forecast (both predictors in one call)
    latitude, longitude = CITIES[city]
    try:
        response = requests.get(
            URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "temperature_2m_max,precipitation_probability_max",
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "start_date": str(selected_date),
                "end_date": str(selected_date),
            },
            timeout=30,
        )
        response.raise_for_status()
        daily = response.json()["daily"]
    except (requests.RequestException, KeyError) as e:
        st.error(f"Could not get the weather forecast: {e}")
        st.stop()
 
    temperature = daily["temperature_2m_max"][0]
    rain_chance = daily["precipitation_probability_max"][0]
    if temperature is None:
        st.error("No temperature forecast is available for that date yet. Try an earlier date.")
        st.stop()
    rain_missing = rain_chance is None
    if rain_missing:
        rain_chance = 0      # some far-out dates have no rain probability yet
 
   features = pd.DataFrame({"temperature": [temperature], "precip_chance": [rain_chance]})
    prediction = max(0, model.predict(features)[0])
 
    # 3. Show the result
    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast high", f"{temperature:.0f} °F")
    col2.metric("Chance of rain", "n/a" if rain_missing else f"{rain_chance:.0f} %")
    col3.metric("Predicted sales", f"{prediction:.0f} cones")
 
    if rain_missing:
        st.warning("No rain forecast is available for this date yet, so the prediction assumes a dry day.")
 
    if temperature < LOW_TEMP or temperature > HIGH_TEMP:
        st.warning(f"This temperature is outside the range the model learned from ({LOW_TEMP} to {HIGH_TEMP} °F). "
                   "Treat the number with caution.")
 
    if prediction > HIGH_SALES:
        st.info("Busy day: make sure you are fully stocked.")
    elif prediction > LOW_SALES:
        st.info("A normal day is expected.")
    else:
        st.info("A slow day: consider a promotion, or order less.")
