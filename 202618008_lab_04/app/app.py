import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import date

# ---------- Load model + metadata once, cached across reruns ----------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load('models/final_airbnb_price_model.pkl')
    with open('models/app_metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
    return pipe, meta

pipe, meta = load_artifacts()
lookup = meta['neighbourhood_lookup']

st.set_page_config(page_title="NYC Airbnb Price Estimator", page_icon="🏠")
st.title("🏠 NYC Airbnb Nightly Price Estimator")
st.write("Fill in the listing details below to get an estimated nightly price.")

# ---------- Input form ----------
# Why st.form: groups all inputs so the app only re-runs (and predicts)
# once, when the user clicks Submit -- not on every single widget change.
with st.form("listing_form"):
    col1, col2 = st.columns(2)

    with col1:
        room_type = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room"])
        neighbourhood_group = st.selectbox(
            "Borough", sorted(lookup['neighbourhood_group'].unique())
        )
        # Cascading dropdown: only show neighbourhoods within the chosen borough
        filtered = lookup[lookup['neighbourhood_group'] == neighbourhood_group]
        neighbourhood = st.selectbox("Neighbourhood", sorted(filtered['neighbourhood'].unique()))
        minimum_nights = st.number_input("Minimum nights", min_value=1, max_value=365, value=2)
        listing_name = st.text_input("Listing title", "Cozy private room near subway")

    with col2:
        availability_365 = st.slider("Days available per year", 0, 365, 180)
        calculated_host_listings_count = st.number_input(
            "Host's total listings count", min_value=1, value=1
        )
        number_of_reviews = st.number_input("Number of reviews", min_value=0, value=5)
        reviews_per_month = st.number_input("Reviews per month", min_value=0.0, value=0.5, step=0.1)
        has_reviews_input = number_of_reviews > 0
        last_review_date = st.date_input(
            "Date of last review", value=date(2019, 6, 1), disabled=not has_reviews_input
        )

    submitted = st.form_submit_button("Estimate Price")

# ---------- Prediction ----------
if submitted:
    # Look up this neighbourhood's average coordinates
    row = lookup[lookup['neighbourhood'] == neighbourhood].iloc[0]
    lat, lon = row['avg_lat'], row['avg_lon']

    # Recompute the SAME derived features used in training (Step 1.5),
    # using the SAME reference values saved in metadata.
    distance_to_center_km = np.sqrt(
        (lat - meta['center_lat'])**2 + (lon - meta['center_lon'])**2
    ) * 111

    if has_reviews_input:
        days_since_last_review = (meta['snapshot_date'] - pd.Timestamp(last_review_date)).days
        days_since_last_review = max(days_since_last_review, 0)
    else:
        days_since_last_review = meta['sentinel_days']

    availability_rate = availability_365 / 365
    name_length = len(listing_name)
    name_word_count = len(listing_name.split())

    # Build a single-row DataFrame with EXACTLY the columns/order the
    # pipeline's ColumnTransformer expects (same FEATURES list as Step 1.7)
    input_df = pd.DataFrame([{
        'room_type': room_type,
        'neighbourhood_group': neighbourhood_group,
        'neighbourhood': neighbourhood,
        'latitude': lat,
        'longitude': lon,
        'distance_to_center_km': distance_to_center_km,
        'minimum_nights': minimum_nights,
        'availability_rate': availability_rate,
        'calculated_host_listings_count': calculated_host_listings_count,
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month,
        'has_reviews': int(has_reviews_input),
        'days_since_last_review': days_since_last_review,
        'name_length': name_length,
        'name_word_count': name_word_count,
    }])

    # Predict in log-space, then convert back to dollars (matches training target)
    log_pred = pipe.predict(input_df)[0]
    price_pred = np.expm1(log_pred)

    st.success(f"### Estimated nightly price: ${price_pred:.2f}")
    st.caption(
        f"Based on a {room_type.lower()} in {neighbourhood}, {neighbourhood_group}, "
        f"~{distance_to_center_km:.1f} km from Midtown Manhattan."
    )

    with st.expander("See the exact features sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))