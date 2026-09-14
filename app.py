import joblib
import pandas as pd
import streamlit as st
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="📺",
    layout="centered"
)
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")
def engineer_features(data):
    data["engagement_ratio"] = (
        data["watch_hours"] / (data["last_login_days"] + 1)
    )

    data["watch_per_profile"] = (
        data["watch_hours"] / data["number_of_profiles"]
    )

    data["is_inactive_30"] = (
        data["last_login_days"] > 30
    ).astype(int)
    return pd.DataFrame([data])

def main():
    st.title("📺 Netflix Customer Churn Predictor")
    st.write(
        "Enter customer information and usage details "
        "to predict whether the customer is likely to churn."
    )
    bundle = load_model()
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]
    model_name = bundle.get("model_name", "Trained Model")
    st.subheader("Customer Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

    with col2:
        region = st.selectbox(
            "Region",
            [
                "North America",
                "South America",
                "Europe",
                "Asia",
                "Africa",
                "Oceania"
            ]
        )

        device = st.selectbox(
            "Primary Device",
            [
                "TV",
                "Mobile",
                "Laptop",
                "Desktop",
                "Tablet"
            ]
        )
    st.subheader("Subscription Information")
    col3, col4 = st.columns(2)
    with col3:
        subscription_type = st.selectbox(
            "Subscription Type",
            ["Basic", "Standard", "Premium"]
        )

    with col4:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Credit Card",
                "Debit Card",
                "PayPal",
                "Gift Card",
                "Crypto"
            ]
        )
    st.subheader("Engagement Information")
    watch_hours = st.slider(
        "Total Watch Hours (recent period)",
        min_value=0.0,
        max_value=120.0,
        value=10.0,
        step=0.5
    )

    last_login_days = st.slider(
        "Days Since Last Login",
        min_value=0,
        max_value=90,
        value=15,
        step=1
    )

    avg_watch_time_per_day = st.slider(
        "Average Watch Time per Day (hours)",
        min_value=0.0,
        max_value=12.0,
        value=1.0,
        step=0.1
    )

    number_of_profiles = st.slider(
        "Number of Profiles",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

    favorite_genre = st.selectbox(
        "Favorite Genre",
        [
            "Action",
            "Comedy",
            "Drama",
            "Documentary",
            "Horror",
            "Romance",
            "Sci-Fi"
        ]
    )

    st.divider()
    if st.button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    ):
        if watch_hours == 0 and avg_watch_time_per_day > 0:
            st.warning(
                "Please check the engagement values. "
                "Total watch hours cannot be 0 when average "
                "daily watch time is greater than 0."
            )
            return
        input_data = {
            "age": age,
            "gender": gender,
            "subscription_type": subscription_type,
            "watch_hours": watch_hours,
            "last_login_days": last_login_days,
            "region": region,
            "device": device,
            "payment_method": payment_method,
            "number_of_profiles": number_of_profiles,
            "avg_watch_time_per_day": avg_watch_time_per_day,
            "favorite_genre": favorite_genre
        }
        input_df = engineer_features(input_data)
        input_df = input_df[bundle["feature_columns"]]
        probability = pipeline.predict_proba(input_df)[0, 1]
        if probability >= threshold:
            prediction = "Churn"
        else:
            prediction = "No Churn"
        st.subheader("Prediction Result")

        st.metric(
            "Prediction",
            prediction
        )
        if prediction == "Churn":
            st.error(
                "⚠️ High churn risk — this customer profile "
                "resembles customers who previously churned."
            )
        else:
            st.success(
                "✅ Low churn risk — this customer profile "
                "resembles retained customers."
            )
        st.caption(
            f"Model: {model_name} | "
            f"Decision Threshold: {threshold:.2f}"
        )
        st.caption(
            "This prediction is based on historical data patterns "
            "and is not a guarantee of future customer behavior."
        )


if __name__ == "__main__":
    main()