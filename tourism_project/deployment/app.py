import os
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Tourism Prediction App",
    page_icon="✈️",
    layout="wide")

st.title("Tourism Package Prediction")
st.write(
    "Fill the customer information below to predict "
    "whether the customer is likely to purchase a tourism package."
)

# 2. CLASSIFICATION THRESHOLD
THRESHOLD = 0.45

# 3. LOAD TRAINED MODEL
model_path = os.path.join(os.path.dirname(__file__), "best_churn_model_v1.joblib")
model = joblib.load(model_path)

# 4. COLLECT USER INPUTS
st.subheader("Customer Information")

with st.form("tourism_prediction_form"):

    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.slider("Age", min_value=18, max_value=70, value=30)
        TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        DurationOfPitch = st.slider("Duration of Pitch", min_value=0, max_value=100, value=15)
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        Gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        NumberOfPersonVisiting = st.slider("Number of Persons Visiting", min_value=1, max_value=5, value=2)
        NumberOfFollowups = st.slider("Number of Follow-ups", min_value=0, max_value=10, value=3)
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
        NumberOfTrips = st.slider("Number of Trips", min_value=0, max_value=20, value=2)

    with col3:
        Passport = st.selectbox("Does the Customer Have a Passport?", ["Yes", "No"])
        PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        OwnCar = st.selectbox("Does the Customer Own a Car?", ["Yes", "No"])
        NumberOfChildrenVisiting = st.slider("Number of Children Visiting", min_value=0, max_value=5, value=1)
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0)

    # Predict Button

    predict_button = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

# 5. CREATE INPUT DATAFRAME AND MAKE PREDICTION

if predict_button:

    input_data = pd.DataFrame([{
        'Age': Age,
        'TypeofContact': TypeofContact,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'Occupation': Occupation,
        'Gender': Gender,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'ProductPitched': ProductPitched,
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': MaritalStatus,
        'NumberOfTrips': NumberOfTrips,
        'Passport': 1 if Passport == "Yes" else 0,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': 1 if OwnCar == "Yes" else 0,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'Designation': Designation,
        'MonthlyIncome': MonthlyIncome
    }])

    st.subheader("Customer Input Data")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    # 6. MODEL PREDICTION

    try:

        # Check whether the model supports probabilities prediction

        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "The trained model does not support predict_proba(). "
                "A probability-enabled classifier is required "
                "to apply the 0.45 threshold."
            )

        # Calculate class probabilities

        probabilities = model.predict_proba(input_data)

        # Identify the probability column for class 1

        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError(
                f"Positive class 1 not found. Model classes: {classes}"
            )

        positive_class_index = classes.index(1)

        purchase_probability = probabilities[
            0, positive_class_index
        ]

        # Apply classification threshold

        prediction = (
            1 if purchase_probability >= THRESHOLD else 0
        )

        # 7. DISPLAY PREDICTION RESULTS

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Purchase Probability",
                f"{purchase_probability:.2%}"
            )

        with col2:

            st.metric(
                "Classification Threshold",
                f"{THRESHOLD:.0%}"
            )

        st.progress(float(purchase_probability))

        if prediction == 1:

            st.success(
                "Prediction: Customer is LIKELY to purchase "
                "the tourism package."
            )

        else:

            st.warning(
                "Prediction: Customer is UNLIKELY to purchase "
                "the tourism package."
            )

        st.caption(
            f"Classification rule: Probability ≥ {THRESHOLD:.0%} "
            "means likely to purchase."
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")
