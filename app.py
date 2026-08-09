import streamlit as st
import pandas as pd

from src.predict import predict_failure


st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)


st.title("⚙️ Industrial Predictive Maintenance")
st.write(
    "Predict the probability of machine failure using "
    "sensor and operating-condition data."
)

st.divider()


# Machine Type
machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)


# Sensor inputs
air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=300.0
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=310.0
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=0,
    max_value=5000,
    value=1500
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100
)


st.divider()


if st.button("Predict Machine Failure", type="primary"):

    input_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    prediction, probability = predict_failure(input_data)

    st.subheader("Prediction Result")

    if probability >= 0.70:
        risk_level = "HIGH"
        st.error("🔴 High Failure Risk")
        recommendation = (
            "Immediate inspection recommended. "
            "Check torque, rotational speed, and tool wear."
        )

    elif probability >= 0.30:
        risk_level = "MODERATE"
        st.warning("🟠 Moderate Failure Risk")
        recommendation = (
            "Schedule preventive inspection and monitor "
            "machine operating conditions closely."
        )

    else:
        risk_level = "LOW"
        st.success("🟢 Low Failure Risk")
        recommendation = (
            "Machine currently appears to operate within "
            "a relatively low-risk range."
        )


    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Failure Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Risk Level",
            risk_level
        )

    st.progress(float(probability))

    st.subheader("Maintenance Recommendation")

    st.info(recommendation)

    st.subheader("Machine Sensor Summary")

    sensor_data = pd.DataFrame({
        "Parameter": [
            "Machine Type",
            "Air Temperature",
            "Process Temperature",
            "Rotational Speed",
            "Torque",
            "Tool Wear"
        ],
        "Value": [
            machine_type,
            f"{air_temperature:.1f} K",
            f"{process_temperature:.1f} K",
            f"{rotational_speed} rpm",
            f"{torque:.1f} Nm",
            f"{tool_wear} min"
        ]
    })

    st.table(sensor_data)

    st.subheader("Key Risk Factors")

    st.write(
        "The model identified torque, tool wear, and rotational speed "
        "as important predictors of machine failure."
    )

    