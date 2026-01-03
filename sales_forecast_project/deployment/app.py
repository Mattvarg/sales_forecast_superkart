import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="MattVarg/sales-forecast-package-model", filename="sales_forecast_package_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Sales Forecast Prediction
st.title("Sales Forecast Prediction")
st.write("Fill the details below to forecast sales.")

# Collect user input
Product_Weight = st.slider("Product Weight", 4.0, 25.0, 5.0)
Product_Sugar_Content = st.selectbox("Sugar Content", ["Low Sugar", "No Sugar", "Regular"])
Product_Allocated_Area = st.slider("Product Allocated Area", 0.001, 1.0, 0.5)
Product_Type = st.selectbox("Product Type", ["Baking Goods", "Canned","Dairy","Frozen Foods","Fruits and Vegetables","Health and Hygiene","Household","Starchy Foods","Soft Drinks","Snack Foods"])
Product_MRP = st.slider("Product MRP", 30.0, 300.0, 100.0)
Store_Establishment_Year = st.slider("Store Establishment Year", 1985, 2020, 2000)
Store_Size = st.selectbox("Store Size", ["High", "Small", "Medium"])
Store_Location_City_Type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type =  st.selectbox("Store Type", ["Food Mart", "Supermarket Type1", "Supermarket Type2","Departmental Store"])

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_Type': Product_Type,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

# Predict button
if st.button("Predict Sales"):
    # Make prediction
    predicted_sales = model.predict(input_data)[0]
    st.success(f"Predicted Sales Total: ${predicted_sales:,.2f}")
