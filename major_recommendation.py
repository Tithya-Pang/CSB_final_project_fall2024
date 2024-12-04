import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px
from streamlit.components.v1 import html

# Load the dataset
file_path = "all-ages.csv"  # Update with the correct file path
dataset = pd.read_csv(file_path)

# Preprocessing the data
features = dataset[["Total", "Employed", "Unemployed", "Unemployment_rate", "Median", "P25th", "P75th"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

label_encoder = LabelEncoder()
dataset["Major_category_encoded"] = label_encoder.fit_transform(dataset["Major_category"])

processed_data = pd.DataFrame(scaled_features, columns=features.columns)
processed_data["Major_category_encoded"] = dataset["Major_category_encoded"]

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(processed_data)
dataset["Cluster"] = clusters

X = processed_data.drop("Major_category_encoded", axis=1)
y = processed_data["Major_category_encoded"]
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="University Major Recommendation", layout="wide")

# Add a header
st.title("🎓 University Major Recommendation System")
st.markdown("""
This tool helps students choose a suitable university major based on their skills, interests, and career aspirations.
It takes your input and recommends majors that align with your preferences and career goals.
""")

# Custom CSS for styling
html("""
    <style>
        .sidebar .sidebar-content {
            background-color: #f0f0f5;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            width: 100%;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stSlider {
            color: #4CAF50;
        }
        .stTitle {
            color: #0073e6;
        }
        .stTextInput input {
            border-radius: 5px;
        }
    </style>
""", height=0)

# Sidebar for User Inputs
st.sidebar.header("🔍 Your Preferences")
skills = st.sidebar.text_input("Enter your skills/interests (comma-separated):", placeholder="e.g., Data Analysis, Programming, Marketing")
salary_preference = st.sidebar.slider("Preferred Median Salary ($):", 30000, 120000, 60000)
unemployment_tolerance = st.sidebar.slider("Maximum Tolerable Unemployment Rate (%):", 0, 15, 5) / 100
career_focus = st.sidebar.slider("How career-focused are you? (1=Low, 5=High)", 1, 5, 3)

# Generate Recommendations
if st.sidebar.button("Get Recommendations"):
    st.subheader("📌 Top Recommended Majors")
    
    # Filter based on user inputs
    filtered_data = dataset[
        (dataset["Median"] >= salary_preference) &
        (dataset["Unemployment_rate"] <= unemployment_tolerance)
    ]
    
    filtered_features = filtered_data[["Total", "Employed", "Unemployed", "Unemployment_rate", "Median", "P25th", "P75th"]]
    scaled_filtered_features = scaler.transform(filtered_features)

    predictions = model.predict(scaled_filtered_features)
    filtered_data["Predicted_Category"] = label_encoder.inverse_transform(predictions)

    # Display top recommendations
    top_recommendations = filtered_data.head(5)
    for _, row in top_recommendations.iterrows():
        st.write(f"**{row['Major']}**")
        st.write(f"- **Category**: {row['Major_category']}")
        st.write(f"- **Median Salary**: ${row['Median']}")
        st.write(f"- **Unemployment Rate**: {row['Unemployment_rate'] * 100:.2f}%")
        st.write(f"- **Predicted Category**: {row['Predicted_Category']}")
        st.write("---")

# Cluster Visualization with Plotly
st.subheader("🔎 Clusters Visualization")
fig = px.scatter(
    dataset,
    x="Median",
    y="Unemployment_rate",
    color="Cluster",
    hover_data=["Major", "Major_category"],
    title="Majors Clusters Based on Salary and Unemployment"
)
st.plotly_chart(fig)

# Display DataFrame in a nice format
st.write("### 📊 Dataset Overview")
st.dataframe(dataset)

# Footer for the application
st.markdown("""
---
Made with Passion by Narithtithya Pang
""")
