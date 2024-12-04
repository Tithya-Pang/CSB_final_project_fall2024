# University Major Recommendation System

## Project Overview

The **University Major Recommendation System** is a Python-based web application that helps students choose an appropriate university major based on their skills, salary expectations, career goals, and tolerance for unemployment rates. It combines machine learning (Random Forest Classifier) and data clustering (KMeans) to provide personalized major recommendations based on a set of user preferences. The system also visualizes clusters of university majors based on economic indicators, helping students understand the job market trends in different fields of study.

---

## Features

- **User Input**: Allows students to input their preferences, including skills/interests, preferred salary, unemployment tolerance, and career focus.
- **Major Recommendations**: Provides top recommendations for university majors based on the user's preferences.
- **Clustering Visualization**: Uses KMeans clustering to group university majors based on factors such as salary and unemployment rate, with an interactive Plotly chart.
- **Data Overview**: Displays the dataset of majors along with key economic indicators like salary and unemployment rates.

---

## Technologies Used

- **Python**: The primary programming language used.
- **Streamlit**: For creating the interactive web-based interface.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and data scaling.
- **Scikit-learn**: For machine learning algorithms (Random Forest Classifier and KMeans Clustering).
- **Plotly**: For creating interactive visualizations of major clusters.
- **HTML**: Used for custom styling in Streamlit components.

---

## Requirements

Before running the application, make sure you have the necessary dependencies installed. You can use `pip` to install them. You can install them all by running the following command:

```bash
pip install -r requirements.txt
