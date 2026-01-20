import streamlit as st
import joblib
import numpy as np
try:
    model = joblib.load('SavedModel/student_performance_model.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# This script downloads a dataset from Kaggle using kagglehub, and moves it to a specified directory on your local machine.
st.title("Student Performance(Score) Predictor")
st.subheader("Predicting Exam Scores Based on Study Hours")
st.markdown("""
Welcome to the Student Performance Predictor app.
This tool helps you estimate a student's exam score based on the number of hours they have studied.
This tool uses a **Linear Regression Machine Learning model** to estimate a student's exam score based on their weekly study habits.
""")
st.write("Predict a student's exam score based on hours studied.")

st.sidebar.header("User Input features")
#Create a slider for 'Hours Studied'
#Min: 0 hours, Max: 168 (total hours in a week), Default: 10 hrs
hours_studied = st.sidebar.slider("Hours Studied per Week", 0, 168, 10)
st.write(f"### You entered: **{hours_studied} hours** of study per week.")
#The model expects a 2D array as input
input_data = np.array([[hours_studied]])
#Make prediction
if st.button("Predict Score"):
    predicted_score = model.predict(input_data)
    # Linear Regression can sometimes predict > 100 or < 0. 
    # Let's make results realistic at (0-100).
    final_score = min(max(predicted_score[0],0),100)
    #6. Display the results to the user
    st.success(f"Expected Exam Score:{final_score:.2f}%")
    #Add feedback based on score
    if final_score >= 85:
        st.balloons()
        st.write("Excellent grade! Keep it UP!")
    elif final_score >= 70:
        st.write("Good job! A little more effort and you can do even better.")
    else:
        st.write("⚠️⚠️⚠️ You need to study more to PASS!!")