import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st

import pymongo
from pymongo import MongoClient
from pymongo import collection
from pymongo import ssl_support

uri = "mongodb+srv://admin:admin@cluster0.zaqge.mongodb.net/heart_disease_db?retryWrites=true&w=majority"
cluster = MongoClient(uri, ssl_cert_reqs=ssl_support.CERT_NONE)
db = cluster["heart_disease_db"]
collection = db["first"]


# Title
st.write("""
# Heart Disease Detection
Prevention is Better Than Cure
""")

# Display image
image = Image.open('C:/Users/Sayuru Dissanayake/Desktop/ML/logo.jpg')
#st.image(image, caption='ML', use_column_width=True)
st.image(image, width=450)


# Get the data 
heart_data = pd.read_csv('C:/Users/Sayuru Dissanayake/Desktop/ML/heart.csv')


# Sub Title
st.subheader('Data Information')
# Show data as a table
# >>st.dataframe(heart_data)
# Show stats
# >>st.write(df.describe())
# Show data as a chart
# >>chart = st.bar_chart(heart_data)


# Splitting the features and Target
x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']


# Splitting data into Training and Test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)


# Getting user inputs
def get_user_inputs():
    unqid = st.sidebar.text_input("Enter Patient ID", "ab001")
    age = st.sidebar.slider('Age', 1, 110, 1)
    sex = st.sidebar.slider('Sex [1=male; 0=female]', 0, 1, 0)
    cp = st.sidebar.slider('Chest Pain Type', 0, 3, 0)
    #trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 94)
    #trestbps = 140
    chol = st.sidebar.slider('Serum Cholestoral', 126, 564, 126)
    fbs = st.sidebar.slider('Fasting Blood Sugar', 0, 1, 0)
    #restecg = st.sidebar.slider('Resting Electrocardiographic Results', 0, 2, 0)
    #restecg = 0
    #thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 71)
    #thalach = 170
    exang = st.sidebar.slider('Exercise Induced Angina', 0,1, 0)
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 0.0)
    #slope = st.sidebar.slider('The Slope of the Peak Exercise ST Segment', 0, 2, 0)
    #slope = 0
    ca = st.sidebar.slider('Number of Major Vessels', 0, 4, 0)
    thal = st.sidebar.slider('Thalium Stress Results', 0, 3, 0)
    
    result = collection.find({"_id":unqid})

    for i in result:
        trestbps = i["trestbps"]
        restecg = i["restecg"]
        thalach = i["thalach"]
        slope = i["slope"]



    # store in a dictionary
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    features = pd.DataFrame(user_data, index= [0])
    return features



# Store user inputs to a variable
user_input = get_user_inputs()

# Display user inputs
st.subheader('User Inputs')
st.write(user_input)



# Logistic Regression
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# training the model with training data
model.fit(x_train, y_train)


# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
st.subheader('Test Accuracy Score : ')
st.write(str(test_data_accuracy * 100) + '%')


prediction = model.predict(user_input)
ans = '-'
if (prediction[0] == 0):
    ans = 'No'

else:
    ans = 'Yes'
st.subheader('Possibility of a Heart Attack : ' + ans)



