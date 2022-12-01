import streamlit as st
import pandas as pd
import numpy as np


st.header("Predicción de tarifa")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv(r"C:\Users\USER\Desktop\FishWeightPredictionApplication-master\nueva_entrada.csv", sep=",", encoding="utf-8")


# load model
import joblib
best_model = joblib.load('my_model_knn.pkl') 


#st.subheader("Please select relevant features of your fish!")
#left_column, right_column = st.columns(2)

print(data)


if st.button('Make Prediction'):
    inputs = data
    prediction = best_model.predict(inputs)
    print("Tarifa", prediction)
    st.write(f"Su Tarifa es de : {prediction}g")



