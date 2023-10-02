import streamlit as st
import numpy as np
import joblib
import pickle
import os

#st.sidebar.markdown("Author: Coffi Rodolphe Segbedji")
st.markdown("### Prediction of car's price based on its characteristics")
st.markdown("##### Application created by *Coffi Rodolphe Segbedji*")
st.markdown("*This application uses a model of machine learning to predict the price of a car*")

# Loading of model
file_path = os.path.abspath('final_model.joblib')
model = joblib.load(filename= file_path)               # load the model with joblib object
model2 = pickle.load(open("final_model_2", "rb"))      # load the model with pickle object


# Definition of a reference function
def inference(symboling, wheel_base, length, width, height, curb_weight, engine_size, compression_ratio, city_ympg, highway_mpg):
    new_data = np.array([
        symboling, wheel_base, length, width, height, curb_weight, 
        engine_size, compression_ratio, city_ympg, highway_mpg
    ])
    pred = model.predict(new_data.reshape(1, -1))
    #pred = model2.predict(new_data.reshape(1, -1))
    return pred


# User types a value for each car's characteristic.

symboling = st.number_input(label = 'Symboling:', value = 1, min_value = -2, max_value = 3)
wheel_base = st.number_input('Wheel-base', value = 95)
length = st.number_input("Length:", value = 150)
width = st.number_input('Width:', value = 65)
height = st.number_input('Height:', value = 50)
curb_weight = st.number_input("Curb-weight:", value = 2000)
engine_size = st.number_input('Engine-size: ', value = 120)
compression_ratio = st.number_input("Compression-ratio:", value = 9)
city_ympg = st.number_input('city-mpg:', value = 20)
highway_mpg = st.number_input("Highway-mpg: ", value = 30)


# Create a button  "Predict" that returns the model prediction.
if st.button("Predict"):
    prediction = inference(
        symboling, wheel_base, length, width, height, curb_weight, 
        engine_size, compression_ratio, city_ympg, highway_mpg
    )
    result = f" Price (in dollars) of this car  is equal to : {prediction[0]}"
    st.success(result)
 
