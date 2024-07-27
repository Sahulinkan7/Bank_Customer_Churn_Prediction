import streamlit as st
import tensorflow as tf 
import pandas as pd
import numpy as np 
import pickle 

model= tf.keras.models.load_model("classification_model.h5")

with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)
    
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender=pickle.load(file)
    
with open("onehot_encoder_geography.pkl","rb") as file:
    onehot_encoder_geography=pickle.load(file)

result_at_top=st.empty()
result_value=st.empty()

st.title("Bank Customer Churn Prediction")

with st.form(key='my_form'):
    geography=st.selectbox("Geography",onehot_encoder_geography.categories_[0])
    gender=st.selectbox("Gender",label_encoder_gender.classes_)
    age=st.slider("Age",18,100)
    balance=st.number_input("Balance")
    credit_score=st.number_input("Credit score")
    estimated_salary=st.number_input("Estimated salary")
    tenure=st.slider("Tenure",0,10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_credit_card=st.selectbox("Has credit card",[0,1])
    is_active_member=st.selectbox("Is active member",[0,1])

    submit_button = st.form_submit_button(label='Submit')
    
if submit_button:
    
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_credit_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })


    geography_encoded=onehot_encoder_geography.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

    input_df=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

    scaled_input=scaler.transform(input_df)

    prediction=model.predict(scaled_input)
    prediction_prob=prediction[0][0]

    st.write(f"prediction probability is {prediction_prob:.2f}")

    if prediction_prob>0.5:
        st.info("Customer is likely to churn")
    else:
        st.info("Customer is not likely to churn")

    st.components.v1.html("""
        <script>
            window.scrollTo(0, 0);
        </script>
    """, height=0)