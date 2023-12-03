#imported important libraries
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
import numpy as np
import base64

#loaded the scaler used in our model
with open('/Users/naakoshie/Downloads/x (1).pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

model_path = '/Users/naakoshie/Downloads/final_model.h5'
#Loaded the model we created
model = load_model(model_path)

#Transformed a local image in order to use it in our css function
def get_base64_of_image(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Converts the image and get the base64 string
image_path = '/Users/naakoshie/Desktop/IMG_9835.JPG'
base64_string = get_base64_of_image(image_path)

# CSS using base64 string for the background
background_css = f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{base64_string}");
    background-size: cover;
    background-repeat: no-repeat;
}}
</style>
"""

# use CSS with Markdown
st.markdown(background_css, unsafe_allow_html=True)

#A function the is used as the home page and includes the introductions and purpose of the application
def page1():
        st.title("Welcome to the Autism Predictor")
        st.write('Brought to you by group 23')
        st.markdown("""
            ## Empowering Early Autism Detection
            - **Innovative Approach:** Leveraging AI to enhance diagnostic accuracy.
            - **User-Friendly Interface:** Designed for easy navigation and use.
            ---
            ### Begin Your Journey
            Explore our autism prediction tool to gain insights and guidance. 
            - Answer simple questions
            - Get instant predictions
            - Understand early signs of autism
            ---
            _"Every child is unique, and early understanding makes a world of difference."_
            """)
        st.write('Please access the Autism Predictor located on the sidebar')


# A function for the autism predictor page which displays the questions that will be required to be used in the predictor
def page2():
    st.title("Autism Predictor")
    
    A9_options = [1, 0]
    A9 = st.selectbox('Question 1: Does your child use simple gestures? (e.g. wave goodbye)', A9_options, key='A9')

    st.write('Pick 1 if your answer is : Sometimes, Rarely or Never')
    st.write('Pick 0 if your answer is : Usually or Always')

    A5_options = [1, 0]
    A5 = st.selectbox('Question 2: Does your child play pretend? (e.g. care for dolls, talk on a toy phone)', A5_options, key='A5')

    st.write('Pick 1 if your answer is : Sometimes, Rarely or Never')
    st.write('Pick 0 if your answer is : Usually or Always')

    A7_options = [1, 0]
    A7 = st.selectbox('Question 3: If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)', A7_options, key='A7')

    st.write('Pick 1 if your answer is : Sometimes, Rarely or Never')
    st.write('Pick 0 if your answer is : Usually or Always')

    A6_options = [1, 0]
    A6 = st.selectbox('Question 4: Does your child follow where you’re looking?', A6_options, key='A6')

    st.write('Pick 1 if your answer is : Sometimes, Rarely or Never')
    st.write('Pick 0 if your answer is : Usually or Always')

    A1_options = [1, 0]
    A1 = st.selectbox('Question 5: Does your child look at you when you call his/her name?', A1_options, key='A1')
    st.write('Pick 1 if your answer is : Sometimes, Rarely or Never')
    st.write('Pick 0 if your answer is : Usually or Always')
    

    A2_options = [1, 0]
    A2 = st.selectbox('Question 6: How easy is it for you to get eye contact with your child?', A2_options, key='A2')
    st.write('Pick 1 if your answer is : Hard, Very Hard')
    st.write('Pick 0 if your answer is : Moderately Easy, Easy ,Very Easy')

    

    A4_options = [1, 0]
    A4 = st.selectbox('Question 7: Does your child point to share interest with you? (e.g. pointing at an interesting sight)', A4_options, key='A4')
    st.write('Pick 1 if your answer is : Sometimes, Rarely or Never')
    st.write('Pick 0 if your answer is : Usually or Always')

    
    A8_options = [1, 0]
    A8 = st.selectbox('Question 8: How would you describe your child’s first words as:', A8_options, key='A8')
    st.write('Pick 1 if your answer is : No meaningful words yet')
    st.write('Pick 0 if your answer is : Single Words')

    A3_options = [1, 0]
    A3 = st.selectbox('Question 9: Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)', A3_options, key='A3')
    st.write('Pick 1 if your answer is : Sometimes, Rarely or Never')
    st.write('Pick 0 if your answer is : Usually or Always')

    Age_Mons = st.number_input('Question 10: Age in Months', min_value = 0, step = 1, key='Age_Mons')


#when the predict button is selected in the main application this if statement executes
    if st.button("Predict"):
        input = np.array([[A9, A5, A7, A6, A1, A2, A4, A8, Age_Mons, A3,]])
        new_data_scaled = loaded_scaler.transform(input)

        prediction_probabilities = model.predict(new_data_scaled)
        # the prediction for the binary classification, assuming a threshold of 0.5
        prediction = (prediction_probabilities < 0.5).astype(int)[0]

        # Confidence score 
        confidence_score = prediction_probabilities[0] * 100 if prediction[0] == 1 else (1 - prediction_probabilities[0]) * 100

        if prediction[0] == 1:
            st.title("This child is likely to have ASD traits")
        else:
            st.title("This child is not likely to have ASD traits")

        st.title(f"Confidence Score: {confidence_score[0]:.2f}%")
        st.title("Thank you for using our model")

# Creates a sidebar that allows page selection
page = st.sidebar.selectbox("Select Page", ["Home Page", "Autism Predictor"])

# Goes to the selected page
if page == "Home Page":
    page1()
elif page == "Autism Predictor":
    page2()


