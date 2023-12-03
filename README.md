# group23._FinalProject
ReadMe for Autism Spectrum Disorder (ASD) Traits Predictor
This application is designed to predict Autism Spectrum Disorder (ASD) traits in toddlers. It utilizes a deep learning model trained on responses to the Q-chat-10-Toddler assessment. The primary goal is to provide a quick and user-friendly way for parents and caregivers to understand the potential ASD traits based on specific behavioral questions and be able to seek help and guidance early inorder to improve the overall health and life of their child.

Features of these application are: 
1.)Binary classification model predicting the likelihood of ASD traits.
2.)User interface for inputting responses to assessment questions.
Our application introduces a user interactive interface with precise instructions on how to answer the questions which will be used by the predictor to provide an answer to the user
3.)Prediction results with confidence scores.
When a user is done answering the questions at the end and they press the predict button they receive real time results of whether their child is autistic or has autistic traits or not
along with a confidence score.
4.)Python: Primary programming language. 
The entire program and application was written in the python programming language
5.)TensorFlow/Keras: For creating and training the neural network model.
We used the keras model classifier to create and train our model
6.)Scikit-Learn: For data preprocessing and model evaluation.
We imported various libraries that we required for the program such as LabelEncoder,RandomForestClassifier,cross_val_score,roc_auc_score,Pipeline,train_test_split, GridSearchCV,StratifiedKFold and StandardScaler
7.)Streamlit: For building the web application.
8.)Pandas and NumPy: For data manipulation and numerical calculations.
9.)Matplotlib and Seaborn: For data visualization and for our exploratory data analysis

Our application is deployed on a streamlit local host.
This is the youtube video link; https://youtu.be/ih-lLGLQzcw
