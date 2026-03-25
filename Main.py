import streamlit as streamlit
import panda as pd

from sklearn.model_selection import train_test_split
from sklearn.Linear_model import LinearRegression
df=pd.read_csv("data.csv")
X=df[["HoursStudied"]] 
y=[["ExamScore"]]
X_Train,X_test,y_train,y_test=train_test_split(X,y,test_sized=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
st.title("Exam Score Predictor")
st.write("Enter hours studied topredict the exam score")
hours=st.number_input("Hours Studied", min_value=0.0,step=0.1)
if st.button("Predict score")
predicted_score=model.preddict([[hours]])[0]
st.success(f"Predicted score:{predicted_score:.2f}")
st.write("Sample Training"
st.dataframe(df)
