import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd

def power(f):
    df = pd.read_csv(f)
    X_train=[]
        
    for i in range(20, len(df)):
         X_train.append(df.values[i-20: i, 0])
        
    model=joblib.load('bitime.pkl', "r")
    y=model.predict(X_train)
    plt.figure(figsize=(30,8))

    plt.xticks(rotation = 45)
    plt.plot(y, "*")


    plt.savefig("./static/plot.png")
    
    #Streamlit funksiyalari
file = st.fileuploader("fayl")
    
power(file)
