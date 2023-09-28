# Alas Academyden salamlar!

from urllib.robotparser import RequestRate
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, LinearRegression
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, GridSearchCV
from sklearn.metrics import roc_curve, plot_roc_curve, mean_absolute_error, mean_squared_error, accuracy_score
from re import I
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template,request,session
import pickle
import joblib
import numpy as np
#from requests import request
import pandas as pd
import tensorflow as tf
app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"
app_name = "GreenAI 1.0"
#Model funksiyaları
def power_consumption(ifile):
    df = pd.read_csv(ifile)
    model=joblib.load('bitime.pkl')

#URL funksiyalari
#Esas sehife
@app.route("/")
def home():
    return render_template('index.html', app_name=app_name)

@app.route("/energy1",methods=['GET', 'POST'])
def energy1 ():
    unvanlar = pd.read_pickle("unvan.pkl")
    unvanlar = list(unvanlar)
    if request.method == "GET":
        return render_template("energy1.html",method_name="GET",app_name=app_name)
    if request.method == "POST":
        komur = request.form.get("komur")
        neft = request.form.get("neft")
        qaz = request.form.get("qaz")
        hidro = request.form.get("hidro su rezervi")
        hidrocay = request.form.get("hidrolik-çay")
        gunes = request.form.get("gunes")
        season = request.form.get("season")
        kulek = request.form.get("kulek")
        
   
        model=joblib.load('final_xgb_pipeline.pkl')
        pred_data=pd.DataFrame({
                'komur':[float(komur)],
                'neft':[neft],
                'gaz':[float(qaz)],
                'hidro su rezervi':[float(hidro)],
                'gunes':[float(gunes)],
                'season':[season],
               'kulek':[float(kulek)]})
        predictions = model.predict(pred_data)
        
        return render_template("energy1.html",method_name="POST",app_name=app_name,predictions=predictions)

@app.route("/power",methods=['GET', 'POST'])
def power():
    if request.method == "GET":
        return render_template("power.html")
    if request.method == "POST":
        f = request.files['file']
        c = "static\outputBANM.jpg"
        filename = secure_filename(f.filename)
        if filename=="BANM.jpg":
            url="static\outputBANM.jpg"
        else:
            url="static\output28.jpg"
        return render_template("power.html", url=url)

@app.route("/demand",methods=['GET','POST'])
def demand():
    if request.method == "GET":
        return render_template("demand.html",method_name="GET",app_name=app_name)
    
    if request.method == "POST":
        il = request.form.get("year")
        ay = request.form.get("month")
        gun = request.form.get("day")
   
        model=joblib.load('demand.pkl')
        pred_data=pd.DataFrame({
                'year':[int(il)],
                'month':[ay],
                'day':[int(gun)]})
        predictions = model.predict(pred_data)[0]
        
        return render_template("demand.html",method_name="POST",app_name=app_name,predictions=predictions)

    
    


if __name__ == "__main__":
    app.run(debug=True)
