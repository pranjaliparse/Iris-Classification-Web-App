#Flask - A microframework for python
from flask import Flask, render_template, url_for, request
#render_template - Render a template from the template folder with a given context
#url_for - Generates a URL to the given endpoint with the method provided

from flask_material import Material 

# Exploratory Data Analysis Packages
import pandas as pd
#Providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language
import numpy as np

import pickle
#allows to serialize python object into a file

result_predictions=0
result_prediction=""

# Machine Learning Packages
from sklearn.externals import joblib
#Joblib - a set of tools to provide light weight pipelining in python
#More efficient on objects that carry large numpy arrays


#create an instance of the Flask class
app = Flask(__name__)
#place holder for current module
Material(app)

@app.route('/')
def index():
    return render_template('index.html')
    #Reading from that template


@app.route('/', methods=['POST'])
def analyze():
    if request.method == 'POST':
        #POST request method requests that a web server accepts the data enclosed in the body of the request message
        #Most likely for storing it
        sepal_length = request.form['sepal_length']
        petal_length = request.form['petal_length']
        sepal_width = request.form['sepal_width']
        petal_width = request.form['petal_width']
        model_choice = request.form['model_choice']
        sample_data=[sepal_length, sepal_width, petal_length, petal_width]

        # Clean the data by converting from Unicode to Float
        clean_data= [float(i) for i in sample_data]
        #Reshaping 
        ex1 = np.array(clean_data).reshape(1,-1)

        #Conditional for ML
        if model_choice == 'logitmodel':
            with open('log_iris_pickle','rb') as f:
                log=pickle.load(f)
            result_predictions = log.predict(ex1)
            #print(result_predictions)


        elif model_choice == 'gnbmodel':
            with open('gnb_iris_pickle','rb') as f:
                gnb=pickle.load(f)
            #pickle file is a binary file
            result_predictions = gnb.predict(ex1)
           

        elif model_choice == 'svmmodel':
            with open('svm_iris_pickle','rb') as f:
                svm=pickle.load(f)
            result_predictions=svm.predict(ex1)


        if(result_predictions==1):
            result_prediction='setosa'
        elif(result_predictions==2):
            result_prediction='versicolor'
        elif(result_predictions==3):
            result_prediction='virginica'


    return render_template('index.html', sepal_length=sepal_length, 
    sepal_width=sepal_width,
    petal_length=petal_length,
    petal_width=petal_width,
    model_selected=model_choice,
    result_prediction=result_prediction,
    clean_data=clean_data)

if __name__=='__main__':
    app.run(debug=True)#Not loading the server multiple times
