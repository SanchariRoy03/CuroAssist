from flask import Flask, render_template, request
import  numpy as mp
import pandas as pd
import  pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        sympotoms = request.form.get('sympotoms')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

if __name__ == "__main__":
    app.run(debug=True)
