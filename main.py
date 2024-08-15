from flask import Flask,request,jsonify
import pickle
import numpy as np

model1 = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('modelB.pkl', 'rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict1',methods=['POST'])
def predict1():
    flux=request.form.get('flux')
    input_query=np.array([flux],dtype=float)
    result=model1.predict(input_query)[0]
    if result>=0.5:
        res=1
    else:
        res=0

    return jsonify({'Result': str(res)})

@app.route('/predict2',methods=['POST'])
def predict2():
    flux=request.form.get('flux')
    input_query=np.array([flux],dtype=float)
    
    result=model2.predict(input_query)[0]
    if result>=0.5:
        res=1
    else:
        res=0

    return jsonify({'Result': str(res)})

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')