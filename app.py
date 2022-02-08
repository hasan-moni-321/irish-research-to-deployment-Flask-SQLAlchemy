
import pickle
from unittest import result
import numpy as np 
from crypt import methods
from flask import Flask, render_template, request

from sqlalchemy import Float, create_engine, MetaData, Table, Column, Integer, String
engine = create_engine('sqlite:///college.db', echo=True) 
meta = MetaData() 

app = Flask(__name__) 

students = Table(
    'student', meta, 
    Column('id', Integer, primary_key=True), 
    Column('sepal_length', Float), 
    Column('sepal_width', Float), 
    Column('petal_length', Float), 
    Column('petal_width', Float), 
    Column('predicted_class', String)  
)
meta.create_all(engine) 


# model loading 
file_name = "logistic_model.sav"
loaded_model = pickle.load(open(file_name, 'rb'))

def prediction(s_l, s_w, p_l, p_w, loaded_model): 
    pre_data = np.array([s_l, s_w, p_l, p_w]) 
    pre_data_reshape = pre_data.reshape(1, -1) 
    pred_result = loaded_model.predict(pre_data_reshape)  
    return pred_result[0]
 

@app.route('/')
def input_data(): 
    return render_template('input.html') 


@app.route('/result', methods=["POST", "GET"]) 
def input(): 
    if request.method == "POST": 
        s_l = request.form['sepal_length']
        s_w = request.form['sepal_width']
        p_l = request.form['petal_length'] 
        p_w = request.form['petal_width'] 

        predicted_result = prediction(s_l, s_w, p_l, p_w, loaded_model)  

        try: 
            ins = students.insert().values(sepal_length = s_l, sepal_width = s_w, petal_length = p_l, petal_width = p_w) 
            conn = engine.connect() 
            result = conn.execute(ins) 
            msg = 'data successfully inserted into database' 
        except: 
            msg = 'oops! something wrong in insertion of the data' 
        finally:
            return render_template('result.html', predicted = predicted_result, msg = msg)  

if __name__ == '__main__': 
    app.run(debug=True) 


