import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.php')

@app.route('/predict',methods=['POST'])
def predict():
   
    get = [float(x) for x in request.form.values()]  
    final_features = [np.array(get)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    print(output)
    if(output==1):
       return render_template('index.php', prediction_text='RESULT IS POSITIVE')
    else:
          return render_template('index2.php', prediction_text='RESULT IS NEGATIVE')  
if __name__ == "__main__":
    app.run(debug=False)
