from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
   return render_template("home.html")

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        sepal_length = request.form.get('sepal_length')
        sepal_width = request.form.get('sepal_width')
        petal_length = request.form.get('petal_length')
        petal_width = request.form.get('petal_width')
        try:
            prediction = preprocessDataAndPredict(sepal_length,sepal_width, petal_length, petal_width)
            return render_template('predict.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return "Please Enter valid values"

def preprocessDataAndPredict(sepal_length, sepal_width, petal_length, petal_width):
    
    test_data = [sepal_length, sepal_width, petal_length, petal_width]
    print(test_data)
    test_data = np.array(test_data)
    test_data = test_data.reshape(1,-1)
    print(test_data)
    file = open("model.pkl","rb")
    trained_model = pickle.load(file)
    prediction = trained_model.predict(test_data)
    return prediction

if __name__=="__main__":
    app.run(debug=True)