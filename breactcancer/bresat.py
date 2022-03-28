from flask import Flask, render_template, request
import numpy as np
app = Flask(__name__)
from tensorflow.keras.models import load_model
model=load_model("breast.h5")

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/upload')
def upload():
    return render_template("upload.html")
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,9)
    print(final_features)
    prediction = model.predict_classes(final_features)
    L_collection = {0: "No Recurrence Event", 1: "Recurrence Event"}
    result=L_collection[prediction[0]]
    print(result)

    return render_template("result.html", prediction_text=f"THE BREAST CANCER IS :  {result} ")


if __name__ == "__main__":
    app.run(debug=True)
