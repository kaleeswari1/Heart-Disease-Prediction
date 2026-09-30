from flask import Flask,render_template,request
import joblib
import pandas as pd

app=Flask(__name__)

model=joblib.load("heart_disease_random_forest.pkl")

@app.route("/",methods=["GET","POST"])
def home():
    prediction=None

    if request.method=="POST":
        data={
            "age":float(request.form["age"]),
            "sex":float(request.form["sex"]),
            "cp":float(request.form["cp"]),
            "trestbps":float(request.form["trestbps"]),
            "chol":float(request.form["chol"]),
            "fbs":float(request.form["fbs"]),
            "restecg":float(request.form["restecg"]),
            "thalach":float(request.form["thalach"]),
            "exang":float(request.form["exang"]),
            "oldpeak":float(request.form["oldpeak"]),
            "slope":float(request.form["slope"]),
            "ca":float(request.form["ca"]),
            "thal":float(request.form["thal"])
        }

        input_data=pd.DataFrame([data])
        result=model.predict(input_data)[0]

        if result==1:
            prediction="Heart Disease Detected"
        else:
            prediction="No Heart Disease Detected"

    return render_template("index.html",prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)