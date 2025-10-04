from flask import Flask, request, render_template
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = scaler.transform([features])

        # Predict probability of churn
        prob = model.predict_proba(final_features)[0][1]
        prediction = 1 if prob > 0.4 else 0   # using threshold 0.4 for sensitivity

        if prediction == 1:
            output = f"⚠️ Customer likely to churn (Probability: {prob:.2f})"
        else:
            output = f"✅ Customer will stay (Probability: {prob:.2f})"

        return render_template("index.html", prediction_text=output)
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
