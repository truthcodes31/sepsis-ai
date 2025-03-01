import numpy as np
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model/sepsis_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    expected_features = model.n_features_in_
except FileNotFoundError:
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found! Make sure to train and save the model first.")
    exit(1)
except Exception as e:
    print(f"❌ ERROR: Unable to load the model: {e}")
    exit(1)


@app.route("/predict", methods=["POST"])
def predict():
    """Predict sepsis risk based on input features."""
    try:
        data = request.json
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        # Validate the number of features
        if features.shape[1] != expected_features:
            return jsonify({
                "error": f"Expected {expected_features} features, but received {features.shape[1]}."
            }), 400

        # Make prediction
        prediction = model.predict(features)[0]
        return jsonify({"sepsis_risk": int(prediction)})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__== "_main_":
    app.run(debug=True)

