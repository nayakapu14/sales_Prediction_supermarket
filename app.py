# app.py
from flask import Flask, request, jsonify, render_template
import pickle, json, os

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "monthly_model.pkl")
META_PATH = os.path.join(BASE, "meta.json")

# Load model & meta
if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    raise FileNotFoundError("monthly_model.pkl or meta.json not found. Run train_model.py first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(META_PATH, "r") as f:
    meta = json.load(f)

def year_month_to_index(year, month):
    start_year = int(meta['start_year'])
    start_month = int(meta['start_month'])
    return (year - start_year) * 12 + (month - start_month)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    try:
        year = int(data.get("year"))
        month = int(data.get("month"))
    except Exception:
        return jsonify({"error":"Please send integers 'year' and 'month' in JSON body."}), 400
    idx = year_month_to_index(year, month)
    # model expects 2D array-like
    pred = float(model.predict([[idx]])[0])
    return jsonify({"year":year, "month":month, "predicted_monthly_sales": pred})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
