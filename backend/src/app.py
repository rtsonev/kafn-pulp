from flask import Flask, request, jsonify
from util import preprocess
from fake_news_detector import FakeNewsDetector


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    if not content or "text" not in content:
        return jsonify({"error": "No text provided"}), 400

    text = preprocess(content["text"])
    detector = FakeNewsDetector("backend/model/0.25/kafn_pulp_doc2vec_model_config.json")
    pred_label, prob = detector.predict_text([text])[0]

    return (
        jsonify({"test result": {"label": pred_label, "probability": float(prob)}}),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=False)
