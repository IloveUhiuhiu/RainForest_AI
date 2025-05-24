from flask import Flask, request, jsonify
from flask_cors import CORS
from model_ai import predict_image
from utils.create_spectrogram import create_spectrogram
from tensorflow.keras.preprocessing import image
app = Flask(__name__)
CORS(app)

@app.route('/', methods= ['GET'])
def index():
    return jsonify({"message": "Hello World"}), 200

@app.route('/ai/recognize-sounds', methods=['POST'])
def recognize_sounds():
    try:
        audio_data = request.files.get('audio')
        if not audio_data:
            return jsonify({"error": "No audio file provided"}), 400
        
        create_spectrogram(audio_data)
        img = image.load_img('static/spectrogram.png', target_size=(224, 224))

        if img is None:
            return jsonify({"error": "Failed to create spectrogram"}), 500
        predictions = str(predict_image(img))
        return jsonify({"message": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)