import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Carrega o modelo uma única vez na inicialização
MODEL_PATH = "modelo_frango.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def prepare_image(image_bytes):
    # Altere o tamanho (180, 180) para o formato exigido pelo seu modelo
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
 #   return img_array / 255.0  # Normalização (se o seu modelo exigir)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
        
    file = request.files["file"]
    try:
        image_bytes = file.read()
        prepared_img = prepare_image(image_bytes)
        
        # Realiza a predição
        predictions = model.predict(prepared_img)
        prediction_list = predictions.tolist()
        
        return jsonify({"predictions": prediction_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # O Render define a porta automaticamente através da variável de ambiente PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

