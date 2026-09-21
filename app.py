import base64
import io
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pega a string enviada pelo PostarTexto
        base64_string = request.form.get('image')

        if not base64_string:
            return jsonify({'erro': 'Nenhuma imagem recebida'}), 400

        # Converte a Base64 para PIL Image
        img_bytes = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Predição do seu modelo
        # resultado = modelo.predict(img)

        return jsonify({'status': 'sucesso', 'resultado': 'OK'}), 200

    except Exception as e:
        return jsonify({'erro': str(e)}), 500
