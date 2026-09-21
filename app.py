from flask import Flask, request, jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # 1. Valida se o JSON foi recebido e se contém a chave 'image'
        if not data or 'image' not in data:
            return jsonify({'error': 'Nenhuma imagem enviada no campo "image"'}), 400

        # 2. Atribui a variável logo após a validação
        img_base64 = data['image']

        # Remove prefixo de data URI, quebras de linha e espaços
        if ',' in img_base64:
            img_base64 = img_base64.split(',')[1]

        img_base64 = img_base64.replace('\n', '').replace('\r', '').strip()

        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes))
        
        if not img_base64:
            return jsonify({'error': 'A string da imagem está vazia'}), 400

        # 5. Decodifica e carrega a imagem
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes))

        # --- SEU MODELO DE PREDIÇÃO AQUI ---
        # resultado = modelo.predict(image)
        resultado = "Predição realizada com sucesso!"

        return jsonify({'prediction': resultado, 'status': 'sucesso'}), 200

    except Exception as e:
        return jsonify({'error': f"Erro interno: {str(e)}"}), 500
