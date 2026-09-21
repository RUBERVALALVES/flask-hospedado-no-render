from flask import Flask, request, jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400

        # Obtém a string base64 e remove prefixos como "data:image/png;base64," se existirem
        img_base64 = data['image']
        if ',' in img_base64:
            img_base64 = img_base64.split(',')[1]

        # Decodifica a imagem Base64
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes))

        # --- AQUI VOCÊ SOMA OU EXECUTA SEU MODELO DE PREDIÇÃO ---
        # Exemplo: resultado = modelo.predict(image)
        resultado_predicao = "Classe Exemplo" 

        return jsonify({'prediction': resultado_predicao, 'status': 'sucesso'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
