import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

# Função de predição do modelo (substitua pela sua lógica/modelo treinado)
def predict_image(img_pil):
    # Exemplo: redimensionar e processar a imagem
    img_resized = img_pil.resize((224, 224))
    
    # --- Coloque a inferência do seu modelo aqui ---
    # resultado = model.predict(...)
    
    return "Classe Exemplo", 0.95

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtém o campo enviado pelo formulário HTML (application/x-www-form-urlencoded)
        base64_data = request.form.get('image_base64')
        
        if not base64_data:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400

        # Remove o prefixo data URI se presente
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]

        # Decodifica a string Base64 para bytes
        img_bytes = base64.b64decode(base64_data)
        
        # Converte os bytes em uma imagem PIL
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Realiza a predição
        label, confidence = predict_image(image)

        return jsonify({
            'success': True,
            'prediction': label,
            'confidence': float(confidence)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
