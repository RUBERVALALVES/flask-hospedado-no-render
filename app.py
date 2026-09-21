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

        img_base64 = data['image']

        # 1. Remove cabeçalho de data URL se existir (ex: data:image/jpeg;base64,)
        if ',' in img_base64:
            img_base64 = img_base64.split(',')[1]

        # 2. Remove quebras de linha e espaços que o KIO4_Base64 pode gerar
        img_base64 = img_base64.replace('\n', '').replace('\r', '').strip()

        # 3. Decodifica a string Base64
        img_bytes = base64.b64decode(img_base64)

        # 4. Abre a imagem usando PIL
        image = Image.open(io.BytesIO(img_bytes))
        image.verify() # Valida se é realmente uma imagem válida
        
        # Recarrega a imagem para uso no modelo após a verificação
        image = Image.open(io.BytesIO(img_bytes))

        # --- SEU MODELO DE PREDIÇÃO AQUI ---
        # resultado = modelo.predict(image)
        resultado = "Imagem processada com sucesso!"

        return jsonify({'prediction': resultado, 'status': 'sucesso'}), 200

    except Exception as e:
        # Retorna o erro detalhado para facilitar o diagnóstico
        return jsonify({'error': f"Erro ao processar imagem: {str(e)}"}), 500
