import base64
import io
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Pega a string Base64 do campo 'image'
        base64_string = request.form.get('image')

        if not base64_string:
            return jsonify({'erro': 'Nenhuma imagem recebida'}), 400

        # Remove prefixos como 'data:image/jpeg;base64,' se existirem
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # CORREÇÃO 1: Trata espaços que o envio urlencoded pode ter trocado do '+'
        base64_string = base64_string.strip().replace(' ', '+')

        # CORREÇÃO 2: Ajusta o PADDING faltando (garante comprimento múltiplo de 4)
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string += '=' * (4 - missing_padding)

        # 2. Decodifica para Bytes e abre a imagem
        img_bytes = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # -------------------------------------------------------------
        # 3. Coloque aqui a chamada para o seu modelo de predição
        # resultado = modelo.predict(img)
        # -------------------------------------------------------------

        return jsonify({'status': 'sucesso', 'resultado': 'OK'}), 200

    except Exception as e:
        print(f'Erro no processamento: {str(e)}')
        return jsonify({'erro': str(e)}), 500
