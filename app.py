import base64
import io
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Captura a string do formulário 'image'
        base64_string = request.form.get('image')

        if not base64_string:
            return jsonify({'erro': 'Nenhuma imagem recebida'}), 400

        # Remove prefixos do tipo data:image/jpeg;base64,
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # CORREÇÃO 1: Corrige espaços trocados pelo envio na URL
        base64_string = base64_string.strip().replace(' ', '+')

        # CORREÇÃO 2: Ajusta o PADDING faltando (adiciona '=' até o tamanho ser múltiplo de 4)
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string += '=' * (4 - missing_padding)

        # 2. Decodifica os bytes e abre com a PIL
        img_bytes = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # -------------------------------------------------------------
        # 3. Lógica do seu modelo de predição
        # resultado = modelo.predict(img)
        # -------------------------------------------------------------

        return (
            jsonify({'status': 'sucesso', 'mensagem': 'Imagem decodificada!'}),
            200,
        )

    except Exception as e:
        print(f'Erro de processamento: {str(e)}')
        return jsonify({'erro': str(e)}), 500
