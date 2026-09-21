import base64
import io
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        base64_string = request.form.get('image')

        print('Primeiros 50 caracteres recebidos:', base64_string[:50] if base64_string else 'VAZIO')
        if not base64_string:
            return jsonify({'erro': 'Nenhum dado enviado na chave image'}), 400

        # Fix 1: Se a string contiver 'data:image/jpeg;base64,...', pega só o que vem depois da vírgula
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Fix 2: Remove quebras de linha ou espaços que possam ter vindo na requisição
        base64_string = base64_string.strip().replace(' ', '+')

        # Converte a string limpa para Bytes
        img_bytes = base64.b64decode(base64_string)

        # Abre a imagem com PIL
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # --- Lógica do seu modelo ---
        # resultado = modelo.predict(img)

        return (
            jsonify({'status': 'sucesso', 'mensagem': 'Imagem identificada!'}),
            200,
        )

    except Exception as e:
        # Exibe o erro exato no log do Render
        print(f'Erro no processamento: {str(e)}')
        return jsonify({'erro': str(e)}), 500
