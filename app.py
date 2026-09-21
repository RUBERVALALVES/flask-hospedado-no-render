import base64
import io
from PIL import Image
from flask import Flask, render_template, request, jsonify, redirect, flash
from werkzeug.utils import secure_filename
from main import getPrediction
import os


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

        result = getPrediction(img)

        if result == "Invalid":
            return render_template('client.html', error_message="Por favor, envie uma foto de fezes do Frango")
        else:
            return render_template('client.html', prediction=result, image='/' + file_path)
        
        resultado = getPrediction(filename)
        # -------------------------------------------------------------

        return jsonify({'status': 'sucesso', 'resultado': 'OK'}), 200

    except Exception as e:
        print(f'Erro no processamento: {str(e)}')
        return jsonify({'erro': str(e)}), 500
