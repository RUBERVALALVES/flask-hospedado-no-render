from flask import Flask, request, jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lê o campo 'image' enviado via formulário POST
        img_base64 = request.form.get('image')

        if not img_base64:
            return jsonify({'error': 'Nenhuma imagem foi recebida no campo "image"'}), 400

        # Limpa o prefixo Data URI se presente (ex: data:image/png;base64,)
        if ',' in img_base64:
            img_base64 = img_base64.split(',')[1]

        # Limpa quebras de linha e espaços que o Android pode gerar
        img_base64 = img_base64.replace('\n', '').replace('\r', '').strip()

        # Decodifica e processa a imagem
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes))

        # --- SEU MODELO DE PREDIÇÃO AQUI ---
        # predicao = modelo.predict(image)
        resultado = "Predição realizada com sucesso!"

        return jsonify({'prediction': resultado, 'status': 'sucesso'}), 200

    except Exception as e:
        return jsonify({'error': f"Erro ao processar imagem: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
