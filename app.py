import os
import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
# import cv2  # Caso use OpenCV
# import numpy as np

app = Flask(__name__)

# Função fictícia para simular sua IA/Modelo
def predizer_imagem(image):
    # Aqui entra o seu modelo (ex: model.predict)
    # Exemplo simples usando a biblioteca PIL:
    largura, altura = image.size
    return f"Imagem processada com sucesso. Resolução: {largura}x{altura}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        dados = request.get_json()
        if not dados or 'imagem_base64' not in dados:
            return jsonify({'erro': 'Nenhuma imagem enviada'}), 400
        
        # Remove cabeçalhos de dados se existirem (ex: "data:image/jpeg;base64,")
        img_data = dados['imagem_base64']
        if "," in img_data:
            img_data = img_data.split(",")[1]

        # Decodifica a string Base64 para Bytes
        conteudo_imagem = base64.b64decode(img_data)
        
        # Converte para o formato PIL Image (útil para a maioria das IAs)
        imagem = Image.open(io.BytesIO(conteudo_imagem))

        # Se precisar converter para formato OpenCV (BGR):
        # imagem_np = np.frombuffer(conteudo_imagem, dtype=np.uint8)
        # img_cv2 = cv2.imdecode(imagem_np, cv2.IMREAD_COLOR)

        # Executa a predição
        resultado = predizer_imagem(imagem)

        return jsonify({'status': 'sucesso', 'predicao': resultado}), 200

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    # O Render configura a porta automaticamente via variável de ambiente
    porta = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=porta)
