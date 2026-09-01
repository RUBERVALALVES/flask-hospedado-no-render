import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

# Carrega o modelo de aprendizado de máquina
model = tf.keras.models.load_model('modelo_frango.h5')


# Ajuste as dimensões de acordo com o seu modelo
IMG_WIDTH, IMG_HEIGHT = 180, 180 

def preprocess_image(img_bytes):
    # Abre a imagem a partir dos bytes recebidos
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensiona para o tamanho esperado pelo modelo
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Converte para array numpy e normaliza (ajuste se seu modelo exige normalização diferente)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    #img_array /= 255.0  
    
    return img_array

@app.route('/', methods=['GET'])
def index():
    # Renderiza a interface web
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    try:
        img_bytes = file.read()
        processed_img = preprocess_image(img_bytes)
        
        # Executa a predição
        predictions = model.predict(processed_img)
        
        # Exemplo para classificação binária ou multiclasse
        # Ajuste a extração da classe/probabilidade de acordo com a saída do seu modelo
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
     if __name__ == '__main__':
         # Obtém a porta configurada no ambiente (essencial para o Render)
         port = int(os.environ.get('PORT', 5000))
         app.run(host='0.0.0.0', port=port)
