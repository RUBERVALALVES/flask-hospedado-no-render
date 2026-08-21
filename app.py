from PIL import Image
import os
from io import BytesIO
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#comando para executar no terminal para testar servidor

#curl -X POST -F "file=@ncd69.JPG" http://localhost:5000/predict

app = Flask(__name__)

# Carrega o modelo salvo previamente (ex: formato .keras ou .h5)
#MODEL_PATH = 'modelo_frango.h5'

#model = load_model(MODEL_PATH)

model = tf.keras.models.load_model('modelo_frango.h5')


# Defina a lista de classes na mesma ordem em que o modelo foi treinado
CLASS_NAMES = ['Coccidiosis', 'Newcastle', 'Sadia', 'Salmonella']
# Altere para as suas classes

# Tamanho da imagem esperado pelo seu modelo (ex: 224x224)
IMG_HEIGHT = 180
IMG_WIDTH = 180


def prepare_image(img_path):
    # Carrega a imagem redimensionando para o tamanho padrão do modelo
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    # Converte para array numpy

  #  x = tf.keras.utils.img_to_array(img)
  #  x = np.expand_dims(x, axis=0)

    x = image.img_to_array(img)

    # Adiciona a dimensão do lote (batch), transformando em (1, altura, largura, canais)
    x = np.expand_dims(x, axis=0)

    # Se o seu modelo foi treinado com normalização (ex: dividido por 255.0), aplique aqui:
 #   x = x / 255.0
    return x


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Arquivo inválido'}), 400

    # Salva temporariamente a imagem recebida
    temp_path = './temp_image.jpg'
    file.save(temp_path)

    try:

        img = tf.keras.utils.load_img(
            temp_path, target_size=(180,180)
        )

        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((180, 180)) 
        
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(

          format(100 * np.max(score))
        )






        # Prepara a imagem e roda a predição
        processed_image = prepare_image(temp_path)
        predictions = model.predict(processed_image)

        # Interpretando a classe vencedora
        predicted_index = int(np.argmax(predictions[0]))
     #   confidence = float(predictions[0][predicted_index])
        confidence = float(np.max(score))
        predicted_label = CLASS_NAMES[predicted_index]

        # Retorna o resultado em JSON
        result = {
            'classe_vencedora': predicted_label,
            'indice': predicted_index,
            'confianca': f"{confidence * 100:.2f}%"


        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Remove o arquivo temporário
        if os.path.exists(temp_path):
           os.remove(temp_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
