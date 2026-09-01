import os
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Configura a pasta onde as imagens enviadas serão salvas temporariamente
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Garante que a pasta de uploads exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Carrega o modelo pré-treinado MobileNetV2 do TensorFlow
# Ele é capaz de reconhecer 1.000 categorias de objetos diferentes
MODEL_PATH = 'modelo_frango.keras'
model = load_model(MODEL_PATH)

Defina a lista de classes na mesma ordem em que o modelo foi treinado
CLASS_NAMES = ['Coccidiosis', 'Newcastle', 'Sadia', 'Salmonella']

def predict_image(img_path):
    # O MobileNetV2 exige imagens no tamanho 224x224
    img = image.load_img(img_path, target_size=(180, 180))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Executa a previsão do modelo
  #  preds = model.predict(x)
    # Decodifica os top 3 resultados (Classe, Descrição, Probabilidade)
  #  decoded_preds = decode_predictions(preds, top=3)[0]

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])


#@app.route('/')
#def index():
 #   return render_template('index.html')

    
    # Formata os resultados para envio fácil para o HTML
#    results = []
 #   for imagenet_id, label, prob in decoded_preds:
  #      results.append({"label": label, "prob": f"{prob * 100:.2f}%"})
   # return results

@app.route('/predict', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Verifica se o arquivo foi enviado no formulário HTML
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Salva o arquivo localmente na pasta static/uploads
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Executa a predição usando o TensorFlow
            predictions = predict_image(filepath)
            
            # Renderiza a página enviando o caminho da imagem e os resultados da IA
            return render_template('index.html', 
                                   image_path=filepath, 
                                   predictions=predictions)
            
    return render_template('index.html', image_path=None, predictions=None)

if __name__ == '__main__':
    app.run(debug=True)
