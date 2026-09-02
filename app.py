import gc
import os
# Desativa o uso de GPU (Render CPU não possui GPU nos planos básicos)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Reduz o nível de log do TF para economizar processamento
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import tensorflow.keras.preprocessing.image
# Força o TF a usar menos memória e otimizar para CPU
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('modelo_frango.h5')

# Class labels
class_labels = ['Coccidiosis', 'Newcastle', 'Sadia', 'Salmonella']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):

    IMAGE_SIZE = 180
    img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    #img_array = tensorflow.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    tf.keras.backend.clear_session()
    gc.collect()
    
    if class_labels[predicted_class_index] == 'Sadia':
        return "Ave Saudavel", confidence_score
    else:
        return f"Doença: {class_labels[predicted_class_index]}", confidence_score

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence = predict_tumor(file_location)

            # Return result along with image path for display
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
