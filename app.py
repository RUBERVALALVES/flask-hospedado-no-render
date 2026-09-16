from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from main import getPrediction
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def submit_file():
    # Suporte para envio direto via PostFile (dados brutos)
    if request.data:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(request.data)
        
        # Executa a predição da foto enviada
        result = getPrediction('temp_image.jpg')
        
        if result == "Invalid":
            return jsonify({'status': 'error', 'message': 'Por favor, envie uma foto válida de fezes do frango'}), 200
            
        return jsonify({'status': 'success', 'prediction': result}), 200

    return jsonify({'status': 'error', 'message': 'Nenhum arquivo enviado'}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
