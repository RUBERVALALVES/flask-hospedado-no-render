import base64
import os
from flask import Flask, flash, redirect, render_template, request
from werkzeug.utils import secure_filename
from main import getPrediction

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = 'secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def cleanup_static_folder():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Failed to delete {file_path} — {e}")

@app.route('/', methods=['POST'])
def submit_file():
    # Verifica se os dados vieram no formato JSON (Base64 do App Inventor)
    if request.is_json:
        data = request.get_json()
        base64_string = data.get('image')

        if not base64_string:
            return {"error": "Nenhuma imagem enviada"}, 400

        # Remove o cabeçalho data URI se o App Inventor enviar (ex: "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        try:
            # Decodifica a string Base64 para bytes
            image_bytes = base64.b64decode(base64_string)
            
            cleanup_static_folder()
            
            filename = 'uploaded_image.jpg'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Salva o arquivo de imagem no servidor
            with open(file_path, 'wb') as f:
                f.write(image_bytes)

            result = getPrediction(filename)

            if result == "Invalid":
                return {"error": "Por favor, envie uma foto de fezes do Frango"}, 400

            # Retorna o resultado em JSON para o App Inventor
            return {
                "prediction": result,
                "image": '/' + file_path
            }, 200

        except Exception as e:
            return {"error": f"Erro ao processar imagem: {str(e)}"}, 500

    # Mantém o suporte antigo para requisições multipart/form-data via HTML
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
            
        cleanup_static_folder()
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        
        result = getPrediction(filename)
        if result == "Invalid":
            return render_template('client.html', error_message="Por favor, envie uma foto de fezes do Frango")
        else:
            return render_template('client.html', prediction=result, image='/' + file_path)

    return {"error": "Formato de requisição não suportado"}, 400
