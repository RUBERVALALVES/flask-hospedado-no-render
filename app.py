import base64
import io
import uuid
from PIL import Image
from flask import Flask, render_template, request, jsonify, redirect, flash
from werkzeug.utils import secure_filename
from main import getPrediction
import os

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = 'secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_static_folder():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Failed to delete {file_path} — {e}")

@app.route('/')
def index():
    return render_template('client.html')

@app.route('/', methods=['POST'])
def submit_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Delete any existing file in static/
        cleanup_static_folder()

        # Save new file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)

        result = getPrediction(filename)

        if result == "Invalid":
            return render_template('client.html', error_message="Por favor, envie uma foto de fezes do Frango")
        else:
            return render_template('client.html', prediction=result, image='/' + file_path)


       

    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)




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

        # 3. Limpa a pasta static (se desejar manter o comportamento da outra rota)
        cleanup_static_folder()

        # 4. Gera um nome único para o arquivo e salva em static/
        filename = f"upload_{uuid.uuid4().hex[:8]}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) 

        # Salva a imagem usando a biblioteca Pillow
        img.save(file_path, 'JPEG')

        # 5. Executa a predição passando o arquivo salvo
        result = getPrediction(filename)

        # -------------------------------------------------------------
        # 3. Coloque aqui a chamada para o seu modelo de predição

        #result = getPrediction(img)

        #if result == "Invalid":
         #   return render_template('client.html', error_message="Por favor, envie uma foto de fezes do Frango")
        #else:
         #   return render_template('client.html', prediction=result, image='/' + file_path)
        
        #resultado = getPrediction(filename)
        # -------------------------------------------------------------

        return jsonify({
            'status': 'sucesso',
            'resultado': result,
            'imagem_url': f'/{file_path}'
        }), 200
     
    except Exception as e:
        print(f'Erro no processamento: {str(e)}')
        return jsonify({'erro': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
