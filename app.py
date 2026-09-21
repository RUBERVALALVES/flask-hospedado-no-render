import base64
import io
import re
from PIL import Image

# ... dentro da rota /predict ...

# 1. Obter os dados (funciona para Form Data ou JSON)
img_base64 = request.form.get('image') or (request.json and request.json.get('image'))

if not img_base64:
    return jsonify({'error': 'Nenhuma imagem foi recebida no campo "image"'}), 400

# 2. Remover cabeçalho Data URI se presente (ex: data:image/png;base64,)
if ',' in img_base64:
    img_base64 = img_base64.split(',')[1]

# 3. Tratar espaço substituído pelo Android (+) e quebras de linha
img_base64 = img_base64.replace(' ', '+').replace('\n', '').replace('\r', '').strip()

# 4. Ajustar Padding do Base64 se estiver incompleto
missing_padding = len(img_base64) % 4
if missing_padding:
    img_base64 += '=' * (4 - missing_padding)

try:
    # 5. Decodificar bytes e abrir a imagem
    img_bytes = base64.b64decode(img_base64)
    image = Image.open(io.BytesIO(img_bytes))
    
    # Força o carregamento dos dados da imagem para validar a integridade
    image.verify() 
    # Recarrega a imagem para uso posterior (verify fecha o ponteiro do arquivo)
    image = Image.open(io.BytesIO(img_bytes))

except Exception as decode_err:
    return jsonify({'error': f"Erro na decodificação da imagem: {str(decode_err)}"}), 400
