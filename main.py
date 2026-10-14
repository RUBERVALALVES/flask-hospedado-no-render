import os
from ultralytics import YOLO

# Caminho para o seu modelo treinado (ex: best.pt na raiz do projeto)
MODEL_PATH = 'best.pt'

# Carrega o modelo YOLO uma única vez quando a aplicação sobe
# Certifique-se de que o arquivo .pt está committado no seu repositório ou baixado no deploy
model = YOLO(MODEL_PATH)

def getPrediction(filename):
    try:
        # Caminho completo da imagem salva pelo app.py na pasta static
        file_path = os.path.join('static', filename)
        
        # Realiza a predição com o YOLO
        # conf=0.25 (limite de confiança opcional)
        results = model(file_path, conf=0.25)
        
        # Extrai o resultado dependendo de como seu modelo foi treinado:
        # Se for um modelo de CLASSIFICAÇÃO:
        if results and len(results) > 0:
            r = results[0]
            
            # Se for YOLO Classification (ex: Coccidiosis, Salmonella, etc.)
            if hasattr(r, 'probs') and r.probs is not None:
                top1_index = r.probs.top1
                top1_conf = r.probs.top1conf.item()
                class_name = r.names[top1_index]
                
                # Opcional: Definir uma regra de validação mínima (ex: se confiança menor que 50%, retornar "Invalid")
                if top1_conf < 0.70:
                    return "Invalid"
                    
                return class_name, top1_conf
                
            # Se for YOLO Object Detection (retorna caixas delimitadoras/bounding boxes)
            elif hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                # Pega a classe com maior confiança detectada na imagem
                best_box = r.boxes[0]
                class_id = int(best_box.cls[0])
                class_name = r.names[class_id]
                return class_name
                
        return "Invalid"
        
    except Exception as e:
        print(f"Erro na predição do YOLO: {str(e)}")
        return "Invalid"
