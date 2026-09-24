import numpy as np
from PIL import Image
#import tflite_runtime.interpreter as tflite
import os
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="novomodelo_frango224.tflite")
#https://github.com/pradeep583/Disease_prediction/blob/main/main.py

interpreter.allocate_tensors()
  
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['Coccidiosis', 'Newcastle', 'Sadia', 'Salmonella']

def getPrediction(filename):
    SIZE = 224
    img_path = os.path.join("static", filename)

    # Load + resize
    img = Image.open(img_path).convert("RGB").resize((SIZE, SIZE))
    img_array = np.array(img, dtype=np.float32)

    # Handle input dtype 
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.float32:
        img = np.asarray(img, dtype=np.float32) / 255.0
    elif input_dtype == np.uint8:
        img = np.asarray(img, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported dtype: {input_dtype}")

    #img = np.expand_dims(img, axis=0)
    img_array = np.expand_dims(img_array, axis=0)
       
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    # Se o modelo for quantizado (uint8), de-quantizar para obter probabilidades
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        if scale > 0:
            output_data = scale * (output_data.astype(np.float32) - zero_point)

    # Aplica Softmax caso a saída sejam logits (opcional, mas recomendado)
    exp_preds = np.exp(output_data - np.max(output_data))
    probabilities = exp_preds / np.sum(exp_preds)

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index]*100)

    print(f"Predições/Probabilidades: {probabilities}") # Para você depurar no terminal
    print(f"Classe detectada: {classes[predicted_index]} com confiança {confidence:.2f}")


    # Função para extrair os dígitos sem arredondar para 1.00
def extrair_texto_original(arr, casas=2):
    resultado = []
    for x in arr:
        # 1. Pega a parte antes do 'e' (ex: "9.9999285")
        parte_num = str(x).split('e')[0]
        
        # 2. Se for menor que 1 (ex: "0.99999285"), remove o "0." para pegar os números reais
        if parte_num.startswith('0.'):
            parte_num = parte_num[2:]
            
        # 3. Pega o primeiro dígito, põe um ponto, e pega os próximos dígitos
        # Isso garante que "0.99999285" vire "9.99" em vez de arredondar para "1.00"
        digitos = parte_num.replace('.', '')
        texto = f"{digitos[0]}.{digitos[1:1+casas]}"
        resultado.append(texto)
        
    return resultado

    # Aplicando no seu array dinâmico:
    apenas_digitos = extrair_texto_original(probabilities)


  
   
    #apenas_digitos = [f"{x:.2e}".split('e')[0] for x in probabilities]
    #apenas_digitos1 = apenas_digitos[0]
    #soma = sum(float(x) for x in apenas_digitos)  

    # Calcula a entropia normalizada
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    max_entropy = np.log(len(classes))
    normalized_entropy = entropy / max_entropy

  
    # Calcula a entropia da distribuição de probabilidade
    #entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    #max_entropy = np.log(len(classes))  # Maior incerteza possível
    confianca =  f"{confidence:.2f}%"
    if confidence < 85.0 or normalized_entropy > 0.50:
        return "Tipo de Imagem Invalida ou pouca confiança", confianca, probabilities, apenas_digitos, soma

    return classes[predicted_index], confianca, probabilities, apenas_digitos, soma

  
    #confianca =  f"{confidence:.2f}"
    #confianca =  f"{confidence:.2f}%"
    #if confidence < 80 or entropy / max_entropy > 0.6:
    #    return "Tipo de Imagem Invalida ou pouca confiança"
    #return classes[predicted_index],  confianca
