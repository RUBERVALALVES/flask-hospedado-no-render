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


    # Opção A: Pegando exatamente os 4 primeiros caracteres do número em notação científica
    # Correto: itera sobre cada elemento ('x') de 'probabilities'

    # 1. Extrai '9.99' para cada probabilidade no array
    # Retorna "0" se o número for menor que 1; caso contrário, extrai a mantissa

    # Se for um valor decimal comum menor que 0.1 (ex: 0.0445), retorna "0.0"
    # Caso contrário, extrai os 3 primeiros caracteres da mantissa (ex: "9.9")
    # Se for menor que 1 (ex: 0.9004522), pega o formato decimal comum "0.9"
    # Se for 1 ou mais (ou notação com mantissa >= 1), pega os 3 primeiros caracteres da notação científica "9.9"
  
    def formatar_probabilidade(x):
    # Trata 0.9004522 -> '0.9'
        if 0.1 <= x < 0.91:
            return f"{x:.1f}"
    
    # Trata decimais comuns como 0.04450481 -> '0.044' e 0.00534999 -> '0.005'
      # if 0.1 <= x < 0.95:
       #     return f"{x:.1f}"
    # Trata apenas decimais comuns entre 0.01 e 0.1 (ex: 0.04450481 -> '0.044')
        #if 0.01 <= x < 0.1:
         #   return f"{x:.3f}"
          
        if 0.001 <= x < 0.1 and not str(x).endswith(('-01', '-02')):
            return f"{x:.3f}"
      
     # Trata decimais pequenos sem notação científica como 0.00534999 -> '0.005'
        if 0.001 <= x < 0.01 and f"{x:.8e}".startswith("1.") == False:
            return f"{x:.3f}"      
    
    # Para valores em notação científica muito pequena com 3 dígitos (ex: 1.1464218e-03 -> '1.14')
        if 1e-4 <= x < 0.01 and f"{x:.8e}".split("e")[0].startswith("1."):
            return f"{x:.8e}".split("e")[0][:4]
        
    # Para todos os outros casos de notação científica (ex: 9.988e-01 -> '9.9', 5.26e-08 -> '5.2', 4.62e-06 -> '4.6')
        return f"{x:.8e}".split("e")[0][:3]

    # Aplica em todo o array de probabilidades
    apenas_digitos = [formatar_probabilidade(x) for x in probabilities]

    soma = sum(float(x) for x in apenas_digitos)
    # 2. Se você quiser apenas o primeiro valor (da classe prevista):
    apenas_digitos2 = apenas_digitos[predicted_index] 
    # ou simplesmente apenas_digitos[0]
     
    # Calcula a entropia normalizada
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    max_entropy = np.log(len(classes))
    normalized_entropy = entropy / max_entropy

  
    # Calcula a entropia da distribuição de probabilidade
    #entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    #max_entropy = np.log(len(classes))  # Maior incerteza possível
    confianca =  f"{confidence:.2f}%"
    if confidence < 85.0 or normalized_entropy > 0.50:
        return "Tipo de Imagem Invalida ou pouca confianca", confianca  #, probabilities, apenas_digitos, soma

    return classes[predicted_index], confianca  #, probabilities, apenas_digitos, soma

  
    #confianca =  f"{confidence:.2f}"
    #confianca =  f"{confidence:.2f}%"
    #if confidence < 80 or entropy / max_entropy > 0.6:
    #    return "Tipo de Imagem Invalida ou pouca confiança"
    #return classes[predicted_index],  confianca
