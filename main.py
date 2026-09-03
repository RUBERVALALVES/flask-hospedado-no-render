import numpy as np
from PIL import Image
#import tflite_runtime.interpreter as tflite
import os
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="modelo_otimizado.tflite")
#https://github.com/pradeep583/Disease_prediction/blob/main/main.py

# Load TFLite model
#interpreter = tflite.Interpreter(model_path="modelo_otimizado.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['Coccidiosis', 'Newcastle', 'Sadia', 'Salmonella']
#CLASS_NAMES = ['Coccidiosis', 'Newcastle', 'Sadia', 'Salmonella']

def getPrediction(filename):
    SIZE = 180
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
    confidence = float(probabilities[predicted_index])

    print(f"Predições/Probabilidades: {probabilities}") # Para você depurar no terminal
    print(f"Classe detectada: {classes[predicted_index]} com confiança {confidence:.2f}")


     
    
    
    #predicted_index = int(np.argmax(output_data))
    #confidence = float(output_data[predicted_index])

    if confidence < 0.5:  
        return "Imagem Invalida"
    return classes[predicted_index],  confidence, probabilities

