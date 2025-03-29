import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import io # Para manejar la imagen en memoria si es necesario

# --- Configuración ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MODEL_PATH = os.path.join('model', 'modelo_VGG16_v2.keras') # Asegúrate que el nombre coincida
LABELS_PATH = 'labels.txt'
# ¡IMPORTANTE! Ajusta este tamaño al que espera tu modelo Keras
# Revisa model.input_shape en tu notebook de entrenamiento, ej: (None, 224, 224, 3) -> (224, 224)
EXPECTED_IMAGE_SIZE = (224, 224)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'una-clave-secreta-muy-fuerte' # Cambia esto en producción real
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Límite de 16MB para subidas

# --- Carga del Modelo y Etiquetas ---
try:
    print(f"Cargando modelo desde: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado exitosamente.")
    # Verifica la forma de entrada esperada por el modelo
    input_shape = model.input_shape
    print(f"Forma de entrada esperada por el modelo: {input_shape}")
    # Extrae el tamaño si es necesario (asumiendo formato BHWC o HWC)
    if len(input_shape) == 4: # Formato (Batch, Height, Width, Channels)
        EXPECTED_IMAGE_SIZE = tuple(input_shape[1:3])
    elif len(input_shape) == 3: # Formato (Height, Width, Channels)
        EXPECTED_IMAGE_SIZE = tuple(input_shape[0:2])
    else:
        print("ADVERTENCIA: No se pudo determinar automáticamente el tamaño de imagen esperado. Usando el valor por defecto.")
    print(f"Tamaño de imagen esperado ajustado a: {EXPECTED_IMAGE_SIZE}")

except Exception as e:
    print(f"Error al cargar el modelo Keras: {e}")
    # Podrías querer detener la app aquí o manejarlo de otra forma
    model = None # Marcar que el modelo no se cargó

try:
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        class_labels = [line.strip() for line in f.readlines()]
    print(f"Etiquetas cargadas: {len(class_labels)} clases.")
    # Verifica si el número de etiquetas coincide con la salida del modelo
    if model and model.output_shape[-1] != len(class_labels):
         print(f"ADVERTENCIA: El número de clases en el modelo ({model.output_shape[-1]}) no coincide con el número de etiquetas en {LABELS_PATH} ({len(class_labels)}).")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo de etiquetas en {LABELS_PATH}")
    class_labels = []
except Exception as e:
    print(f"Error al leer el archivo de etiquetas: {e}")
    class_labels = []


# --- Funciones Auxiliares ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size):
    """Preprocesa la imagen para que coincida con la entrada del modelo."""
    try:
        img = Image.open(image_path).convert('RGB') # Asegurar que sea RGB
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # --- Normalización ---
        # ¡IMPORTANTE! Ajusta esto según cómo entrenaste tu modelo.
        # Opciones comunes:
        # 1. Escalar a [0, 1]: img_array = img_array / 255.0
        # 2. Escalar a [-1, 1]: img_array = (img_array / 127.5) - 1
        # 3. Usar preprocesamiento específico del modelo (ej. ResNet):
        #    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        # Usaremos la opción 1 (escala [0, 1]) como ejemplo:
        img_array = img_array / 255.0

        # --- Añadir dimensión de Batch ---
        img_array = np.expand_dims(img_array, axis=0) # Shape: (1, height, width, channels)
        return img_array
    except Exception as e:
        print(f"Error al preprocesar la imagen {image_path}: {e}")
        return None

# --- Rutas de la Aplicación ---
@app.route('/')
def index():
    """Muestra la página principal para subir la imagen."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recibe la imagen, la procesa, realiza la predicción y muestra el resultado."""
    if model is None:
        flash("Error: El modelo de IA no pudo ser cargado.", "error")
        return redirect(url_for('index'))
    if not class_labels:
        flash("Error: Las etiquetas de las clases no pudieron ser cargadas.", "error")
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No se encontró el archivo en la solicitud', 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No se seleccionó ningún archivo', 'warning')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Crear carpeta uploads si no existe
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            print(f"Archivo guardado en: {filepath}")

            # Preprocesar la imagen
            processed_image = preprocess_image(filepath, EXPECTED_IMAGE_SIZE)

            if processed_image is None:
                flash("Error al procesar la imagen.", "error")
                # Considera borrar el archivo si falla el preprocesamiento
                # if os.path.exists(filepath):
                #     os.remove(filepath)
                return redirect(url_for('index'))

            # Realizar la predicción
            print("Realizando predicción...")
            predictions = model.predict(processed_image)
            print(f"Predicciones raw: {predictions}")

            # Obtener la clase con mayor probabilidad
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) # Convertir a float nativo

            if predicted_class_index < len(class_labels):
                predicted_label = class_labels[predicted_class_index]
            else:
                 predicted_label = f"Clase desconocida (índice {predicted_class_index})"
                 print(f"ADVERTENCIA: Índice predicho ({predicted_class_index}) fuera de rango para las etiquetas ({len(class_labels)}).")


            print(f"Predicción: {predicted_label}, Confianza: {confidence:.4f}")

            confidence_percentage = f"{confidence*100:.2f}%"

            # Mostrar el resultado
            return render_template('result.html',
                                   species=predicted_label,
                                   confidence=confidence_percentage,
                                   image_filename=filename) # Pasamos el nombre para mostrar la imagen

        except Exception as e:
            print(f"Error durante la predicción o guardado: {e}")
            flash(f"Ocurrió un error inesperado: {e}", "error")
            # Considera borrar el archivo si falla la predicción
            # if os.path.exists(filepath):
            #     os.remove(filepath)
            return redirect(url_for('index'))
        # finally:
            # Opcional: Borrar la imagen subida después de usarla
            # if os.path.exists(filepath):
            #    try:
            #        os.remove(filepath)
            #        print(f"Archivo temporal eliminado: {filepath}")
            #    except Exception as e:
            #        print(f"Error al eliminar archivo temporal {filepath}: {e}")

    else:
        flash('Tipo de archivo no permitido. Usa png, jpg, jpeg, gif, webp.', 'warning')
        return redirect(url_for('index'))

# --- Ruta para servir imágenes subidas (necesario para result.html) ---
# ¡OJO! En producción real, es mejor usar un servidor web como Nginx o Apache
# para servir archivos estáticos de forma más eficiente y segura.
# Esto es SÓLO para desarrollo local sencillo.
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Sirve un archivo desde la carpeta de subidas."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- Iniciar la Aplicación ---
if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    # debug=True es útil para desarrollo, ¡DESACTÍVALO en producción!
    # host='0.0.0.0' permite acceder desde otras máquinas en la red local
    app.run(host='0.0.0.0', port=5000, debug=True)