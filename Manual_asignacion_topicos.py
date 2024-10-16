import pandas as pd

# Definir la ruta del archivo de entrada y salida
input_file_path = r'C:\Universidad\2024-1\Seminario 1\VADER\textos_procesados_con_temas_manual.csv'
output_file_path = r'C:\Universidad\2024-1\Seminario 1\VADER\textos_procesados_con_sentimiento_manual.csv'

# Cargar el dataset con modelado de temas
data = pd.read_csv(input_file_path)

# Asegurarse de que no haya valores nulos en 'texto_limpio'
data['texto_limpio'] = data['texto_limpio'].fillna('')

# Mapeo de tópicos a categorías
topic_mapping = {
    0: 'Bateria',
    1: 'Camara',
    2: 'Rendimiento'
}

# Asignar la categoría de tópico basada en dominant_topic
data['category'] = data['dominant_topic'].map(topic_mapping)

# Seleccionar y reorganizar las columnas necesarias
data = data[['Producto', 'Marca', 'Modelo', 'calificacion', 'Fecha', 'Texto', 'texto_limpio', 'sentimiento', 'tokens', 'dominant_topic', 'category']]

print(data)

# Guardar el dataset con las nuevas columnas
data.to_csv(output_file_path, index=False)

print("Dataset guardado con éxito en la ruta especificada.")