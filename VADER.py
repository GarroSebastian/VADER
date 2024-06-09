import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Definir la ruta del archivo de entrada y salida
input_file_path = r'C:\Universidad\2024-1\Seminario 1\VADER\textos_procesados_con_temas.csv'
output_file_path = r'C:\Universidad\2024-1\Seminario 1\VADER\textos_procesados_con_sentimiento.csv'

# Cargar el dataset con modelado de temas
data = pd.read_csv(input_file_path)

# Asegurarse de que no haya valores nulos en 'texto_limpio_reseña'
data['texto_limpio_reseña'] = data['texto_limpio_reseña'].fillna('')

# Inicializar el analizador de sentimientos VADER
analyzer = SentimentIntensityAnalyzer()

# Aplicar el análisis de sentimientos VADER
data['scores'] = data['texto_limpio_reseña'].apply(lambda x: analyzer.polarity_scores(x))
data['compound'] = data['scores'].apply(lambda x: x['compound'])
data['sentiment_VADER'] = data['compound'].apply(lambda x: 'positivo' if x >= 0.05 else ('negativo' if x <= -0.05 else 'neutral'))

# Separar los puntajes de VADER en columnas independientes (opcional)
data[['neg', 'neu', 'pos']] = data['scores'].apply(pd.Series).drop(columns=['compound'])

# Seleccionar y reorganizar las columnas necesarias
data = data[['producto','fecha','texto_limpio_reseña', 'calificacion', 'dominant_topic', 'category_aspect', 'scores', 'compound', 'sentiment_VADER']]

# Guardar el dataset con las nuevas columnas
data.to_csv(output_file_path, index=False)

print("Dataset guardado con éxito en la ruta especificada.")
