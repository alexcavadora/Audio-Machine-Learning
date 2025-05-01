# *CRISP-DM*. Clasificación de Géneros Musicales y Análisis de Similitud
## 1. Comprensión del Negocio
### 1.1 Objetivos de Negocio
- Objetivo primario: Desarrollar un modelo que pueda clasificar con precisión canciones en sus respectivos géneros basándose en características de audio
- Objetivo secundario: Crear un sistema de recomendación que pueda identificar canciones similares basándose en características de audio
- Objetivo terciario: Obtener información sobre qué características de audio diferencian más fuertemente los géneros musicales

### 1.2 Criterios de Éxito del Negocio
- El modelo de clasificación de géneros alcanza al menos un 75% de precisión en datos de prueba
- El sistema de recomendación identifica correctamente canciones de estilo/sensación similar independientemente del género
- El proyecto identifica al menos 3 hallazgos clave sobre características de audio que distinguen géneros

### 1.3 Evaluación de la Situación
- Recursos disponibles: Conjuntos de datos musicales (por ejemplo, GTZAN, Million Song Dataset, datos API de Spotify)
- Suposiciones: Las características de audio pueden ser extraídas y correlacionadas significativamente con etiquetas de género
- Restricciones: Recursos computacionales, posibles limitaciones de derechos de autor en datos musicales
- Riesgos: La extracción de características de audio puede no capturar todas las características musicales relevantes

### 1.4 Terminología
- **Características de Audio**: Propiedades medibles extraídas de señales de audio
- **MFCC (Coeficientes Cepstrales de Frecuencia Mel)**: Representación del espectro de potencia a corto plazo del sonido, reflejando cómo los humanos perciben el tono
- **Centroide Espectral**: Medida del "centro de masa" del espectro, indicando el brillo de un sonido
- **Ancho de Banda Espectral**: Medida del rango de frecuencias presentes en una señal
- **Característica Croma**: Representación del contenido tonal de la música, frecuentemente usado para detectar características armónicas y melódicas
- **Tasa de Cruce por Cero**: Tasa a la que una señal cambia de positivo a negativo o viceversa, correlacionándose con la ruidosidad
- **Energía RMS**: Energía de la raíz cuadrada media, representando la sonoridad de una señal de audio
- **Tempo**: Velocidad o ritmo de una pieza musical medido en beats por minuto (BPM)
- **Detección de Inicio**: Identificación del comienzo de notas musicales o eventos en una señal de audio
- **Agregación de Características**: Proceso de combinar características a nivel de marco en representaciones a nivel de canción
- **Transformada de Fourier**: Transformada matemática que convierte señales del dominio del tiempo a representaciones del dominio de frecuencia
- **Escala Mel**: Escala perceptual de tonos juzgados por oyentes para estar a distancias iguales entre sí
- **Agrupamiento K-means**: Algoritmo de aprendizaje no supervisado que agrupa puntos de datos similares
- **Matriz de Confusión**: Tabla usada para evaluar el rendimiento del modelo de clasificación
- **Puntuación F1**: Media armónica de precisión y exhaustividad, proporcionando un equilibrio entre estas métricas
- **t-SNE**: Técnica de reducción de dimensionalidad para visualizar datos de alta dimensión

### 1.5 Plan del Proyecto
- Cronograma: TBD
- Recursos necesarios: Entorno Python con bibliotecas para procesamiento de audio y ML
- Evaluación inicial de herramientas y técnicas: Librosa para extracción de características de audio, scikit-learn para modelado ML

## 2. Comprensión de los Datos

### 2.1 Recolección de Datos
- Fuentes de datos primarias:
  *  GTZAN, Free Music Archive, API de Spotify
  * Archivos de audio en formato WAV o MP3 con etiquetas de género
  * Metadatos incluyendo artista, título de canción, año de lanzamiento, etc.
- Métodos de adquisición de datos:
  * Descarga directa de conjuntos de datos establecidos
  * Acceso a API para servicios de streaming de música
  * Web scraping donde sea apropiado y legal

### 2.2 Descripción de los Datos
- Formatos de datos: Archivos de audio, vectores de características, metadatos
- Número de instancias: TBD
- Características a extraer:
  * Características temporales: Tasa de cruce por cero, energía RMS
  * Características espectrales: Centroide espectral, ancho de banda, contraste
  * Características cepstrales: MFCCs (Coeficientes cepstrales de frecuencia Mel)
  * Características rítmicas: Tempo, fuerza del beat
  * Características tonales: Croma, clave, modo

### 2.3 Exploración de Datos
- Análisis estadístico de características a través de géneros
- Visualización de distribuciones de características
- Análisis de correlación entre características
- Agrupamiento inicial para identificar agrupaciones naturales

### 2.4 Verificación de Calidad de Datos
- Comprobar datos faltantes en archivos de audio o metadatos
- Identificar posibles valores atípicos en distribuciones de características
- Evaluar desequilibrio de clases entre géneros
- Verificar calidad y consistencia de audio (tasas de muestreo, duraciones)

## 3. Preparación de Datos
### 3.1 Selección de Datos
- Criterios para la inclusión/exclusión de canciones
- Estrategia de muestreo para divisiones de entrenamiento/validación/prueba
- Selección de las características de audio más relevantes basadas en la exploración

### 3.2 Limpieza de Datos
- Manejo de valores faltantes en metadatos
- Normalización de archivos de audio (volumen, duración, tasa de muestreo)
- Eliminación de silencio, intros, outros si es necesario
- Detección y tratamiento de valores atípicos

### 3.3 Construcción de Características
- Metodología de extracción de características de audio:
  * Parámetros de ventana (tamaño de marco, longitud de salto)
  * Métodos de agregación (media, desviación estándar, etc.)
- Características derivadas:
  * Combinaciones de características
  * Características de evolución temporal
  * Reducción de dimensionalidad (PCA, t-SNE)

### 3.4 Integración de Datos
- Combinación de características de audio con metadatos
- Creación de estructura de conjunto de datos unificada
- Estandarización/normalización de características

### 3.5 Formato de Datos
- Estructura final del conjunto de datos
- Formato del vector de características
- Codificación de variables objetivo

## 4. Modelado

### 4.1 Técnicas de Modelado
- Para clasificación de género:
  * Random Forest
  * Máquinas de Vectores de Soporte
  * Redes Neuronales Convolucionales
  * K-Vecinos Más Cercanos
- Para similitud de canciones:
  * Similitud de coseno
  * Distancia euclidiana
  * Distancia de Mahalanobis
  * Enfoques de agrupamiento (K-means, DBSCAN)

### 4.2 Diseño de Prueba
- Estrategia de validación cruzada
- Métricas de rendimiento:
  * Precisión, exactitud, exhaustividad, puntuación F1 para clasificación
  * Evaluación subjetiva para recomendaciones de similitud
- Modelos de referencia para comparación

### 4.3 Construcción del Modelo
- Configuraciones de parámetros para cada algoritmo
- Procedimientos de entrenamiento
- Análisis de importancia de características
- Consideraciones de interpretabilidad del modelo

### 4.4 Evaluación del Modelo
- Resultados de evaluación
- Comparación de modelos
- Fortalezas y limitaciones de cada enfoque
- Selección de modelo(s) final(es)

## 5. Evaluación

### 5.1 Evaluación de Resultados
- Evaluación contra criterios de éxito del negocio
- Hallazgos clave sobre características que distinguen géneros
- Patrones o insights inesperados
- Análisis de matriz de confusión para clasificaciones erróneas

### 5.2 Proceso de Revisión
- Verificación de que todos los objetivos son abordados
- Identificación de factores pasados por alto
- Evaluación de fortalezas/debilidades metodológicas

### 5.3 Determinación de Próximos Pasos
- Recomendaciones para mejora del modelo
- Características adicionales para explorar
- Potenciales aplicaciones de negocio

## 6. Despliegue
### 6.1 Plan de Despliegue
- Estrategia de implementación para demostración en clase
- Potenciales aplicaciones del mundo real:
  * Sistemas de recomendación de streaming musical
  * Generación automatizada de listas de reproducción
  * Herramientas de producción musical

### 6.2 Plan de Monitoreo y Mantenimiento
- Métodos para actualizar el modelo con nueva música
- Métricas de monitoreo de rendimiento
- Calendario de reentrenamiento del modelo

### 6.3 Informe Final del Proyecto
- Resumen de hallazgos
- Documentación técnica
- Guías de usuario para aplicación del modelo
- Visualización de resultados

### 6.4 Revisión del Proyecto
- Lecciones aprendidas
- Evaluación del éxito del proyecto
- Recomendaciones para trabajo futuro

## Apéndice

### A. Estructura del Repositorio de Código
```
proyecto_genero_musical/
├── datos/
│   ├── brutos/               # Archivos de audio originales
│   ├── procesados/           # Características extraídas
│   └── metadatos/            # Información de canciones
├── notebooks/
│   ├── 1_exploracion_datos.ipynb
│   ├── 2_extraccion_caracteristicas.ipynb
│   ├── 3_modelado.ipynb
│   └── 4_evaluacion.ipynb
├── src/
│   ├── datos/                # Scripts de procesamiento de datos
│   ├── caracteristicas/      # Código de extracción de características
│   ├── modelos/              # Definiciones de modelos
│   └── visualizacion/        # Funciones de gráficos
├── modelos/                  # Archivos de modelos guardados
├── resultados/               # Figuras y resultados
└── README.md
```

### B. Bibliotecas Requeridas
- Procesamiento de audio: librosa, pydub
- Manipulación de datos: pandas, numpy
- Aprendizaje automático: scikit-learn, tensorflow/keras
- Visualización: matplotlib, seaborn, plotly

### C. Referencias
- TBD