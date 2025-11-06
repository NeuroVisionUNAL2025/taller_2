# Taller de Visión por Computador — Informe

Este trabajo es una práctica experimental en visión por computador y procesamiento digital de imágenes. El objetivo general es aplicar, con datos reales, las operaciones fundamentales que permiten a un sistema de visión interpretar una escena: capturar una imagen, corregirla, transformarla y extraer información cuantitativa.

Todas las imágenes usadas fueron tomadas directamente por el equipo con una cámara de teléfono celular, en distintas condiciones de iluminación y contexto. El procesamiento se desarrolló en Python utilizando librerías de código abierto (OpenCV, NumPy y Matplotlib), y se documentó paso a paso.

## Estructura del informe
- [Introducción]
- [Marco Teórico]
- [Metodología]
- [Experimentos y Resultados]
- [Análisis y Discusión]
- [Conclusiones]
- [Referencias Bibliográficas]
- [Análisis de contribución individual]

## Introducción

## Marco Teórico

## Metodología

En esta sección se describe de manera detallada el pipeline de procesamiento, su justificación técnica y los diagramas de flujo correspondientes por módulo.

### 1.1 Módulo de carga y preprocesado

#### Descripción del pipeline
Se cargan las tres imágenes (img1, img2, img3) tomadas desde diferentes posiciones del comedor y se convierten a escala de grises para facilitar el análisis de características.

#### Justificación
La conversión a escala de grises reduce el procesamiento computacional y mejora la estabilidad de los detectores de características, ya que se basan en gradientes de intensidad.

#### Diagrama del proceso
[Inicio] → [Leer imágenes] → [Convertir a gris] → [Salida: imágenes en escala de grises]

### 1.2 Módulo de detección de Keypoints y cálculo de descriptores

#### Descripción del pipeline
Se selecciona el tipo de detector (SIFT, ORB, AKAZE) y se aplican las funciones detectAndCompute para obtener los puntos clave y sus descriptores correspondientes.

#### Justificación
Se requiere probar distintos detectores dadas sus diversas propiedades: SIFT es robusto a escala y rotación, ORB es rápido y eficiente en tiempo real, y AKAZE ofrece un equilibrio entre velocidad y precisión.

#### Diagrama del proceso
[Entrada: imagen gris] → [Seleccionar detector] → [Detectar keypoints] → [Calcular descriptores]

### 1.3 Módulo de matching de descriptores

#### Descripción del pipeline
Se utilizan los descriptores obtenidos para emparejar características entre imágenes con un BFMatcher (Brute Force Matcher) y se filtran mediante el test de razón de Lowe (ratio test). A continuación se calculan las coordenadas de los puntos emparejados y se aplica RANSAC para estimar una homografía robusta que relacione las imágenes y elimine outliers.

#### Justificación
El test de Lowe reduce los falsos positivos garantizando que los matches sean consistentes y fiables antes de estimar la transformación geométrica. Por su parte, RANSAC es un método iterativo que encuentra el mejor modelo ajustado a los datos válidos, evitando que los emparejamientos erróneos afecten el registro.

#### Diagrama del proceso
[Entrada: descriptores] → [BFMatcher knnMatch] → [Filtrar con ratio test] → [Buenos matches] → [RANSAC iterativo] → [Homografía óptima H] → [Máscara de inliers]

### 1.4 Módulo de construcción del lienzo panorámico y fusión

#### Descripción del pipeline
Se calculan las dimensiones del lienzo de salida a partir de las esquinas transformadas de cada imagen, y se aplican las homografías para proyectarlas en un mismo plano. Finalmente, se combinan para formar una única vista coherente del comedor.

#### Justificación
La homografía permite transformar las imágenes en un sistema de referencia común, creando un mosaico que conserva las proporciones y relaciones espaciales reales.

#### Diagrama del proceso
[Entrada: imágenes y homografías] → [Transformar esquinas] → [WarpPerspective] → [Combinar imágenes]

### 1.5 Módulo de interfaz de calibración y medición

#### Descripción del pipeline
Se muestra el mosaico final en una ventana ajustada a 1080 px de ancho. El usuario selecciona dos puntos para calibrar la escala en cm, y luego puede medir distancias con nuevos pares de clics.

#### Justificación
Permitir la interacción directa sobre la imagen facilita la calibración basada en objetos de referencia y la medición visual en unidades físicas reales.

#### Diagrama del proceso
[Mostrar mosaico] → [Clics de calibración] → [Calcular escala] → [Clics de medición] → [Mostrar distancia]


## Experimentos y Resultados

## Análisis y Discusión

## Conclusiones

## Referencias Bibliográficas

## Análisis de contribución individual