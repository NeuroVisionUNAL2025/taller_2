# Taller de Visión por Computador — Informe

Este trabajo es una práctica experimental en visión por computador y procesamiento digital de imágenes. El objetivo general es aplicar, con datos reales, las operaciones fundamentales que permiten a un sistema de visión interpretar una escena: capturar una imagen, corregirla, transformarla y extraer información cuantitativa.

Todas las imágenes usadas fueron tomadas directamente por el equipo con una cámara de teléfono celular, en distintas condiciones de iluminación y contexto. El procesamiento se desarrolló en Python utilizando librerías de código abierto (OpenCV, NumPy y Matplotlib), y se documentó paso a paso.

## Estructura del informe
- [Introducción](#introducción)
- [Marco Teórico](#marco-teórico)
- [Metodología](#metodología)
- [Experimentos y Resultados](#experimentos-y-resultados)
- [Análisis y Discusión](#análisis-y-discusión)
- [Conclusiones](#conclusiones)
- [Referencias Bibliográficas](#referencias-bibliográficas)
- [Análisis de contribución individual](#análisis-de-contribución-individual)

## Introducción

## Marco Teórico

### 1.1 SIFT (Scale-Invariant Feature Transform)
Propuesto por Lowe (2004), puede detectar extremos coincidentes entre varias escenas con diferentes condiciones de ruido,  distorsión o iluminación, a través de Difference-of-Gaussian, la localización precisa de puntos clave, la asignación de orientación para la invarianza a la rotación, y la creación de un descriptor local altamente distintivo.

### 1.2 ORB (Oriented FAST and Rotated BRIEF)
ORB fue desarrollado como una alternativa más eficiente computacionalmente. Rublee, R. et all (2011) explican que ORB combina el detector FAST para una orientación rápida y precisa de los puntos con un descriptor BRIEF orientado para una búsqueda eficiente de vecinos cercanos, proporcionando un método rápido, eficiente para aplicaciones en dispositivos menos robustos o en aplicaciones en tiempo real.

### 1.3 AKAZE (Accelerated KAZE)

### 1.4 RANSAC (Random Sample Consensus)
Desarrollado por Fischler y Bolles (1981), RANSAC permite interpretar o suavizar datos que contienen un porcentaje significativo de errores, como los generados por los detectores de características, así, logra ajustar un modelo incluso con un gran número de datos iniciales con errores. El principio de RANSAC es utilizar el número mínimo de puntos necesarios para definir un modelo y, a partir de ese modelo, verificar si el resto de los datos son compatibles.

## Metodología

En esta sección se describe de manera detallada el pipeline de procesamiento, su justificación técnica y los diagramas de flujo correspondientes por módulo.

### 2.1 Módulo de carga y preprocesado

#### Descripción del pipeline
Se cargan las tres imágenes (img1, img2, img3) tomadas desde diferentes posiciones del comedor y se convierten a escala de grises para facilitar el análisis de características.

#### Justificación
La conversión a escala de grises reduce el procesamiento computacional y mejora la estabilidad de los detectores de características, ya que se basan en gradientes de intensidad.

#### Diagrama del proceso
[Inicio] → [Leer imágenes] → [Convertir a gris] → [Salida: imágenes en escala de grises]

### 2.2 Módulo de detección de Keypoints y cálculo de descriptores

#### Descripción del pipeline
Se selecciona el tipo de detector (SIFT, ORB, AKAZE) y se aplican las funciones detectAndCompute para obtener los puntos clave y sus descriptores correspondientes.

#### Justificación
Se requiere probar distintos detectores dadas sus diversas propiedades: SIFT es robusto a escala y rotación, ORB es rápido y eficiente en tiempo real, y AKAZE ofrece un equilibrio entre velocidad y precisión.

#### Diagrama del proceso
[Entrada: imagen gris] → [Seleccionar detector] → [Detectar keypoints] → [Calcular descriptores]

### 2.3 Módulo de matching de descriptores

#### Descripción del pipeline
Se utilizan los descriptores obtenidos para emparejar características entre imágenes con un BFMatcher (Brute Force Matcher) y se filtran mediante el test de razón de Lowe (ratio test). A continuación se calculan las coordenadas de los puntos emparejados y se aplica RANSAC para estimar una homografía robusta que relacione las imágenes y elimine outliers.

#### Justificación
El test de Lowe reduce los falsos positivos garantizando que los matches sean consistentes y fiables antes de estimar la transformación geométrica. Por su parte, RANSAC es un método iterativo que encuentra el mejor modelo ajustado a los datos válidos, evitando que los emparejamientos erróneos afecten el registro.

#### Diagrama del proceso
[Entrada: descriptores] → [BFMatcher knnMatch] → [Filtrar con ratio test] → [Buenos matches] → [RANSAC iterativo] → [Homografía óptima H] → [Máscara de inliers]

### 2.4 Módulo de construcción del lienzo panorámico y fusión

#### Descripción del pipeline
Se calculan las dimensiones del lienzo de salida a partir de las esquinas transformadas de cada imagen, y se aplican las homografías para proyectarlas en un mismo plano. Finalmente, se combinan para formar una única vista coherente del comedor.

#### Justificación
La homografía permite transformar las imágenes en un sistema de referencia común, creando un mosaico que conserva las proporciones y relaciones espaciales reales.

#### Diagrama del proceso
[Entrada: imágenes y homografías] → [Transformar esquinas] → [WarpPerspective] → [Combinar imágenes]

### 2.5 Módulo de interfaz de calibración y medición

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

* Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, 60(2), 91–110. Recuperado de: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
* Rublee, Ethan & Rabaud, Vincent & Konolige, Kurt & Bradski, Gary. (2011). ORB: an efficient alternative to SIFT or SURF. Proceedings of the IEEE International Conference on Computer Vision. 2564-2571. 10.1109/ICCV.2011.6126544. Recuperado de: https://www.researchgate.net/publication/221111151_ORB_an_efficient_alternative_to_SIFT_or_SURF
* Fischler, M. A., & Bolles, R. C. (1981). Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography. Communications of the ACM, 24(6), 381–395. Recuperado de:  https://dl.acm.org/doi/10.1145/358669.358692


## Análisis de contribución individual