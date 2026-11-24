# Resultados de validación

Fecha: 2026-11-24

## Dataset MNIST

Se evaluó el checkpoint actualizado sobre 10.000 imágenes de prueba.

- Aciertos: 9.764 / 10.000.
- Exactitud: 97,64 %.
- Por clase: 0: 99,5 %, 1: 99,5 %, 2: 98,8 %, 3: 97,7 %, 4: 97,6 %, 5: 98,4 %, 6: 98,6 %, 7: 97,8 %, 8: 92,4 %, 9: 95,0 %.

Se midió además con inclinación, perspectiva, cambios de grosor, desenfoque, contraste y trazos parcialmente borrados:

- Aciertos: 9.192 / 10.000.
- Exactitud con transformaciones de documento: 91,92 %.
- Es una evaluación sintética de resistencia a transformaciones, no una medida de exactitud sobre documentos reales.

## Ruta sintética de cámara

Las siete imágenes se construyen con ejemplos reconocidos de MNIST en escenas 640 × 480. Se extrae la misma región que usa la vista de cámara, y se ejecutan la segmentación, el modelo y el estimador de color. Se incluyen pruebas específicas para dígitos pequeños y con poca separación.

| Escena | Resultado esperado | Resultado |
|---|---|---|
| `camara_0123_negro_en_papel.png` | 0123, negro | 4 segmentos; correcto |
| `camara_456_colores_en_papel.png` | 456, rojo/verde/azul | 3 segmentos; correcto |
| `camara_789_colores_en_fondo_oscuro.png` | 789, amarillo/morado/azul | 3 segmentos; correcto |
| `camara_123_mas_colores_en_papel.png` | 123, naranja/cian/rosa | 3 segmentos; correcto |
| `camara_09_blanco_y_gris_en_fondo_oscuro.png` | 09, blanco/gris | 2 segmentos; correcto |
| `camara_5142_digitos_pequenos.png` | 5142 en trazos pequeños | 4 segmentos; correcto |
| `camara_5142_digitos_juntos.png` | 5142 con poca separación | 4 segmentos; correcto |

La misma segmentación de documento también pasó las escenas de tinta negra y de varios colores sobre papel: 0123 negro, 456 rojo/verde/azul y 123 naranja/cian/rosa. En esas escenas, tanto la cantidad de trazos como los dígitos y los colores coincidieron.

## Lanzador C++

Se ejecutó el `.exe` con la imagen de ejemplo `12345`, se confirmó el registro y se verificó la creación del CSV temporal. El lanzador devolvió código 0; el CSV de prueba se eliminó al terminar.

La prueba de cámara usa escenas sintéticas y la misma función que procesa el recorte capturado. No abrió la cámara física del equipo.
