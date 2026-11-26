# Registro de gastos con lectura de cifras manuscritas

Este prototipo universitario ayuda a registrar gastos a partir de una imagen, un PDF o la cámara. La aplicación propone una lectura y deja que el usuario la revise antes de guardarla.

## Qué permite hacer

- Reconocer dígitos manuscritos del 0 al 9 en imágenes y documentos.
- Buscar datos por categoría, como total, importe, precio o fecha.
- Mostrar en la cámara el texto detectado y capturar una zona para analizar sus cifras.
- Revisar y corregir el resultado antes de guardarlo en `gastos.csv`.
- Consultar los gastos guardados desde el historial.

El color de la tinta se estima con reglas de OpenCV y HSV. El modelo de PyTorch se encarga de clasificar las formas de los dígitos; no fue entrenado para reconocer colores.

## Cómo está organizado

- `cpp/src/main.cpp`: ejecutable de entrada que inicia el programa Python.
- `python/expense_gui.py`: menú, selección de archivos y revisión de resultados.
- `python/expense_app.py`: procesamiento de imágenes, cámara, predicción y guardado.
- `python/document_search.py`: búsqueda de etiquetas y números con OCR.
- `python/pdf_support.py`: lectura de texto y renderizado de páginas PDF.
- `python/train.py`: arquitectura y entrenamiento del Vision Transformer.
- `models/digits_vit.pth`: pesos del modelo que usa la aplicación.
- `ejemplos/`: imágenes para probar el programa.

## Requisitos

Para compilar y ejecutar en Windows se necesita:

- Python 3.10 o posterior.
- CMake 3.20 o posterior.
- Visual Studio 2022 con la carga de trabajo **Desarrollo para el escritorio con C++**.
- Una cámara web si se quiere probar el modo de cámara.
- Tesseract OCR para buscar etiquetas en imágenes y PDF escaneados. Los PDF con texto seleccionable no necesitan OCR para extraer sus importes.

## Preparar y ejecutar

Abre PowerShell en la carpeta del proyecto y prepara el entorno de Python:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Compila el lanzador C++:

```powershell
cmake -S .\cpp -B .\build
cmake --build .\build --config Release
```

Abre el menú principal:

```powershell
.\build\Release\expense_reader.exe
```

Desde el menú puedes elegir una imagen, un PDF, la cámara o el historial. Al terminar una revisión, la aplicación vuelve al menú.

También puedes abrir una imagen directamente:

```powershell
.\build\Release\expense_reader.exe .\ejemplos\importe_demo.png
.\build\Release\expense_reader.exe .\ejemplos\importe_54_demo.png
```

Para abrir un PDF:

```powershell
.\build\Release\expense_reader.exe "C:\ruta\a\tu_documento.pdf"
```

Para usar la cámara directamente:

```powershell
.\build\Release\expense_reader.exe --camera
```

## Buscar datos en documentos

Al revisar una imagen o un PDF, selecciona el dato que quieres encontrar: total, importe, precio unitario, fecha o dígitos. El programa intenta localizar el campo elegido y muestra el recorte para que puedas comprobarlo. Si no encuentra el campo, puedes seleccionar la zona manualmente.

Para reconocer etiquetas impresas en imágenes y PDF escaneados, instala Tesseract OCR en Windows:

```powershell
winget install --id tesseract-ocr.tesseract --exact
```

En la cámara, el OCR analiza el recuadro mientras se muestra la imagen. Al capturar, el modelo procesa los dígitos manuscritos. Las horas como `5:30` se presentan como horas y no como importes. La cámara necesita buena iluminación y los dígitos deben verse completos.

## Entrenar el modelo

El archivo `models/digits_vit.pth` ya está incluido, así que no hace falta entrenar para ejecutar la aplicación.

Para entrenar un modelo nuevo desde MNIST, ejecuta:

```powershell
.\.venv\Scripts\python.exe .\python\train.py --epochs 8 --output .\models\digits_vit_nuevo.pth
```

El conjunto MNIST se descarga si todavía no está en `data/`. Para continuar desde los pesos incluidos y guardar el resultado por separado:

```powershell
.\.venv\Scripts\python.exe .\python\train.py --epochs 3 --init-from .\models\digits_vit.pth --output .\models\digits_vit_ajustado.pth
```

El modelo usa un Vision Transformer pequeño: divide cada dígito en parches de 7 × 7, procesa esos parches con dos bloques Encoder y clasifica el resultado en una de diez clases. El entrenamiento agrega inclinación, cambios de escala, desenfoque y variaciones de grosor y contraste para acercarse a imágenes tomadas con cámara.

## Pruebas

Para evaluar el modelo y probar la segmentación con las imágenes de ejemplo:

```powershell
.\.venv\Scripts\python.exe .\python\validate_pipeline.py
```

Después de compilar, puedes probar la lectura de PDF y el flujo de guardado:

```powershell
.\.venv\Scripts\python.exe .\python\validate_pdf_support.py
```

En la evaluación de referencia, el modelo obtuvo 97,64 % en las 10.000 imágenes de prueba de MNIST y 91,98 % con transformaciones sintéticas que imitan algunos defectos de documentos. Esta segunda cifra no representa la exactitud en boletas reales. Las escenas de prueba y sus resultados están en `ejemplos/pruebas_tiempo_real/`.

## Alcance

La lectura es una propuesta: una cifra manuscrita puede confundirse, aunque el programa indique confianza alta. Revisa el valor y corrígelo antes de guardarlo. La precisión con documentos reales depende de la letra, la calidad de la imagen, el tamaño de los números y la iluminación.

