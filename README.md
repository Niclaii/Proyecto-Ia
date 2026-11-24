# Registro de gastos escritos a mano

Prototipo universitario que reconoce importes manuscritos con el Transformer visual del PDF y los registra localmente tras confirmarlos.

## Tecnologías

- **C++:** ejecutable de entrada y gestión del proceso.
- **Python + PyTorch:** Transformer, entrenamiento, visión por computador e inferencia.
- **CUDA:** PyTorch usa la GPU automáticamente si se instala una versión CUDA compatible; el entorno actual tiene PyTorch CPU.

El Transformer conserva los parámetros didácticos del documento: parches de 7 x 7, `d_model=64`, cuatro cabezas, dos bloques Encoder, posiciones sinusoidales y LayerNorm Post-LN. El entrenamiento usa ligeras transformaciones de escritura y valida el modelo antes de guardarlo. El checkpoint actual obtuvo 98,20 % en MNIST de prueba tras incorporar cuatro recortes personales. El modelo clasifica dígitos del 0 al 9. La cámara detecta automáticamente si el fondo es claro u oscuro y estima el color de cada trazo (negro, blanco, gris o colores comunes). El sistema propone el importe, permite corregirlo y guarda fecha, importe y categoría en `gastos.csv`.

La cámara muestra video en vivo y captura al pulsar Espacio. Después de capturar, la app muestra por consola el dígito y el color estimado de cada trazo. El recuadro ahora se centra en la escritura y elimina bordes horizontales largos de la pantalla para que no se unan con los trazos. Las imágenes deben mostrar dígitos claros y separados; los colores se estiman visualmente y pueden variar con la iluminación.

## Preparar el entorno Python

En PowerShell, desde la carpeta del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\python\train.py --epochs 8
```

La primera ejecución descarga MNIST y guarda los pesos en `models/digits_vit.pth`. Sin una compilación CUDA de PyTorch, el entrenamiento usa CPU.

Si corriges una predicción y la cantidad de cifras coincide con los segmentos detectados, la app puede guardar recortes aislados en `data/custom_digits/` con tu permiso. Para continuar el entrenamiento con esas muestras:

```powershell
.\.venv\Scripts\python.exe .\python\train.py --epochs 5 --init-from .\models\digits_vit.pth --output .\models\digits_vit_personal.pth
```

Revisa la validación antes de reemplazar el modelo activo por el nuevo checkpoint.

## Compilar y ejecutar el programa C++

Requiere Visual Studio con **Desarrollo para el escritorio con C++** y CMake.

```powershell
.\.venv\Scripts\cmake.exe -S .\cpp -B .\build
.\.venv\Scripts\cmake.exe --build .\build --config Release
.\build\Release\expense_reader.exe .\ejemplos\importe_demo.png
```

Al ejecutar `expense_reader.exe` sin argumentos se abre el menú gráfico. También puedes abrirlo con `--gui`. Desde el menú eliges PDF, imagen, cámara o historial; al cerrar una revisión, vuelves al menú principal.

```powershell
.\build\Release\expense_reader.exe
```

Las imágenes de ejemplo contienen `12345` y `54`; puedes usarlas para ejecutar el primer flujo sin depender de una cámara. La segunda permite comprobar la separación de dos dígitos:

```powershell
.\build\Release\expense_reader.exe .\ejemplos\importe_54_demo.png
```

También puedes elegir dónde guardar el CSV:

```powershell
.\build\Release\expense_reader.exe .\ejemplos\importe.jpg .\datos\gastos.csv
```

Para evaluar el checkpoint sobre MNIST y generar/probar escenas sintéticas con la misma segmentación que usa la cámara:

```powershell
.\.venv\Scripts\python.exe .\python\validate_pipeline.py
```

Las imágenes generadas y las instrucciones para mostrarlas frente a la cámara están en `ejemplos/pruebas_tiempo_real/`.

## Revisar un documento PDF

Al abrir una imagen o un PDF, elige qué dato quieres buscar: total, monto/importe, precio unitario, fecha o dígitos sueltos. El programa busca primero la etiqueta elegida y solo lee el valor asociado o recorta su zona; no mezcla todos los números de la página. Si hay varios resultados, podrás seleccionar uno. Los importes manuscritos siguen siendo una propuesta del modelo: la ventana muestra el recorte para que compares y corrijas el valor antes de guardarlo. Si no encuentra la etiqueta, puedes marcar la zona a mano.

```powershell
.\build\Release\expense_reader.exe "C:\ruta\a\tu_documento.pdf"
```

Para que la búsqueda por etiqueta funcione en imágenes y PDF escaneados, instala Tesseract OCR una vez en Windows:

```powershell
winget install --id tesseract-ocr.tesseract --exact
```

El programa usa OCR para localizar etiquetas y valores impresos. Para cifras manuscritas usa el modelo de dígitos y muestra el recorte junto a su propuesta. Revisa el número y escribe manualmente la coma o el punto decimal antes de guardar; la confianza del modelo puede ser alta aunque una cifra manuscrita se confunda. La fecha detectada también se puede editar en la ventana de revisión.

Para probar extracción multipágina, renderizado de escaneos y el flujo C++ completo con un PDF de demostración temporal:

```powershell
.\.venv\Scripts\python.exe .\python\validate_pdf_support.py
```

Para abrir la vista en vivo de la cámara, encuadra todos los dígitos dentro del rectángulo, sepáralos un poco y deja margen respecto al borde. Pulsa Espacio para capturar o Esc para cancelar:

```powershell
.\build\Release\expense_reader.exe --camera
```

Para analizar imágenes, cámara o recortes de PDF escaneados necesitas un modelo entrenado en `models/digits_vit.pth`; los PDF con texto seleccionable no dependen del modelo. El ejecutable C++ inicia el backend Python con el entorno virtual del proyecto.

## Siguientes mejoras

1. Reunir ejemplos escritos por varias personas y medir exactitud y correcciones.
2. Mejorar la segmentación de cifras conectadas y el centrado de cada dígito.
3. Mejorar la captura en distintas condiciones de luz y cámara.
4. Comparar el Transformer con una CNN como línea base.

