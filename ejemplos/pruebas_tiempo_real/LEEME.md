# Escenas de prueba para cámara

Estas imágenes se generan a partir de ejemplos del conjunto MNIST y simulan el recorte que recibe la cámara a 640 × 480. Incluyen el borde horizontal y un pequeño ícono para comprobar que el segmentador los descarte. También hay escenas con cuatro dígitos pequeños y con poca separación.

Desde la carpeta del proyecto, ejecuta:

```powershell
.\.venv\Scripts\python.exe .\python\validate_pipeline.py
```

El script evalúa el checkpoint con las 10.000 imágenes de prueba de MNIST y después procesa las escenas con la misma función de recorte y segmentación que usa la cámara. Debe informar `OK` en cada escena.

También puedes abrir las imágenes en un teléfono o una segunda pantalla y apuntar la cámara hacia ellas. Coloca los dígitos dentro del recuadro verde y pulsa Espacio. Las escenas cubren dígitos negros sobre papel, trazos rojos, naranjas, amarillos, verdes, cian, azules, morados, rosas, blancos y grises, además de fondos claros y oscuros.
