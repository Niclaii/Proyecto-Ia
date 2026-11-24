"""Evalúa el checkpoint y genera escenas sintéticas para la ruta de cámara."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from expense_app import prepare_digits  # noqa: E402
from train import DigitVisionTransformer  # noqa: E402


FRAME_SIZE = (640, 480)
ROI = (int(FRAME_SIZE[0] * 0.30), int(FRAME_SIZE[1] * 0.34),
       int(FRAME_SIZE[0] * 0.48), int(FRAME_SIZE[1] * 0.36))


def load_and_evaluate(model_path: Path, max_test_samples: int | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitVisionTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_set = datasets.MNIST(ROOT / "data", train=False, download=False,
                              transform=transforms.ToTensor())
    if max_test_samples is not None:
        test_set = torch.utils.data.Subset(test_set, range(min(max_test_samples, len(test_set))))
    loader = DataLoader(test_set, batch_size=512, shuffle=False)

    correct_by_class = np.zeros(10, dtype=np.int64)
    seen_by_class = np.zeros(10, dtype=np.int64)
    examples: list[tuple[float, np.ndarray] | None] = [None] * 10
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = logits.softmax(dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            confidence = probabilities.max(dim=1).values
            for i, label_tensor in enumerate(labels):
                label = int(label_tensor)
                seen_by_class[label] += 1
                if int(predictions[i]) == label:
                    correct_by_class[label] += 1
                    score = float(confidence[i])
                    if examples[label] is None or score > examples[label][0]:
                        raw = test_set.dataset.data[offset + i].numpy() if isinstance(test_set, torch.utils.data.Subset) else test_set.data[offset + i].numpy()
                        examples[label] = (score, raw.copy())
            offset += len(labels)

    accuracy = correct_by_class.sum() / max(1, seen_by_class.sum())
    return model, device, accuracy, correct_by_class, seen_by_class, examples


def make_camera_frame(sequence: str, examples: list[tuple[float, np.ndarray] | None],
                      background: tuple[int, int, int], inks: list[tuple[int, int, int]],
                      output_path: Path, digit_height: int, gap: int = 12) -> np.ndarray:
    left, top, width, height = ROI
    frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 75, dtype=np.uint8)
    roi = np.full((height, width, 3), background, dtype=np.uint8)

    glyphs = []
    for char in sequence:
        example = examples[int(char)]
        if example is None:
            raise RuntimeError(f"No se encontró un ejemplo MNIST correcto para {char}.")
        source = example[1]
        ys, xs = np.where(source > 24)
        crop = source[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        new_width = max(1, round(crop.shape[1] * digit_height / crop.shape[0]))
        resized = cv2.resize(crop, (new_width, digit_height), interpolation=cv2.INTER_CUBIC)
        glyphs.append(np.clip(resized, 0, 255).astype(np.uint8))

    total_width = sum(glyph.shape[1] for glyph in glyphs) + gap * (len(glyphs) - 1)
    if total_width > width - 12:
        raise RuntimeError(f"La secuencia {sequence} no cabe en el recuadro de cámara.")
    x = (width - total_width) // 2
    y = (height - digit_height) // 2
    for index, (char, glyph) in enumerate(zip(sequence, glyphs)):
        alpha = glyph.astype(np.float32)[:, :, None] / 255.0
        ink = np.array(inks[index], dtype=np.float32).reshape(1, 1, 3)
        patch = roi[y : y + digit_height, x : x + glyph.shape[1]]
        blended = patch.astype(np.float32) * (1 - alpha) + ink * alpha
        roi[y : y + digit_height, x : x + glyph.shape[1]] = np.clip(blended, 0, 255).astype(np.uint8)
        x += glyph.shape[1] + gap

    # Imitan un borde de pantalla y un icono pequeño fuera de la escritura.
    cv2.line(roi, (0, 8), (width - 1, 8), (190, 190, 190), 2)
    cv2.circle(roi, (12, 28), 5, (210, 210, 210), -1)
    frame[top : top + height, left : left + width] = roi
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise OSError(f"No se pudo guardar {output_path}")
    return roi


def test_camera_scenes(model, device, examples, output_dir: Path) -> bool:
    scenes = [
        ("0123", (245, 245, 245), [(15, 15, 15)] * 4,
         ["negro"] * 4, "negro_en_papel", 76, 12),
        ("456", (245, 245, 245), [(0, 0, 240), (0, 230, 0), (230, 40, 10)],
         ["rojo", "verde", "azul"], "colores_en_papel", 100, 12),
        ("789", (12, 12, 12), [(0, 230, 240), (230, 0, 220), (240, 40, 10)],
         ["amarillo", "morado", "azul"], "colores_en_fondo_oscuro", 94, 12),
        ("123", (245, 245, 245), [(0, 125, 240), (230, 210, 0), (200, 70, 240)],
         ["naranja", "cian", "rosa"], "mas_colores_en_papel", 100, 12),
        ("09", (12, 12, 12), [(230, 230, 230), (105, 105, 105)],
         ["blanco", "gris"], "blanco_y_gris_en_fondo_oscuro", 100, 12),
        ("5142", (245, 245, 245), [(15, 15, 15)] * 4,
         ["negro"] * 4, "digitos_pequenos", 36, 12),
        ("5142", (245, 245, 245), [(15, 15, 15)] * 4,
         ["negro"] * 4, "digitos_juntos", 65, 4),
    ]
    passed = True
    for sequence, background, inks, expected_colors, suffix, digit_height, gap in scenes:
        path = output_dir / f"camara_{sequence}_{suffix}.png"
        roi = make_camera_frame(sequence, examples, background, inks, path, digit_height, gap)
        digit_images, colors = prepare_digits(roi, adaptive=True)
        batch = torch.from_numpy(np.stack(digit_images)[:, None, :, :]).to(device)
        with torch.inference_mode():
            probabilities = model(batch).softmax(dim=1).cpu().numpy()
        predictions = "".join(str(int(value)) for value in probabilities.argmax(axis=1))
        color_ok = colors == expected_colors
        scene_ok = predictions == sequence and len(colors) == len(sequence) and color_ok
        passed &= scene_ok
        print(f"ESCENA {sequence}: predicción={predictions}; colores={','.join(colors)}; "
              f"segmentos={len(digit_images)}; {'OK' if scene_ok else 'FALLO'}")
        print(f"  Imagen de prueba: {path.relative_to(ROOT)}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "digits_vit.pth")
    parser.add_argument("--max-test-samples", type=int,
                        help="Limita la evaluación MNIST para una comprobación rápida.")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "ejemplos" / "pruebas_tiempo_real")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"No existe el modelo: {args.model}")

    model, device, accuracy, correct, seen, examples = load_and_evaluate(
        args.model, args.max_test_samples
    )
    print(f"MNIST test: exactitud={accuracy:.2%} ({int(correct.sum())}/{int(seen.sum())}); dispositivo={device}")
    print("Exactitud por dígito: " + ", ".join(
        f"{digit}:{correct[digit] / max(1, seen[digit]):.1%}" for digit in range(10)
    ))
    missing = [digit for digit, example in enumerate(examples) if example is None]
    if missing:
        print(f"No hay ejemplos acertados para generar todas las escenas: {missing}")
        return 1

    scenes_ok = test_camera_scenes(model, device, examples, args.output_dir)
    return 0 if scenes_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
