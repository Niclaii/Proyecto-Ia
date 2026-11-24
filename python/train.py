"""Entrena el Transformer visual pequeño y guarda sus pesos para PyTorch."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms


class CorrectedDigitDataset(Dataset):
    """Optional 28x28 crops labeled by their 0-9 parent directory."""

    def __init__(self, root: Path, transform) -> None:
        self.transform = transform
        self.samples = [
            (path, int(path.parent.name))
            for path in root.glob("[0-9]/*.png")
            if path.parent.name.isdigit()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("L")), label


class SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("d_model debe ser divisible por el número de cabezas")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.heads, self.head_width
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        weights = scores.softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, tokens, width)
        return self.projection(context)


class EncoderBlock(nn.Module):
    """Bloque con LayerNorm después de cada conexión residual (Post-LN)."""

    def __init__(self, width: int = 64, heads: int = 4, hidden: int = 256) -> None:
        super().__init__()
        self.attention = SelfAttention(width, heads)
        self.norm1 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, width)
        )
        self.norm2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attention(x))
        return self.norm2(x + self.ffn(x))


def sinusoidal_positions(tokens: int, width: int) -> torch.Tensor:
    positions = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)
    scales = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros(tokens, width)
    encoding[:, 0::2] = torch.sin(positions * scales)
    encoding[:, 1::2] = torch.cos(positions * scales)
    return encoding.unsqueeze(0)


class DigitVisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = 7
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_projection = nn.Linear(self.patch_size * self.patch_size, 64)
        self.register_buffer("positions", sinusoidal_positions(16, 64))
        self.blocks = nn.Sequential(EncoderBlock(), EncoderBlock())
        self.classifier = nn.Linear(64, 10)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.unfold(image).transpose(1, 2)
        tokens = self.patch_projection(patches) + self.positions
        encoded = self.blocks(tokens)
        return self.classifier(encoded.mean(dim=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("models/digits_vit.pth"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--custom-dir", type=Path, default=Path("data/custom_digits"))
    parser.add_argument("--init-from", type=Path,
                        help="Opcional: continúa el entrenamiento desde un checkpoint existente")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs debe ser al menos 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en {device}...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    training_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=8, translate=(0.08, 0.08), scale=(0.90, 1.10), shear=5, fill=0
        ),
        transforms.ToTensor(),
    ])
    plain_transform = transforms.ToTensor()
    augmented_train = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=training_transform
    )
    plain_train = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=plain_transform
    )
    test_set = datasets.MNIST(
        args.data_dir, train=False, download=True, transform=plain_transform
    )
    split_generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(augmented_train), generator=split_generator).tolist()
    train_indices, validation_indices = indices[:-5000], indices[-5000:]
    base_subset = Subset(augmented_train, train_indices)
    personal_set = CorrectedDigitDataset(args.custom_dir, training_transform)
    if len(personal_set):
        combined_train = ConcatDataset([base_subset, personal_set])
        personal_share = 0.02
        personal_weight = len(base_subset) * personal_share / (
            (1.0 - personal_share) * len(personal_set)
        )
        sample_weights = [1.0] * len(base_subset) + [personal_weight] * len(personal_set)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(base_subset), replacement=True,
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(
            combined_train, batch_size=args.batch_size, sampler=sampler
        )
        print(f"Ejemplos personales adicionales: {len(personal_set)}")
    else:
        train_loader = DataLoader(
            base_subset, batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )
    validation_loader = DataLoader(
        Subset(plain_train, validation_indices), batch_size=args.batch_size
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = DigitVisionTransformer().to(device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=True))
        print(f"Continuando desde: {args.init_from}")
    learning_rate = 0.0002 if args.init_from else 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    best_validation = 0.0
    best_state = copy.deepcopy(model.state_dict())
    if args.init_from:
        model.eval()
        correct = seen = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                correct += (predictions == labels).sum().item()
                seen += labels.numel()
        best_validation = correct / seen
        print(f"Validación inicial: {best_validation:.2%}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = seen = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                correct += (predictions == labels).sum().item()
                seen += labels.numel()
        validation_accuracy = correct / seen
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"Época {epoch + 1}/{args.epochs} | "
            f"pérdida={total_loss / len(train_loader):.4f} | "
            f"exactitud validación={validation_accuracy:.2%}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    correct = seen = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            seen += labels.numel()
    print(f"Exactitud final en MNIST de prueba: {correct / seen:.2%}")
    torch.save(model.state_dict(), args.output)
    print(f"Pesos PyTorch guardados en: {args.output.resolve()}")


if __name__ == "__main__":
    main()
