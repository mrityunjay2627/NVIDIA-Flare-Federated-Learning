import os
import numpy as np
from PIL import Image, ImageDraw
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)

def make_good_image(size=224):
    """Clean textured surface - no defects"""
    img = Image.new("RGB", (size, size))
    pixels = np.random.randint(180, 220, (size, size, 3), dtype=np.uint8)
    # Add subtle uniform texture
    for i in range(0, size, 8):
        pixels[i:i+4, :] = pixels[i:i+4, :] - 10
    return Image.fromarray(pixels)

def make_defect_image(size=224):
    """Surface with visible scratch/crack/blob defect"""
    img = make_good_image(size)
    draw = ImageDraw.Draw(img)
    defect_type = random.choice(["scratch", "crack", "blob"])

    if defect_type == "scratch":
        x1, y1 = random.randint(20, 100), random.randint(20, 100)
        x2, y2 = random.randint(120, 200), random.randint(120, 200)
        draw.line([(x1, y1), (x2, y2)], fill=(80, 60, 60), width=3)

    elif defect_type == "crack":
        x, y = random.randint(50, 150), random.randint(50, 150)
        for _ in range(5):
            x2 = x + random.randint(-20, 20)
            y2 = y + random.randint(-20, 20)
            draw.line([(x, y), (x2, y2)], fill=(60, 50, 50), width=2)
            x, y = x2, y2

    elif defect_type == "blob":
        x, y = random.randint(50, 170), random.randint(50, 170)
        r = random.randint(10, 25)
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(100, 80, 70))

    return img

# Generate per factory
for factory_id in range(1, 4):
    for label, fn, count in [("good", make_good_image, 80), ("defect", make_defect_image, 40)]:
        out_dir = Path(f"factory_{factory_id}/{label}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = fn()
            img.save(out_dir / f"{label}_{i:04d}.png")
    print(f"Factory {factory_id} done — 80 good, 40 defect")

print("Synthetic dataset ready!")
