# ─────────────────────────────────────────────────────────────────────────────────────────────────────
#   IELU SEGMENTĀCIJAS MODEĻU APMĀCĪBA - Edvards Bārtulis, Aleksandrs Kozaļetovs, Anastasija Ostrovska
# ─────────────────────────────────────────────────────────────────────────────────────────────────────

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from skimage.draw import polygon
from torchmetrics import JaccardIndex
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp
import time
import random
import warnings
warnings.filterwarnings("ignore")

# ==================== BDD100K KLASES ====================
CLASSES = [
    "ceļš", "trotuārs", "ēka", "siena", "žogs", "stabs",
    "luksofors", "ceļa zīme", "veģetācija", "reljefs",
    "debesis", "cilvēks", "braucējs", "auto", "kravas auto",
    "autobuss", "vilciens", "motocikls", "velosipēds", "nezināms"
]
NUM_CLASSES = len(CLASSES)
ADAS_IDS = [0, 11, 13, 6]  # ceļš, cilvēks, auto, luksofors

# ==================== KATEGORIJU KARTE ====================
CATEGORY_MAP = {
    'pedestrian': 11, 'person': 11, 'rider': 12,
    'car': 13, 'truck': 14, 'bus': 15, 'train': 16,
    'motorcycle': 17, 'bicycle': 18,
    'traffic light': 6, 'traffic sign': 7,
    'drivable area': 0, 'lane': 0,
}

# ==================== KLAŠU SVARI  ====================
class_weights = torch.tensor([
    1.0, 2.0, 1.5, 3.0, 3.0, 4.0,
    10.0, 8.0, 1.0, 1.0, 0.5,        # 6 = luksofors
    10.0, 10.0, 2.0, 4.0, 5.0, 8.0, # 11 = cilvēks
    10.0, 8.0, 0.5
], dtype=torch.float32)
if torch.cuda.is_available():
    class_weights = class_weights.cuda()

# ==================== MASKU PRIEKŠAPSTRĀDE ====================
def preprocess_masks(img_dir, ann_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        base = os.path.splitext(img_name)[0]
        ann_path = os.path.join(ann_dir, base + '.json')
        mask_path = os.path.join(mask_dir, base + '.png')
        if os.path.exists(mask_path):
            continue

        img = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
        h, w = img.size[1], img.size[0]
        mask = np.full((h, w), 19, dtype=np.uint8)

        if os.path.exists(ann_path):
            try:
                with open(ann_path) as f:
                    data = json.load(f)
                for obj in data.get('objects', []):
                    cat = obj.get('classTitle')
                    if cat not in CATEGORY_MAP:
                        continue
                    cid = CATEGORY_MAP[cat]
                    points = obj.get('points', {}).get('exterior', [])
                    geom = obj.get('geometryType')

                    if geom == 'rectangle' and len(points) >= 2:
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        mask[int(min(y1,y2)):int(max(y1,y2)), int(min(x1,x2)):int(max(x1,x2))] = cid
                    elif geom == 'polygon' and len(points) > 2:
                        rr, cc = polygon([p[1] for p in points], [p[0] for p in points], (h, w))
                        mask[rr, cc] = cid
            except Exception as e:
                print(f"Kļūda {ann_path}: {e}")

        Image.fromarray(mask).save(mask_path)
    print(f"Maskas gatavas → {mask_dir}")

# ==================== KOPĪGĀS TRANSFORMĀCIJAS ====================
class RandomScaleCrop:
    def __init__(self, size=512):
        self.size = size
    def __call__(self, img, mask):
        w, h = img.size
        scale = np.random.uniform(0.8, 1.4)
        nw, nh = int(w * scale), int(h * scale)
        img = transforms.Resize((nh, nw))(img)
        mask = transforms.Resize((nh, nw), Image.NEAREST)(mask)
        i = np.random.randint(0, nh - self.size + 1)
        j = np.random.randint(0, nw - self.size + 1)
        img = img.crop((j, i, j + self.size, i + self.size))
        mask = mask.crop((j, i, j + self.size, i + self.size))
        return img, mask

class CustomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class ResizeBoth:
    def __init__(self, size=(512, 512)):
        self.size = size
    def __call__(self, img, mask):
        img = transforms.Resize(self.size)(img)
        mask = transforms.Resize(self.size, Image.NEAREST)(mask)
        return img, mask

# ==================== KOPĪGĀ TRANSFORMĀCIJA ====================
class SegmentationTransform:
    def __init__(self, joint_transforms, image_transforms):
        self.joint = joint_transforms
        self.image = image_transforms

    def __call__(self, img, mask):
        for t in self.joint:
            img, mask = t(img, mask)
        img = self.image(img)
        mask = torch.tensor(np.array(mask)).long()  # uz long tenzoru
        return img, mask

# ==================== TRANSFORMĀCIJU KONFIGURĀCIJA ====================
image_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

joint_train = [RandomScaleCrop(512), CustomHorizontalFlip(0.5)]
joint_val = [ResizeBoth((512, 512))]

train_transform = SegmentationTransform(joint_train, image_train)
val_transform = SegmentationTransform(joint_val, image_val)

# ==================== DATU KOPA ====================
class BDD100KDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask_path = os.path.join(self.mask_dir, os.path.splitext(name)[0] + '.png')
        mask = Image.open(mask_path) if os.path.exists(mask_path) else Image.new("L", img.size, 19)

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

# ==================== MODEĻU IELĀDE ====================
def load_deeplabv3():
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, 1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, 1)
    return model

def load_segformer():
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model.decode_head.classifier = nn.Conv2d(256, NUM_CLASSES, 1)
    model.config.num_labels = NUM_CLASSES
    return model, processor

def load_unetpp():
    return smp.UnetPlusPlus(encoder_name="efficientnet-b3", encoder_weights="imagenet", classes=NUM_CLASSES)

# ==================== APMĀCĪBA ====================
def train_model(model, train_loader, val_loader, name, epochs=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    best = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            if isinstance(model, SegformerForSemanticSegmentation):
                out = model(pixel_values=imgs).logits
                out = nn.functional.interpolate(out, size=(512,512), mode="bilinear", align_corners=False)
            else:
                out = model(imgs)['out'] if hasattr(model, 'aux_classifier') else model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
        epoch_time = time.time() - epoch_start

        # validācija + замер инференса
        model.eval()
        m = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(device)
        infer_times = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                
                start_inf = time.time()
                if isinstance(model, SegformerForSemanticSegmentation):
                    out = model(pixel_values=imgs).logits
                    out = nn.functional.interpolate(out, size=(512,512), mode="bilinear")
                else:
                    out = model(imgs)['out'] if hasattr(model, 'aux_classifier') else model(imgs)
                infer_times.append(time.time() - start_inf)
                
                m.update(out.argmax(1), masks)
        val_miou = m.compute().item()
        
        avg_infer = np.mean(infer_times) * 1000
        fps = 1000 / avg_infer if avg_infer > 0 else 0

        print(f"[{name}] Epoha {epoch+1}/{epochs} → Val mIoU: {val_miou:.4f} | "
              f"Epohas laiks: {epoch_time/60:.1f} min | "
              f"Inference: {avg_infer:.1f} ms/batch ≈ {fps:.1f} FPS")

        if val_miou > best:
            best = val_miou
            torch.save(model.state_dict(), f"{name}_best.pth")

    total_time = time.time() - start_time
    print(f"\n{name} apmācība pabeigta! Kopējais laiks: {total_time/60:.1f} min "
          f"({total_time:.0f} sek)\n")
    return model

# ==================== VIZUALIZĀCIJA ====================
def visualize(model, loader, name, device, num=10):
    model.eval()
    os.makedirs("visualizations", exist_ok=True)
    cnt = 0
    with torch.no_grad():
        for imgs, masks in loader:
            if cnt >= num: break
            imgs = imgs.to(device)
            if isinstance(model, SegformerForSemanticSegmentation):
                out = model(pixel_values=imgs).logits
                out = nn.functional.interpolate(out, size=(512,512), mode="bilinear", align_corners=False)
            else:
                out = model(imgs)['out'] if hasattr(model, 'aux_classifier') else model(imgs)
            pred = out.argmax(1).cpu().numpy()
            imgs = imgs.cpu()

            for i in range(imgs.shape[0]):
                if cnt >= num: break
                img = imgs[i].permute(1,2,0).numpy()
                img = np.clip(img * [0.229,0.224,0.225] + [0.485,0.456,0.406], 0, 1)

                plt.figure(figsize=(20,5))
                plt.subplot(1,4,1); plt.imshow(img); plt.title("Oriģinālais"); plt.axis('off')
                plt.subplot(1,4,2); plt.imshow(masks[i].numpy(), cmap='tab20', vmin=0, vmax=19); plt.title("GT"); plt.axis('off')
                plt.subplot(1,4,3); plt.imshow(pred[i], cmap='tab20', vmin=0, vmax=19); plt.title(name); plt.axis('off')
                plt.subplot(1,4,4); plt.imshow(img); plt.imshow(pred[i], cmap='tab20', alpha=0.5, vmin=0, vmax=19); plt.title("Pārklājums"); plt.axis('off')
                plt.suptitle(f"{name} #{cnt+1}", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"visualizations/{name}_{cnt+1:02d}.png", dpi=150)
                plt.close()
                cnt += 1

# ==================== NOVĒRTĒŠANA ====================
def evaluate(model, loader, name, device):
    model.eval()
    m = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average=None).to(device)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            if isinstance(model, SegformerForSemanticSegmentation):
                out = model(pixel_values=imgs).logits
                out = nn.functional.interpolate(out, size=(512,512), mode="bilinear", align_corners=False)
            else:
                out = model(imgs)['out'] if hasattr(model, 'aux_classifier') else model(imgs)
            m.update(out.argmax(1), masks)
    ious = m.compute().cpu().numpy()
    print(f"\n{name} → Kopējais mIoU: {ious.mean():.4f}")
    for i in ADAS_IDS:
        print(f"   {CLASSES[i]:<12}: {ious[i]:.4f}")
    return ious

# ==================== GALVENĀ ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ierīce: {device}")

    # Priekšapstrāde (vienreiz)
    preprocess_masks("train/img", "train/ann", "train/mask")
    preprocess_masks("val/img", "val/ann", "val/mask")

    train_ds = BDD100KDataset("train/img", "train/mask", transform=train_transform)
    val_ds   = BDD100KDataset("val/img",   "val/mask",   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=6, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=6, shuffle=False, num_workers=4, pin_memory=True)

    # 1. DeepLabV3+
    dl = load_deeplabv3()
    dl = train_model(dl, train_loader, val_loader, "DeepLabV3+", epochs=12)
    dl_ious = evaluate(dl, val_loader, "DeepLabV3+", device)
    visualize(dl, val_loader, "DeepLabV3+", device)

    # 2. SegFormer
    sf, _ = load_segformer()
    sf = train_model(sf, train_loader, val_loader, "SegFormer", epochs=12)
    sf_ious = evaluate(sf, val_loader, "SegFormer", device)
    visualize(sf, val_loader, "SegFormer", device)

    # 3. UNet++ (EfficientNet-B3)
    unet = load_unetpp()
    unet = train_model(unet, train_loader, val_loader, "UNet++-EffB3", epochs=12)
    unet_ious = evaluate(unet, val_loader, "UNet++-EffB3", device)
    visualize(unet, val_loader, "UNet++-EffB3", device)

    # REZULTĀTS
    print("\n" + "="*80)
    print("GALĪGAIS SALĪDZINĀJUMS (galvenās klases)")
    print("="*80)
    print(f"{'Klase':<12} {'DeepLab':<10} {'SegFormer':<12} {'UNet++'}")
    for i in ADAS_IDS:
        print(f"{CLASSES[i]:<12} {dl_ious[i]:.3f}      {sf_ious[i]:.3f}        {unet_ious[i]:.3f}")
    print(f"Kopējais mIoU   {dl_ious.mean():.3f}      {sf_ious.mean():.3f}        {unet_ious.mean():.3f}")
    print("\nGatavs! 30 vizualizācijas mapē visualizations/")