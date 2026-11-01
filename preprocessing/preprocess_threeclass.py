# ===== DenseNet-121 Three-class =====
from google.colab import drive
drive.mount('/content/drive')

import os, re, shutil, random, getpass
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CLASSES = ["Normal", "Bacterial", "Viral"]
SRC  = "/content/kermany_data/chest_xray"
DATA = "/content/threeclass"
OUT  = "/content/drive/MyDrive/pneumonia_project_threeclass"
os.makedirs(f"{OUT}/preprocessing", exist_ok=True)

# ---- 1. Dataset download  ----
if not os.path.exists(SRC):
    os.environ["KAGGLE_API_TOKEN"] = getpass.getpass("kaggle_key")
    !pip install kaggle -q
    !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
    !unzip -q chest-xray-pneumonia.zip -d /content/kermany_data
print("Dataset ready:", os.path.exists(SRC))

# ---- 2. 3-class-a reorganize (filename: "bacteria" / "virus") ----
if not os.path.exists(f"{DATA}/Viral"):
    for cls in CLASSES:
        os.makedirs(f"{DATA}/{cls}", exist_ok=True)
    for split in ["train", "val", "test"]:
        for f in os.listdir(f"{SRC}/{split}/NORMAL"):
            shutil.copy(f"{SRC}/{split}/NORMAL/{f}", f"{DATA}/Normal/{split}_{f}")
        for f in os.listdir(f"{SRC}/{split}/PNEUMONIA"):
            name = f.lower()
            if "bacteria" in name:
                shutil.copy(f"{SRC}/{split}/PNEUMONIA/{f}", f"{DATA}/Bacterial/{split}_{f}")
            elif "virus" in name:
                shutil.copy(f"{SRC}/{split}/PNEUMONIA/{f}", f"{DATA}/Viral/{split}_{f}")
for cls in CLASSES:
    print(cls, len(os.listdir(f"{DATA}/{cls}")))

# ---- 3. Patient-wise stratified split (~80 / 10 / 10) ----
paths, labels, groups = [], [], []
for i, cls in enumerate(CLASSES):
    for f in os.listdir(f"{DATA}/{cls}"):
        name = re.sub(r"^(train|val|test)_", "", f)
        if name.lower().startswith("person"):
            g = name.split("_")[0]
        else:
            m = re.search(r"IM-(\d+)", name)
            g = "IM" + m.group(1) if m else name
        paths.append(f"{DATA}/{cls}/{f}"); labels.append(i); groups.append(g)
paths, labels, groups = np.array(paths), np.array(labels), np.array(groups)

sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
trainval_idx, test_idx = next(sgkf.split(paths, labels, groups))
sgkf2 = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
tr, va = next(sgkf2.split(paths[trainval_idx], labels[trainval_idx], groups[trainval_idx]))
train_idx, val_idx = trainval_idx[tr], trainval_idx[va]

for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
    c = [int((labels[idx] == k).sum()) for k in range(3)]
    print(f"{name}: {len(idx)} | Normal {c[0]} | Bacterial {c[1]} | Viral {c[2]}")

split_name = np.empty(len(paths), dtype=object)
split_name[train_idx], split_name[val_idx], split_name[test_idx] = "train", "val", "test"
pd.DataFrame({"path": paths, "label": labels, "class": [CLASSES[l] for l in labels],
            "group": groups, "split": split_name}).to_csv(f"{OUT}/preprocessing/splits.csv", index=False)

# ---- 4. Resize 224x224 + ImageNet normalization + augmentation ----
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class XrayDataset(Dataset):
    def __init__(self, idx, tf):
        self.paths, self.labels, self.tf = paths[idx], labels[idx], tf
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.labels[i])

BATCH = 32
train_loader = DataLoader(XrayDataset(train_idx, train_tf), batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(XrayDataset(val_idx, eval_tf), batch_size=BATCH, num_workers=2, pin_memory=True)
test_loader  = DataLoader(XrayDataset(test_idx, eval_tf), batch_size=BATCH, num_workers=2, pin_memory=True)

# ---- 5. Class weights (imbalance handling) ----
cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=labels[train_idx])
cw = torch.tensor(cw, dtype=torch.float32)
print("Class weights (Normal, Bacterial, Viral):", cw)

x, y = next(iter(train_loader))
print("Batch check:", x.shape, y.shape)