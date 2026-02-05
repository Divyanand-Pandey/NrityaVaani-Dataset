import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------------- CONFIG ----------------
DATA_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0003
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

# -------- TRANSFORMS (224x224 GUARANTEED) --------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor()
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# ------------------------------------------------

# ---------------- DATASETS ----------------
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=val_test_transform)
test_ds  = datasets.ImageFolder(f"{DATA_DIR}/test",  transform=val_test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
# ------------------------------------------------

# ---------------- MODEL ----------------
model = models.mobilenet_v2(pretrained=True)

# Replace last layer
model.classifier[1] = nn.Linear(
    model.last_channel,
    len(train_ds.classes)
)

model.to(DEVICE)
# ---------------------------------------

# ------------- LOSS & OPTIMIZER -------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# --------------------------------------------

# ---------------- TRAIN ----------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {running_loss:.3f} | Val Acc: {val_acc:.2f}%")

print("✅ Training finished")
# ---------------------------------------

# ---------------- TEST ----------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"🎯 Test Accuracy: {100 * correct / total:.2f}%")
# ---------------------------------------


# ---------------- SAVE ----------------
torch.save(model.state_dict(), "nrityavaani_mobilenet.pth")
print("💾 Model saved: nrityavaani_mobilenet.pth")