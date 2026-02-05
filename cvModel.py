import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import os

# ---------- CONFIG ----------
MODEL_PATH = "nrityavaani_mobilenet.pth"
DATA_DIR = "final_dataset/train"   # only for class names
CAMERA_INDEX = 1                   # <-- IMPORTANT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

# Load class names
classes = sorted(os.listdir(DATA_DIR))

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Open webcam (FIXED)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("🎥 Live Mudra Detection running — press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = transform(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        idx = prob.argmax().item()
        conf = prob[0][idx].item()

    label = f"{classes[idx]}  ({conf*100:.1f}%)"

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("NrityaVaani – Live Mudra Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
