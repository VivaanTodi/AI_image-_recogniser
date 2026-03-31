import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import re

# 🔹 Load model
model_name = "microsoft/kosmos-2-patch14-224"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name)

device = "cpu"
model.to(device)
model.eval()

# 🔹 Open webcam
cap = cv2.VideoCapture(0)

print("📸 Press SPACE to capture image...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera - Press SPACE", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        cv2.imwrite("captured.jpg", frame)
        print("✅ Image captured!")
        break

    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# 🔹 Load image
image = Image.open("captured.jpg").convert("RGB")

# 🔹 Prompt
prompt = "Describe the image in as much detail as possible:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

# 🔥 MAX GENERATION (no real limit)
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,   # 🔥 VERY HIGH (acts unlimited)
        do_sample=True,       # more natural text
        temperature=0.7
    )

# 🔹 Decode
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 🔥 CLEAN OUTPUT
output = re.sub(r"<.*?>", "", output).strip()

if "Describe" in output:
    output = output.split("Describe")[-1].strip()

print("\n🧠 Description:")
print(output)
