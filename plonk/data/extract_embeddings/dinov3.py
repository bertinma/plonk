import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMAGE_PATH = "/home/mbertin@trusted.lan/Documents/data/datasets/geoloc/osv5m/images/test/00/1000055667540054.jpg"

print(f"Loading {MODEL_NAME} ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval().to(device)
print(f"Model on {device}")


img = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=[img], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
print(embeddings.shape)


cls_raw = outputs.last_hidden_state[:, 0]
cls_pooled = outputs.pooler_output

print(cls_raw[:5], cls_pooled[:5])
print((cls_raw - cls_pooled).abs().max().item())
