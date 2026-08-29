# DINOv3 Setup

This project requires the official DINOv3 ViT-S/16 model.

## 1. Request access

1. Open [DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
2. Sign in, request access, and accept the license.
3. After approval, create a **Read** token at [Hugging Face tokens](https://huggingface.co/settings/tokens).

## 2. Install and log in

```bash
cd /home/user/dagkusue1/vision-systems/FINAL
source .venv/bin/activate
python -m pip install --upgrade "transformers>=4.56" huggingface_hub
hf auth login
```

Never share or commit the token. Revoke it immediately if exposed.

## 3. Download

```bash
mkdir -p weights/dinov3-vits16-hf
hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir weights/dinov3-vits16-hf
```

## 4. Verify

```bash
test -f weights/dinov3-vits16-hf/model.safetensors && echo "DINOv3 ready"
```

Keep weights under `FINAL/weights/`. They are ignored by Git. Each collaborator should request access and download the model independently.
