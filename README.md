# mdok Replication — Mmatias89/Mdok-Qwen3-14B

Replication of the mdok detector using Qwen3-14B for the "Voight-Kampff" Generative AI Authorship Verification at PAN 2025.

## Step 1: Download model weights locally

Run from inside `modelRunAttempt/`.

**Production model (14B, ~28GB — for Tira submission):**
```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Mmatias89/Mdok-Qwen3-14B', local_dir='./model')"
```

**Test model (0.6B, fits in 4GB VRAM — for local pipeline testing):**
```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Mmatias89/test-model', local_dir='./model-test')"
```

## Step 2: Build the image

**Test image (use this to verify the pipeline locally):**
```bash
    docker build --build-arg MODEL_DIR=./model-test -t mdok-test .
```

**Production image (for Tira submission):**
```bash
docker build -t mdok-replication .
```

## Step 3: Local Tira test

```bash
tira-run \
  --input-dataset generative-ai-authorship-verification-panclef-2026/pan26-generative-ai-detection-smoke-test-20260330-training \
  --image mdok-test \
  --command 'mdok -i $inputDataset/dataset.jsonl -o $outputDir'
```

Once the pipeline works with the test image, submit `mdok-replication` to Tira.

## Step 4: Push production image to registry

```bash
docker tag mdok-replication YOUR_DOCKERHUB_USER/mdok-replication:latest
docker push YOUR_DOCKERHUB_USER/mdok-replication:latest
```