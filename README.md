# mdok Replication — Mmatias89/Mdok-Qwen3-14B

Replication of the mdok detector using Qwen3-14B for the "Voight-Kampff" Generative AI Authorship Verification at PAN 2025.

**Production image (for Tira submission):**
```bash
docker build -t mdok-styloch .
```

```bash
tira-run \
  --input-dataset generative-ai-authorship-verification-panclef-2026/pan26-generative-ai-detection-smoke-test-20260330-training \
  --image mdok-styloch \
  --command 'mdok -i $inputDataset/dataset.jsonl -o $outputDir'
```