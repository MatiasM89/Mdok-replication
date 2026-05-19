# mdok Replication — Mmatias89/Mdok-Qwen3-14B

Replication of the mdok detector using Qwen3-14B for the "Voight-Kampff" Generative AI Authorship Verification at PAN 2025.

**Production image (for Tira submission):**
```bash
docker build -t mdok-styloch .
```

```bash
tira-cli code-submission \
	--path . \
	--task generative-ai-authorship-verification-panclef-2026 \
	--mount-hf-model Mmatias89/Mdok-Qwen3-14B \
	--dataset pan26-generative-ai-detection-smoke-test-20260330-training \
	--command '/usr/local/bin/mdok -i $inputDataset/*.jsonl -o $outputDir' \
	--dry-run
```
