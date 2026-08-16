# Whole-Image OCR Loss

The LeX-10K OCR run adds a frozen official [UniRec-0.1B](https://huggingface.co/topdu/unirec-0.1b) recognizer to the existing PixelDiT flow-matching objective.

For each selected low-noise training sample, PixelDiT predicts velocity `v_hat` and reconstructs the complete denoised image:

```text
x0_hat = x_t - sigma_t * v_hat
```

The full `x0_hat` is resized differentiably to UniRec's input envelope. UniRec receives teacher-forced transcript tokens and produces logits. The OCR term is token cross-entropy with padding ignored:

```text
L = L_flow + 0.05 * CE(UniRec(x0_hat, target_prefix), target_next_token)
```

UniRec parameters are frozen, but the cross-entropy gradient flows through UniRec to `x0_hat`, then to PixelDiT. It does not use `generate`, argmax, edit distance, a text detector, NMS, or OCR boxes.

GT polygons are used only once in the dataset loader to order multiple GT strings from top to bottom, then left to right. They are not provided to the OCR model and are never used to crop or position the prediction.

The initial configuration limits OCR compute and prevents high-noise supervision: it is evaluated on 25% of samples with `sigma <= 0.25`, capped at one OCR sample per local batch. Logs report `flow_loss`, `ocr_loss`, and global `ocr_samples`.

## Required local model

Use the official Hugging Face endpoint, not a mirror:

```bash
pip install -r t2i/requirements-ocr.txt
HF_ENDPOINT=https://huggingface.co huggingface-cli download topdu/unirec-0.1b \
  --local-dir /home/wangye/huggingface_ckpts/unirec-0.1b
```

The job config expects that directory and does not permit a Hugging Face download while training.
