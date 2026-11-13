# Deployment audit

Start with worship-v3, the best validation macro F1 so far. The app already has six
Core ML packages in SongMap/Resources/MLModels/SongFormer, matching the 30/120 s,
layer-10, four-stream profile. This audit does not change app resources.

Run from Section-Annotations with the existing Python 3.11 environment:

```bash
python -m deployment.audit --benchmark --device mps --output outputs.local/deployment-audit.json
```

Optional `--songmap-root /absolute/path/to/SongMapApp` inventories all six local
packages. LFS pointers are flagged; tiny pointer files are not model-size evidence.
No downloads or package changes are performed. Choose a new output name on rerun.
Only use a trusted locally generated checkpoint (.pt uses torch.load).

Timing is warmed-up PyTorch head-only on cached features, including input transfer.
It is not end-to-end, Core ML, iPhone latency, or peak memory. The ideal 6-bit size
is arithmetic for head parameters only, not an actual compressed model. Encoders
must be counted separately; caches are not shipped to iPhone.

## Integration gates

1. Confirm the target iPhone and iOS. Inventory actual local packages (LFS fetched).
2. Export the fine-tuned label pipeline, preserving all 128 output logits and the
   allowed IDs [0,1,2,3,4,5,6,26]. Fine-tuning changed function_head; do not replace
   the entire Core ML pipeline with a standalone linear classifier expecting audio.
3. Verify float export against PyTorch on identical cached feature fixtures over
   supported lengths, then verify encoder parity against Python audio preprocessing.
4. Keep Apple boundaries and pool per-frame softmax probabilities within each
   section. SongFormerAnalyzer currently produces argmax frame labels, so its
   output cannot reproduce probability pooling. Adapt and test before integration.
5. Establish float Core ML validation metrics before compression. Then test
   6-bit weight palettization where supported by the chosen coremltools/op/runtime
   combination. Six-bit weights do not mean six-bit activations or a guaranteed
   speedup, and are not equivalent to universally supported int6 arithmetic.
6. Compare compressed versus float on validation, keeping model selection away
   from repeated test use. Report per-class F1 as well as aggregate metrics.
7. On the physical iPhone measure complete audio -> Apple boundaries -> labels:
   cold/warm latency, peak memory, package size, compute units, OS and model name.

No conversion or 6-bit artifact is claimed by this audit. Export/compression and
on-device parity remain subsequent work after collecting the inventory.
