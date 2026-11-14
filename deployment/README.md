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

## Fine-tuned FP16 head export and parity

Open `coreml_export_experiment.ipynb` in the repository root, Python 3.11 kernel.
Install `deployment/requirements-coreml.txt` in that environment. Coremltools
8.x is used with the existing torch 2.4 training stack; successful conversion
and macOS runtime prediction must be established locally before integration.

Exporter runs on CPU to avoid tracing MPS-specific operations. It fixes dataset
ID 5 and exposes only `input_embeddings` -> `function_logits`, retaining 128
outputs and the app's 8 allowed class IDs. There is no dataset/mask input: Swift
only supplies those when present. No change to encoders or the app is made.
Only trusted `best.pt` is accepted and split/cache metadata is checked.
Range T is 3..10500; callers must use multiples of 3. Trace is explicitly
checked across lengths before conversion, and FP16 synthetic parity is checked
after conversion. Attention flash is disabled only on the export copy, and
its numerical equivalence is checked. If conversion, dynamic-shape checks, or
parity fail, stop and inspect the full log; an output folder is not proof of
a usable model. Never bypass the checks to obtain a package.

`deployment.verify_coreml` compares the original PyTorch attention implementation
with Core ML CPU_ONLY on identical validation cache chunks, including their real
lengths up to 420s. It pools probabilities over approved eligible sections with
the existing engine. First test one track, then use `--limit 0` for all 32
validation tracks. The report includes original/converted metrics, frame and
section prediction agreement, and per-chunk logit errors. Defaults atol=.03,
rtol=.02 are provisional numerical smoke tolerances, not an accuracy guarantee;
inspect the validation F1 delta and individual class changes as well.
Full encoder preprocessing parity and iPhone compute-unit behavior remain
separate checks. This stage does not quantize weights or claim a 6-bit artifact.
