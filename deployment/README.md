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

## Diagnose FP16 drift with FP32

Open `coreml_precision_comparison.ipynb`. Reuse the existing FP16 package, export
a new package with `--precision fp32` to a separate directory, and run
`verify_coreml --diagnostic --limit 0` for both on the same validation inventory.
`--diagnostic` suppresses only the final exception for finite numerical tolerance
mismatch. It does not relax atol/rtol or mark a candidate approved; reports retain
`logit_parity_passed=false`, mismatch status and `ready_for_app_integration=false`.
Shape, non-finite outputs, checkpoint mismatch and runtime errors still stop the
process. The default verifier remains strict.

Reports now include all-logit and allowed-class-logit checks, probability max/mean
errors after 8-class softmax, per-frame argmax agreement, section agreement, and
validation F1/confusion matrices. The generic metrics key is `coreml` for both
precisions. `coreml_fp16` remains an alias only when candidate name is fp16.
Precision is recorded in new export metadata; old FP16 exports have no precision
metadata, so `--candidate-name fp16` is an informational label only.

The notebook checks plan, provenance, annotation identity and checkpoint equality
before comparing reports. FP32 closer than FP16 would support a precision-related
explanation; drift in both calls for investigating attention/export paths on real
long chunks. Neither outcome alone proves a cause. Full-model encoder parity and
physical-iPhone performance are still separate from head validation.

## Independent weight-compression candidates

Open `coreml_compression_experiment.ipynb` after a successful full FP32 validation.
The notebook finds the latest successful `coreml-comparison-*/fp32-validation.json`;
inspect the printed source paths before running. Each candidate starts from the
same uncompressed FP32 head, not from FP16 or another compressed model.

`deployment.compress_coreml --package ... --validation-report ... --scheme int8
--output-dir ...` uses symmetric per-channel INT8 weight compression. Scheme
`palette6` uses kmeans, 6-bit LUT indices, grouped channels of 32, and one worker.
Both use a 2048-element threshold; smaller constants and unsupported weights can
remain float. Operation counts must show newly added compressed-weight ops or
the command fails. Metadata, source validation hash, package sizes and actual
compression ratio are reported. Existing packages are never overwritten.
The source report path/checkpoint metadata must match and full numerical
validation must have passed. This assumes the locally verified source package
has not been modified since validation.

These APIs compress weights, not activations. FLOAT32 compute is inherited from
the source; neither candidate promises INT8/INT6 execution or lower latency.
`precision` metadata describes the source compute export; `compression_scheme`
describes storage separately. Grouping keeps more codebooks than a single LUT
and trades a little storage for potentially better accuracy. Actual speed and
peak memory must be measured on a physical iPhone after accuracy verification.

The notebook verifies one track then all validation tracks, using diagnostic
mode without changing the numerical tolerances. It compares F1, accuracy,
per-class metrics, prediction agreement and size. No candidate is automatically
approved. This stage covers the label pipeline only; the encoders account for
most of the app's size and require separate parity/compression experiments.

## Physical iPhone benchmark, isolated from SongMap

Open `iphone_benchmark_experiment.ipynb` on the Mac. `prepare_iphone` creates a new
standalone Xcode project under outputs.local, precompiles distinct FP32/INT8/
palette6 resources with Xcode coremlcompiler, and includes the same real cached
feature prefix for each. By default it represents 30s audio; optional --seconds
120 or 420 produces larger fixtures for subsequent tests. The reference logits
are Mac Core ML FP32, not manual labels or a full accuracy evaluation.

Open Benchmark.xcodeproj, select your personal signing team and physical iPhone,
then Run. Shared SongMap project, signing and app resources are untouched. The
Swift target uses optimized builds even for Debug. No Apple signing credentials
are stored by the preparation script. Share the JSON from the benchmark screen.
Try all and cpuAndNeuralEngine modes, then the other modes if useful. Requested
compute units do not prove every operation executes on ANE. Avoid simulator
results when comparing physical iPhone performance. Run without debugger for
final timings and repeat under comparable power/thermal conditions.

Reports contain model-load time, first prediction, 5 warm prediction times,
source package bytes, frame/probability differences versus the identical Mac
reference, hardware, OS, thermal state and process-footprint snapshots. Footprints
are not peak memory and include app/fixture/runtime allocations. Model order can
affect allocator state. Load/first timing is not guaranteed cold OS cache. These
measurements do not include encoder processing, Apple boundaries, or all-song
accuracy; no end-to-end speedup is claimed. Swift/iOS compilation must be checked
on the Mac: Linux CPU tests do not validate the Apple SDK build.

## Existing encoder checks before compression

`verify_encoders --songmap-root ... --seconds 30 --output ...` checks the first
validation song's short-window FM/MuQ streams against corresponding cache columns.
Repeat with 120 for long-window streams. Full audio is resampled before slicing,
matching feature extraction at window edges. It uses the app's reflected log-mel
input, conv packages and rotary frequency arrays, including their stored precision.
It allows at most four extra/missing tail frames and reports the actual compared
prefix; no boundary alignment is silently changed. This is a limited diagnostic,
not a full encoder parity pass or compressed-encoder evaluation.

Code inspection also found that SongMap SectionLabeler currently performs majority
voting on frame labels, snaps boundaries to barlines, folds silence/short sections
and merges repeats. Those operations differ from the fixed Apple boundaries and
probability pooling used in training evaluation. Head benchmark success therefore
does not establish parity of the app's rendered section map.
