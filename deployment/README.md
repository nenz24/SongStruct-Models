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

The iPhone fixture preparation copies cached memmap slices into an owned, writable,
native float32 C-order [1,T,4096] array before Core ML prediction. A read-only
mmap/newaxis view can fail at the Python/Core ML value bridge even when its
shape/dtype are correct. The source cache is never modified. Input buffer
metadata is printed for diagnosis; feature contents are not logged.

## Impact of encoder conversion on section accuracy

Open `coreml_encoder_pipeline_experiment.ipynb`. Both branches use the same
verified, uncompressed FP32 Core ML head, approved eligible targets and original
section-probability pooling. The reference branch reads the frozen PyTorch
encoder cache. The candidate branch runs the five existing app frontend/encoder
packages with librosa-decoded audio, 30/120s encoder windows, and 420s head blocks.
Thus this isolates encoder conversion impact from head conversion/compression.
It is not yet Swift decoding, app bar snapping/merging, or iPhone latency parity.

Start with one complete song, then all validation. Core ML features are written
to a separate `outputs.local/coreml-encoder-cache`; frozen training features,
labels and checkpoints are unchanged. The cache is keyed by actual hashes of
encoder packages/manifest, feature profile, coremltools/librosa versions, audio
hash and CPU compute units. Completed tracks are reused after interruption;
changed provenance requires a new --cache-dir. Partial tracks are regenerated.
Allow several GiB additional disk for validation caches, with an upfront free-
space check. No encoders are downloaded or retrained.

Reports compare F1, per-class metrics, confusion matrices, coverage and section
prediction agreement, and list changed sections. Full-validation annotations,
split and source-cache provenance must match the prior FP32 proof. Prediction
component timings cover newly encoded tracks in this run only, exclude many
other costs, and are not iPhone/end-to-end timings. A diagnostic result is not
an automatic app-integration approval or a compressed-encoder result.

If an encoder produces NaN/Inf, the pipeline now stops at the first failing
component (log_mel, conv or encoder) before propagating it into other models.
`outputs.local/coreml-encoder-cache/last_encoder_failure.json` records track,
absolute sample window, short/long stream, component, input/output shapes and
finite ranges/counts. It contains no waveform/embedding arrays. Finite statistics
are diagnostics, not repaired data. No NaN replacement, failed-track exclusion,
window splitting, or precision fallback is applied. Existing complete diagnostic
tracks are reused on rerun; share the failure JSON to diagnose the specific model
and input before changing precision or normalization.

## Exact-window MusicFM NaN reproducer

Open `musicfm_failure_diagnostic.ipynb`. It uses the failure report's absolute
sample indices (for Pilihanku: 5760000..8617204 at 24kHz), checks source audio and
Core ML package hashes against the failure provenance, decodes/resamples the full
song before slicing, and stores the audio/conv/rotary arrays separately under a
new outputs.local folder. No clipping or normalization is added.

The same conv/rotary inputs run three times through one CPU_ONLY MusicFM instance,
once through a newly loaded instance, and once after the immediately preceding
long window on another fresh instance. This last test is a limited length-transition
probe, not replay of every earlier short/long window in the full-song run.
Raw outputs are preserved, including NaNs; reports list invalid frame indices and
fully versus partially invalid frame counts. Declared MIL tensor dtype counts
are inspected where available, without claiming actual kernel execution precision.

Original MusicFM PyTorch is then loaded on CPU FP32 from the exact weights/stats
used for cache extraction. No MuQ or training head is loaded. Core ML and PyTorch
run on the same original audio; the PyTorch result is a diagnostic reference,
not an automatic fallback in the evaluated pipeline. A finite-frame subset
similarity is explicitly labelled and never treats corrupted frames as valid.
Partial reports persist before the PyTorch phase. Share diagnostic.json (not the
raw waveform/features). No model is re-exported or repaired by this reproducer.

## Original-weight FP32 MusicFM candidate

Open `musicfm_fp32_repair_experiment.ipynb` after exact-window diagnostics. The
exporter loads original MusicFM weights/stats matching cache provenance, takes
only layers 0..9 (hidden_states[10], without the final encoder normalization),
and checks that wrapper against the installed native transformers encoder. It
exposes conv_in [1,T,1024] and rotary [2,T,1,1,64], T<=3000, retaining the current
app interface. FP32 compute is explicitly requested; it does not upcast the old
FP16 package or try to reconstruct precision already lost in old weights.

On the exact failed window, reports distinguish original versus app convolution
outputs, native versus app rotary values, original Core ML versus native PyTorch
on IDENTICAL saved conv/rotary inputs, and the new FP32 candidate versus PyTorch.
Source package/input hashes are checked. Dynamic tracing is tested on the long
failed window. All three real-input FP32 parity checks must pass before the
export report enables a pipeline override. Synthetic/CPU tests cannot validate
the real Apple SDK conversion; if it fails, share the partial repair report/log.

`evaluate_pipeline --musicfm-export-report ...` optionally uses this new MusicFM
package after checking exact-window proof, metadata and frozen weights/stats.
The frontend, MuQ, app rotary frequency values and FP32 head remain unchanged
for this first controlled experiment. The override is hashed into diagnostic
cache provenance, and the notebook uses a separate cache folder. No SongMap
resources are overwritten. Full validation and subsequent iPhone tests remain
required; FP32 storage/time may be larger than the current FP16 encoder.

## Compress repaired MusicFM, not the invalid legacy encoder

Open `musicfm_compression_experiment.ipynb`. It finds the repaired, uncompressed
MusicFM full-validation report, checks its source repair-report hash, and makes
independent weight-only INT8/per-channel and palette6/grouped-channel candidates
using the same compression settings as the head experiments. Smaller/unsupported
constants can remain float. Added compressed-weight operation counts and actual
package sizes are recorded. Activation quantization and integer execution are
not claimed; source FLOAT32 compute export is retained.

Each candidate is tested three times on the exact saved pathological conv/rotary
inputs. Any NaN/Inf or output-shape change blocks pipeline evaluation. Finite
lossy numerical mismatch is reported separately and is NOT treated as proof of
accuracy. A candidate proof explicitly says full validation is pending, and only
enables diagnostic evaluation, never app integration. Original models and the
verified FP32 source are unchanged.

For pipeline comparison, frontend, rotary, MuQ and head FP32 stay fixed. Use the
repaired Core ML pipeline F1 (~0.721639 in the current run), not the original
training-cache F1, as the encoder compression baseline. Both comparisons remain
visible in reports. Do not combine the palette head during this experiment.

`evaluate_pipeline --reuse-muq-cache <baseline diagnostic cache>` avoids rerunning
unchanged MuQ. It checks package hashes for log-mel, MusicFM conv, MuQ conv and
MuQ, runtime/librosa/platform/profile identity, cache provenance integrity,
audio hashes and exact chunk offsets. Only the MusicFM short/long columns are
replaced; MuQ columns are copied without mutation. Candidate frame counts must
align to the baseline within the existing four-frame tail tolerance. No silence,
label, boundary or NaN imputation is performed. A distinct candidate cache includes
the reused-MuQ provenance ID, and changed source models/settings cause rejection.
Candidate caches require additional disk; completed tracks are resumable.

Compare full-validation F1, per-class metrics, coverage, accuracy and model size.
Any speed/peak-memory decision still needs physical-iPhone encoder benchmarks.

## Compressed MusicFM on iPhone and cached head combinations

Open `musicfm_device_and_combination_experiment.ipynb`. Part A evaluates each
already cached MusicFM encoder candidate with the validated palette6 head.
Neither audio nor encoders are opened: only candidate feature files and the
compressed head. Split, annotations, checkpoint and cache provenance must match.
Reports retain metrics with the FP32 head on the SAME encoder so head effects
are isolated from encoder conversion effects. No test split is used.

Part B prepares a separate Xcode project with `prepare_musicfm_iphone`. By default
only MusicFM INT8 and palette6 are bundled; the large FP32 source stays on Mac
as the reference. Optional --include-fp32 bundles it for advanced testing. There
are two identical real-input fixtures per candidate: a 750-frame prefix and the
full saved Pilihanku conv/rotary window. The short prefix is not fresh 30s audio
frontend processing. Fixtures preserve both rotary planes; FP32 reference outputs
are computed for each actual input length. Candidate validation/source hashes are
checked before compiling separate model resources. No SongMap files are modified.

Select a signing team and physical iPhone in Xcode. The app runs one candidate
and fixture per click, avoiding simultaneous loading of large models. Start short
with all, then failed_full, then compare cpuAndNeuralEngine. Share JSON reports.
Metrics cover compiled load, first/warm predictions, non-finite counts for every
call, first-output embedding errors/cosine versus Mac FP32, and process-footprint
snapshots (NOT peak memory). It also records raw compiled-resource size, device,
OS, power/thermal state, input hashes and requested compute units (not proof of
ANE placement). No whole-song, frontend, MuQ, head or Apple timing is measured.
Source package size and compiled resource size are distinct. Restart the app
between candidates and repeat tests for cleaner performance comparisons.

Apple SDK compilation/inference cannot be established by Linux CPU tests; run
preparation/build on Mac and actual device tests before claiming deployment
readiness. Non-finite device outputs are recorded, not imputed or silently dropped.

### Fixed shapes after the iPhone MPSGraph assertion

A real-device run completed INT8 short/all, but a later run aborted inside
MPSGraph with invalid expand_dims axis and "shape for TensorData is not static".
This points to the GPU runtime/compiler path; PointerUI/LaunchServices/share
messages do not explain that assertion. The specific compiler defect is not
proven. It cannot be caught by Swift do/catch to perform a reliable fallback.

Preparation now materializes separate single-function fixed-shape packages for
each candidate/fixture with coremltools 8.3's public materialization API. It
reloads the dynamic source per shape because materialization can mutate the MIL
program. CPU parity is checked against that same compressed dynamic candidate
with a tighter 1e-3 tolerance, declared dtypes are inspected to detect unexpected
FLOAT16 introduction, and compressed-weight ops must remain. Exact input shapes
are checked in the saved spec before compiling. Receipts remain beside the project.

Swift selects a distinct compiled resource per fixture, reports shape policy and
resource name, and prints the active case before model load/prediction. UI defaults
to cpuOnly for initial checks; GPU modes are an explicit subsequent experiment.
This is a candidate workaround, not a verified iOS runtime fix. Source compressed
packages, encoder caches and accuracy reports are unchanged. Static copies
duplicate weights in the benchmark bundle, so this is not the final app packaging.
Regenerate in a new output folder via notebook setup then Part B; no need to rerun
Part A, training, compression, or encoder feature extraction.
