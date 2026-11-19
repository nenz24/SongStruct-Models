#!/usr/bin/env python3
"""Compare tiny context adapters with/without explicit audio recurrence; frozen manual train/val plan."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from audit_manual_context import LABELS, load_frozen_inputs, score, validate_details
from evaluate_frozen_context import verify_scores
from train_section_context import atomic_torch_save

PROTOCOL = 'repetition-context-v1'
MODES = ('context_adapter', 'repetition_adapter')
FEATURE_POLICY = dict(name='centered-four-stream-cosine-v1', streams=4, width=1024,
                      threshold=.25, excluded_neighbor_distance=1, maximum_sections=512)
SETTINGS = dict(hidden=32, dropout=.2, epochs=20, learning_rate=1e-4, weight_decay=.01,
                residual_penalty=.02, accumulation=8, max_grad_norm=1., patience=5, seed=42)
GUARDS = dict(chorus_max_f1_drop=.01, inst_max_f1_drop=.02, bridge_max_f1_drop=0.)


def recurrence_graph(audio):
    """No labels, priors or timings. Four NxN products, never an NxNx4096 allocation."""
    if (audio.ndim != 2 or audio.shape[1] != 4096 or not 0 < len(audio) <= FEATURE_POLICY['maximum_sections']
            or not torch.isfinite(audio).all()):
        raise ValueError('Expected 1..512 finite pooled sections, width 4096.')
    n = len(audio)
    streams = audio.float().reshape(n, 4, 1024)
    streams = F.normalize(streams - streams.mean(0, keepdim=True), dim=-1, eps=1e-6)
    similarity = sum(streams[:, j] @ streams[:, j].T for j in range(4)) / 4
    similarity = similarity.clamp(-1, 1)
    index = torch.arange(n, device=audio.device)
    allowed = (index[:, None] - index[None, :]).abs() > 1
    positive = (similarity - FEATURE_POLICY['threshold']).clamp_min(0) * allowed
    weights = positive.square()
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
    return similarity, weights, allowed


def adapter_features(audio, timing, hidden, residual, mode):
    if mode not in MODES: raise ValueError('Unknown adapter mode.')
    n = len(audio)
    if (timing.shape != (n, 6) or hidden.ndim != 2 or len(hidden) != n or residual.shape != (n, 8)
            or any(not torch.isfinite(t).all() for t in (timing, hidden, residual))):
        raise ValueError('Invalid frozen representation shapes/values.')
    similarity, weights, allowed = recurrence_graph(audio)
    positive = (similarity > FEATURE_POLICY['threshold']) & allowed
    positions = timing[:, 0]
    stats = torch.stack([
        similarity.clamp_min(0).masked_fill(~positive, 0).max(1).values,
        positive.sum(1) / allowed.sum(1).clamp_min(1),
        (weights * similarity).sum(1),
        (weights * (positions[:, None] - positions[None, :]).abs()).sum(1),
        weights @ positions,
        -(weights * weights.clamp_min(1e-8).log()).sum(1) / math.log(max(n, 2)),
    ], dim=1)
    neighbor_hidden, neighbor_residual = weights @ hidden, weights @ residual
    if mode == 'context_adapter':
        neighbor_hidden = torch.zeros_like(neighbor_hidden)
        neighbor_residual = torch.zeros_like(neighbor_residual)
        stats = torch.zeros_like(stats)
    features = torch.cat((hidden, neighbor_hidden, residual, neighbor_residual, stats), dim=1)
    if not torch.isfinite(features).all(): raise ValueError('Non-finite adapter features.')
    return features


class RepetitionAdapter(nn.Module):
    def __init__(self, width=150, hidden=32, dropout=.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 8))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features)


def prepare_teacher_samples(teacher, samples, target):
    """Precompute once without gradients; eligible labels are loss targets only."""
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Teacher must stay frozen in eval mode.')
    prepared = []
    for i, sample in enumerate(samples, 1):
        if sample['split'] not in ('train', 'val'): raise ValueError('No test inference in this experiment.')
        # Ordinary no_grad tensors can subsequently be used as adapter training inputs.
        with torch.no_grad():
            audio, timing = sample['audio'].to(target), sample['timing'].to(target)
            hidden = teacher.context((teacher.audio(audio) + teacher.position(timing)).unsqueeze(0))[0]
            residual = teacher.output(hidden)
            scores = sample['probabilities'].to(target).clamp_min(1e-8).log() + residual[sample['indices'].to(target)]
            hidden, residual = hidden.cpu(), residual.cpu()
            features = {mode: adapter_features(sample['audio'], sample['timing'], hidden, residual, mode).detach() for mode in MODES}
        prepared.append(dict(track_id=sample['track_id'], split=sample['split'], identity=sample['identity'],
            indices=sample['indices'], labels=sample['labels'], details=sample['details'], base_scores=scores.cpu().detach(),
            features=features))
        print(f'Frozen teacher [{i}/{len(samples)}]: {sample["split"]} {sample["track_id"]}', flush=True)
    return prepared


def predict(adapter, samples, mode):
    if adapter is not None and adapter.training: raise ValueError('Prediction requires eval mode.')
    details = []; loss_sum = 0.
    with torch.no_grad():
        for sample in samples:
            if sample['split'] not in ('train', 'val'): raise ValueError('Test is not used.')
            scores = sample['base_scores']
            if adapter is not None: scores = scores + adapter(sample['features'][mode])[sample['indices']]
            if not torch.isfinite(scores).all(): raise ValueError('Non-finite adapter logits.')
            loss_sum += F.cross_entropy(scores, sample['labels'], reduction='sum').item()
            for base, probabilities, index in zip(sample['details'], scores.softmax(-1).tolist(), sample['indices'].tolist()):
                order = sorted(range(8), key=lambda j: probabilities[j], reverse=True)
                details.append({**base, 'prediction': LABELS[order[0]], 'probabilities': dict(zip(LABELS, probabilities)),
                    'confidence': probabilities[order[0]], 'margin': probabilities[order[0]] - probabilities[order[1]],
                    'original_section_index': index + 1})
    result = score(details); result['section_nll'] = loss_sum / len(details)
    return result, details


def supervised_loss(adapter, sample, mode, penalty):
    if sample['split'] != 'train': raise ValueError('Only manual TRAIN targets receive gradients.')
    delta = adapter(sample['features'][mode])[sample['indices']]
    loss = F.cross_entropy(sample['base_scores'] + delta, sample['labels'], reduction='sum')
    loss = loss + penalty * delta.square().mean(-1).sum()
    if not torch.isfinite(loss): raise ValueError('Non-finite training loss.')
    return loss


def guard_status(candidate, baseline):
    return {label: candidate['per_class'][label]['f1'] >= baseline['per_class'][label]['f1'] - drop - 1e-12
            for label, drop in (('chorus', GUARDS['chorus_max_f1_drop']), ('inst', GUARDS['inst_max_f1_drop']),
                                ('bridge', GUARDS['bridge_max_f1_drop']))}


def choose_candidate(baseline, candidates):
    selected = 'identity'; best = baseline['macro_f1_present']
    # Prefer the simpler ablation if scores tie; do not claim recurrence helps without beating it.
    for mode in MODES:
        metrics = candidates[mode]['validation']
        if all(guard_status(metrics, baseline).values()) and metrics['macro_f1_present'] > best + 1e-12:
            selected, best = mode, metrics['macro_f1_present']
    return selected


def train_epoch(adapter, optimizer, samples, mode, settings):
    if any(s['split'] != 'train' for s in samples): raise ValueError('Train iterator contains a non-train record.')
    adapter.train(); order = list(samples); random.shuffle(order)
    optimizer.zero_grad(set_to_none=True); accumulated = targets = 0; total = 0.
    for i, sample in enumerate(order, 1):
        loss = supervised_loss(adapter, sample, mode, settings['residual_penalty']); loss.backward()
        accumulated += len(sample['labels']); targets += len(sample['labels']); total += loss.item()
        if i % settings['accumulation'] == 0 or i == len(order):
            for p in adapter.parameters():
                if p.grad is not None: p.grad.div_(accumulated)
            nn.utils.clip_grad_norm_(adapter.parameters(), settings['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); accumulated = 0
    return total / targets


def fit_mode(train, val, mode, baseline, settings, identity, folder, read, write, epoch_budget=None):
    """Atomic epoch resume, including shuffle/dropout RNG and optimizer state."""
    if any(s['split'] != 'val' for s in val): raise ValueError('Validation iterator must contain only val.')
    random.seed(settings['seed']); torch.manual_seed(settings['seed'])
    model = RepetitionAdapter(train[0]['features'][mode].shape[1], settings['hidden'], settings['dropout'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    model.eval(); initial, _ = predict(model, val, mode); verify_scores(initial, baseline)
    best_score = baseline['macro_f1_present']; best_epoch = epoch = bad = 0; history = []
    best_state = deepcopy(model.state_dict())
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder/'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)  # Own local optimizer/RNG checkpoint only.
        if state['identity'] != identity or state['mode'] != mode: raise ValueError('Resume checkpoint inputs differ.')
        model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
        best_state, best_score, best_epoch, epoch, bad, history = (state[k] for k in ('best_model','best_score','best_epoch','epoch','bad_epochs','history'))
        random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
    def save_latest():
        atomic_torch_save(latest, dict(identity=identity, mode=mode, epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), best_model=best_state, best_score=best_score, best_epoch=best_epoch,
            bad_epochs=bad, history=history, python_rng=random.getstate(), torch_rng=torch.get_rng_state()))
        write(folder/'history.json', history)
    save_latest()
    end = settings['epochs'] if epoch_budget is None else min(settings['epochs'], epoch + epoch_budget)
    while epoch < end and bad < settings['patience']:
        started = time.perf_counter()
        train_loss = train_epoch(model, optimizer, train, mode, settings)
        model.eval(); metrics, _ = predict(model, val, mode)
        guards = guard_status(metrics, baseline)
        improved = all(guards.values()) and metrics['macro_f1_present'] > best_score + 1e-12
        epoch += 1
        if improved:
            best_score, best_epoch, bad = metrics['macro_f1_present'], epoch, 0
            best_state = deepcopy(model.state_dict())
        else: bad += 1
        history.append(dict(epoch=epoch, train_loss=train_loss, validation=metrics, guards=guards,
                            selected=improved, seconds=time.perf_counter()-started))
        save_latest()
        print(f'{mode} epoch {epoch}: loss={train_loss:.4f}, val F1={metrics["macro_f1_present"]:.4f}, '
              f'acc={metrics["section_accuracy"]:.4f}, guards={guards}' + (' * best' if improved else ''), flush=True)
    model.load_state_dict(best_state, strict=True); model.eval()
    validation, predictions = predict(model, val, mode); train_metrics, _ = predict(model, train, mode)
    atomic_torch_save(folder/'best.pt', dict(identity=identity, architecture=PROTOCOL, mode=mode, settings=settings,
        feature_policy=FEATURE_POLICY, guards=GUARDS, model=best_state, best_epoch=best_epoch))
    return dict(best_epoch=best_epoch, identity_retained=best_epoch==0, train=train_metrics, validation=validation,
                guards=guard_status(validation, baseline), parameters=sum(p.numel() for p in model.parameters()),
                epochs_completed=epoch), predictions


def validate_audit(audit, inputs):
    if (audit.get('protocol') != 'manual-context-audit-v1' or not audit.get('full_train_val')
            or not audit.get('original_context_validation_reproduced')
            or audit.get('context_checkpoint_sha256') != inputs['checkpoint_sha']
            or audit.get('plan_id') != inputs['data']['plan_id'] or audit.get('provenance_id') != inputs['provenance']
            or audit.get('source_run_sha256') != inputs['sha_file'](inputs['run_path'])
            or audit.get('source_calibration_sha256') != inputs['sha_file'](inputs['calibration_path'])):
        raise ValueError('A matching full manual-context audit with reproduced validation is required.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songstruct-root', type=Path, default=Path(__file__).resolve().parents[2]/'Section-Annotations')
    parser.add_argument('--frozen-evaluation', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke-test', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    inputs = load_frozen_inputs(args.songstruct_root, args.frozen_evaluation, args.cache_dir, 1 if args.smoke_test else 0)
    from finetuning.common import read, write, fingerprint, sha_file
    inputs['sha_file'] = sha_file
    audit_path = args.audit.expanduser().resolve(); audit = read(audit_path); validate_audit(audit, inputs)
    teacher_sha = sha_file(inputs['snapshot']); audit_sha = sha_file(audit_path)
    prepared = prepare_teacher_samples(inputs['model'], inputs['samples'], inputs['target'])
    del inputs['samples']  # No need to retain pooled 4096-wide tensors while training adapters on CPU.
    torch.set_num_threads(2)
    groups = {split: [s for s in prepared if s['split']==split] for split in ('train','val')}
    base = {}
    for split, samples in groups.items():
        base[split], details = predict(None, samples, MODES[0])
        validate_details([r for r in inputs['rows'] if r['split']==split], details)
        if not args.smoke_test: verify_scores(base[split], audit['splits'][split]['context'])
    if args.smoke_test:
        for mode in MODES:
            torch.manual_seed(SETTINGS['seed']); random.seed(SETTINGS['seed'])
            model=RepetitionAdapter(prepared[0]['features'][mode].shape[1], SETTINGS['hidden'], SETTINGS['dropout'])
            model.eval(); initial,_=predict(model,groups['val'],mode); verify_scores(initial,base['val'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=SETTINGS['learning_rate'])
            loss=train_epoch(model,optimizer,groups['train'],mode,SETTINGS)
            model.eval(); predict(model,groups['val'],mode)
            print(f'Smoke OK: {mode}; {sum(p.numel() for p in model.parameters()):,} parameters, loss={loss:.4f}',flush=True)
        if sha_file(inputs['snapshot'])!=teacher_sha or sha_file(audit_path)!=audit_sha: raise ValueError('Source files changed.')
        print('No run/checkpoint saved. Teacher/encoders/manual targets unchanged.',flush=True); return
    source = dict(protocol=PROTOCOL, settings=SETTINGS, guards=GUARDS, feature_policy=FEATURE_POLICY,
        context_checkpoint_sha256=teacher_sha, audit_sha256=audit_sha, plan_id=inputs['data']['plan_id'],
        provenance_id=inputs['provenance'], sample_identities=[[s['track_id'],s['identity']] for s in prepared],
        teacher_device=str(inputs['target']), adapter_device='cpu', torch=str(torch.__version__),
        source_sha256=sha_file(Path(__file__)), loader_sha256=sha_file(Path(__file__).with_name('audit_manual_context.py')))
    identity=fingerprint(source); out=args.output_dir.expanduser().resolve()
    if (out/'run.json').exists():
        if not args.resume or read(out/'run.json')!={'identity':identity,**source}:
            raise ValueError('Run exists/changed: use --resume with identical inputs, or a fresh output directory.')
    else:
        if args.resume: raise ValueError('No run to resume.')
        out.mkdir(parents=True,exist_ok=False); write(out/'run.json',{'identity':identity,**source})
    print(f'Train {len(groups["train"])}, val {len(groups["val"])}; frozen context epoch 5; adapters CPU only.',flush=True)
    candidates={}; predictions={}
    for mode in MODES:
        candidates[mode],predictions[mode]=fit_mode(groups['train'],groups['val'],mode,base['val'],SETTINGS,
            identity,out/mode,read,write)
        validate_details(inputs['groups']['val'],predictions[mode])
        write(out/mode/'validation-predictions.json',dict(identity=identity,sections=predictions[mode]))
    if sha_file(inputs['snapshot'])!=teacher_sha or sha_file(audit_path)!=audit_sha: raise ValueError('Frozen source changed during training.')
    selected=choose_candidate(base['val'],candidates)
    _,before=predict(None,groups['val'],MODES[0]); changes={}
    for mode in MODES:
        changes[mode]=[dict(track_id=a['track_id'],start=a['start'],end=a['end'],label_manual=a['label_manual'],
            baseline_prediction=a['prediction'],candidate_prediction=b['prediction']) for a,b in zip(before,predictions[mode])
            if a['prediction']!=b['prediction']]
    delta=candidates[MODES[1]]['validation']['macro_f1_present']-candidates[MODES[0]]['validation']['macro_f1_present']
    report=dict(protocol=PROTOCOL,experiment_identity=identity,split='val',tracks=len(groups['val']),full_validation=True,
        plan_id=inputs['data']['plan_id'],provenance_id=inputs['provenance'],context_checkpoint_sha256=teacher_sha,
        baseline=base,candidates=candidates,selected=selected,repetition_minus_ablation_macro_f1=delta,
        explicit_repetition_improvement_observed=selected==MODES[1] and delta>1e-12,changed_sections=changes,
        test_used=False,encoder_executed=False,teacher_weights_changed=False,manual_labels_changed=False,
        boundary_changed=False,ready_for_app_integration=False,feature_cache_read_only=True,
        limitations=['Validation selected model/epoch and has guided earlier experiments; not a new independent final accuracy estimate.',
            'Teacher already uses global context. Ablation removes only the NEW explicit audio recurrence features.',
            'Original encoder cache + PyTorch context only; palette6/Core ML, audio boundary accuracy and iPhone speed are not evaluated.'])
    write(out/'comparison.json',report)
    print('Selected:',selected,'| recurrence minus ablation F1:',f'{delta:+.6f}',flush=True)
    print('Report:',out/'comparison.json',flush=True)


if __name__=='__main__':main()
