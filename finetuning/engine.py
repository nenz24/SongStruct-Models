"""Section-label objective and metrics; no boundary supervision."""
import math
from pathlib import Path
from .common import LABELS, CLASS_IDS, fingerprint, iter_features, read, revision


def manifest(cfg):
    root = Path(cfg['work_dir'])
    data, plan = read(root / 'manifest.json'), read(root / 'split_plan.json')
    if (not data['groups_confirmed'] or not plan['groups_confirmed']
            or data['plan_id'] != plan['id'] or fingerprint({k: v for k, v in plan.items() if k != 'id'}) != plan['id']):
        raise ValueError('Split belum dikonfirmasi atau identitas split berubah.')
    seen = {}
    for row in data['records']:
        if plan['groups'].get(row['track_id']) != row['group_id'] or plan['group_splits'][row['group_id']] != row['split']:
            raise ValueError('Record tidak sesuai frozen split.')
        if row['audio_sha256'] in seen and seen[row['audio_sha256']] != row['split']:
            raise ValueError('Audio identik bocor antar split.')
        seen[row['audio_sha256']] = row['split']
    provenance = read(root / 'encoder_provenance.json')
    if revision(cfg) != provenance['upstream_revision']:
        raise ValueError('Upstream source changed after feature extraction.')
    return data, provenance['id']


def configure_training(model, stage):
    if stage not in ('function_head', 'label_backbone'):
        raise ValueError('trainable must be function_head or label_backbone')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('function_head.') if stage == 'function_head' else not name.startswith('boundary_head.'))
    return [p for p in model.parameters() if p.requires_grad]


def train_mode(model, stage):
    if stage == 'function_head':
        # Keep all frozen dropout paths deterministic, matching inference.
        model.eval()
        model.function_head.train()
    else:
        model.train()
        model.boundary_head.eval()


def logits(model, values, target):
    import numpy as np
    import torch
    embeddings = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).unsqueeze(0).to(target)
    return model.forward_func({'input_embeddings': embeddings,
                               'dataset_ids': torch.tensor([5], dtype=torch.long, device=target),
                               'masks': None})['function_logits'][0][:, list(CLASS_IDS)]


def pooled_sections(scores, sections, offset):
    import torch
    probabilities = scores.float().softmax(dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError('NaN/Inf in function probabilities.')
    times = torch.arange(scores.shape[0], dtype=torch.float32, device=scores.device) * (3.0 / 25) + offset
    result = []
    for index, section in enumerate(sections):
        weights = (torch.minimum(times + 3.0 / 25, torch.tensor(section['end'], device=times.device))
                   - torch.maximum(times, torch.tensor(section['start'], device=times.device))).clamp_min(0)
        covered = weights.sum()
        if covered.item() > 0:
            pooled = (probabilities * weights[:, None]).sum(0) / covered
            result.append((index, pooled, float(covered.item())))
    return result


def label_loss(scores, sections, offset):
    import torch
    terms = [-p[LABELS.index(sections[i]['label'])].clamp_min(1e-8).log()
             for i, p, _ in pooled_sections(scores, sections, offset)]
    return torch.stack(terms).mean() if terms else None


def metrics_from_sections(predictions):
    matrix = [[0] * len(LABELS) for _ in LABELS]
    total_time = correct_time = loss = 0.0
    low_coverage = 0
    for true, pred, duration, probability, coverage in predictions:
        matrix[true][pred] += 1
        total_time += duration
        correct_time += duration * (true == pred)
        loss -= math.log(max(probability, 1e-8))
        low_coverage += coverage < 0.98
    count = len(predictions)
    f1s, details = [], {}
    for i, name in enumerate(LABELS):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        details[name] = {'support': support, 'f1': f1}
        if support:
            f1s.append(f1)
    return {'sections': count, 'section_accuracy': sum(matrix[i][i] for i in range(len(LABELS))) / count if count else 0,
            'duration_weighted_accuracy': correct_time / total_time if total_time else 0,
            'macro_f1_present': sum(f1s) / len(f1s) if f1s else 0,
            'section_nll': loss / count if count else 0,
            'low_coverage_sections': low_coverage, 'per_class': details,
            'label_order': list(LABELS), 'confusion_matrix': matrix}


def evaluate(model, records, cfg, target, provenance_id):
    import torch
    model.eval()
    predictions = []
    with torch.inference_mode():
        for row in records:
            sections = row['sections']
            sums = [[0.0] * len(LABELS) for _ in sections]
            coverage = [0.0] * len(sections)
            for values, offset, provenance in iter_features(cfg, row):
                if provenance != provenance_id:
                    raise ValueError('Mixed feature provenance.')
                scores = logits(model, values, target)
                for i, probabilities, covered in pooled_sections(scores, sections, offset):
                    for j, p in enumerate(probabilities.cpu().tolist()):
                        sums[i][j] += p * covered
                    coverage[i] += covered
            for i, section in enumerate(sections):
                if coverage[i] <= 0:
                    raise ValueError(f"No feature coverage for {row['track_id']} section {i + 1}")
                probs = [v / coverage[i] for v in sums[i]]
                true = LABELS.index(section['label'])
                pred = max(range(len(LABELS)), key=lambda j: probs[j])
                duration = section['end'] - section['start']
                predictions.append((true, pred, duration, probs[true], coverage[i] / duration))
    return metrics_from_sections(predictions)
