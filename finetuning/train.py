"""Fine-tune label prediction only, with frozen SSL caches and validation early stopping."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from .common import config, device, fingerprint, iter_features, load_head, sha_file, write
from .engine import configure_training, evaluate, label_loss, logits, manifest, train_mode, training_windows, training_class_weights


def save_weights(path, model, **metadata):
    import torch
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, **metadata}, path)


def export_ema(path, model):
    from ema_pytorch import EMA
    from safetensors.torch import save_file
    wrapper = EMA(model, include_online_model=False)
    wrapper.initted.fill_(True)
    save_file({k: v.detach().cpu().contiguous() for k, v in wrapper.state_dict().items()}, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--smoke-test', action='store_true', help='Run one training window including backward and optimizer step, then exit without creating a run.')
    args = parser.parse_args()
    cfg = config(args.config)
    data, provenance = manifest(cfg)
    train = [r for r in data['records'] if r['approved'] and r['split'] == 'train']
    val = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    if not train or not val:
        raise ValueError('Training membutuhkan approved audio pada train DAN val. Jangan memindahkan holdout untuk memaksa training.')
    import torch
    import numpy as np
    settings = cfg['training']
    if min(settings['epochs'], settings['gradient_accumulation'], settings['patience']) < 1:
        raise ValueError('epochs, accumulation, patience must be positive.')
    class_counts, class_weights = training_class_weights(train, settings.get('class_weighting', 'none'))
    print('Class weighting:', settings.get('class_weighting', 'none'), flush=True)
    print(json.dumps({'train_counts': class_counts, 'weights': class_weights}), flush=True)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    target = device(args.device)
    run_name = args.run_name or datetime.now(timezone.utc).strftime('worship-%Y%m%d-%H%M%S')
    if Path(run_name).name != run_name or run_name in ('.', '..'):
        raise ValueError('run-name must be a simple directory name.')
    run = Path(cfg['work_dir']) / 'runs' / run_name
    if not args.smoke_test:
        run.mkdir(parents=True, exist_ok=False)
    model, _ = load_head(cfg, target)
    params = configure_training(model, settings['trainable'])
    optimizer = torch.optim.AdamW(params, lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    window_seconds = settings.get('window_seconds')
    checkpointing = settings.get('gradient_checkpointing', False)
    if window_seconds is not None:
        # Validate before baseline evaluation or starting an epoch.
        list(training_windows(np.empty((0, 4096)), 0, window_seconds))
    if args.smoke_test:
        train_mode(model, settings['trainable'])
        for row in train:
            for values, offset, signature in iter_features(cfg, row):
                if signature != provenance:
                    raise ValueError('Mixed encoder caches.')
                for piece, start in training_windows(values, offset, window_seconds):
                    scores = logits(model, piece, target, checkpoint=checkpointing)
                    loss = label_loss(scores, row['sections'], start, fractional=window_seconds is not None, class_weights=class_weights)
                    if loss is None:
                        continue
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite smoke loss.')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, settings['max_grad_norm'], error_if_nonfinite=True)
                    optimizer.step()
                    print(f'Smoke OK: {len(piece)/25:.2f}s, loss={loss.item():.4f}; backward + optimizer completed. No checkpoint saved.', flush=True)
                    return
        raise ValueError('No supervised window for smoke test.')
    metadata = {'plan_id': data['plan_id'], 'provenance_id': provenance, 'config': cfg,
                'initial_checkpoint_sha256': sha_file(cfg['checkpoint']),
                'annotations_id': fingerprint([(r['id'], r['annotation_sha256']) for r in train + val]),
                'trainable': settings['trainable'], 'train_class_counts': class_counts, 'class_weights': class_weights}
    write(run / 'run.json', metadata)
    print(f'Train {len(train)}, val {len(val)}; device={target}; trainable={sum(p.numel() for p in params):,}', flush=True)
    baseline = evaluate(model, val, cfg, target, provenance)
    write(run / 'baseline_val.json', baseline)
    best = baseline['macro_f1_present']
    save_weights(run / 'best.pt', model, epoch=0, validation=baseline, **metadata)
    export_ema(run / 'best.safetensors', model)
    bad_epochs = 0
    for epoch in range(1, settings['epochs'] + 1):
        train_mode(model, settings['trainable'])
        order = list(train)
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        accumulated, losses, steps = 0, [], 0

        def update(count):
            # Average actual accumulated chunks, including the final short group.
            for parameter in params:
                if parameter.grad is not None:
                    parameter.grad.div_(count)
            torch.nn.utils.clip_grad_norm_(params, settings['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for row in order:
            for values, offset, signature in iter_features(cfg, row):
                if signature != provenance:
                    raise ValueError('Mixed encoder caches.')
                for piece, start in training_windows(values, offset, window_seconds):
                    scores = logits(model, piece, target, checkpoint=checkpointing)
                    loss = label_loss(scores, row['sections'], start, fractional=window_seconds is not None, class_weights=class_weights)
                    if loss is None:
                        continue
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite label loss.')
                    loss.backward()
                    accumulated += 1
                    losses.append(loss.item())
                    if accumulated == settings['gradient_accumulation']:
                        update(accumulated)
                        accumulated = 0
                        steps += 1
                    del scores, loss
        if accumulated:
            update(accumulated)
            steps += 1
        if not losses:
            raise ValueError('Tidak ada supervised section pada train split.')
        result = evaluate(model, val, cfg, target, provenance)
        record = {'epoch': epoch, 'train_loss': sum(losses) / len(losses), 'optimizer_steps': steps, 'val': result}
        with (run / 'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        save_weights(run / 'latest.pt', model, epoch=epoch, validation=result, **metadata)
        print(f"Epoch {epoch}: loss={record['train_loss']:.4f}, val F1={result['macro_f1_present']:.4f}, acc={result['section_accuracy']:.4f}", flush=True)
        if result['macro_f1_present'] > best:
            best, bad_epochs = result['macro_f1_present'], 0
            save_weights(run / 'best.pt', model, epoch=epoch, validation=result, **metadata)
            export_ema(run / 'best.safetensors', model)
        else:
            bad_epochs += 1
            if bad_epochs >= settings['patience']:
                print('Early stopping; baseline retained if no validation improvement.')
                break
    print(f'Best model: {run / "best.safetensors"}. Boundary head tidak dioptimasi; MuQ/MusicFM tidak dilatih.')


if __name__ == '__main__':
    main()
