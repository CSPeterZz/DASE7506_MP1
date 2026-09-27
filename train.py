"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
# Final defaults below supersede the original baseline recipe documented above.
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from student import MODEL_OPTIONS


TRAIN_OPTIONS = {
    'run_dir': 'runs/final-s17',
    'device': 'cuda', 'precision': 'fp32', 'threads': 16, 'seed': 17,
    'steps': 3000, 'batch_size': 128, 'eval_every': 100, 'log_every': 100,
    'peak_lr': 0.002, 'warmup_steps': 100, 'min_lr_ratio': 0.1,
    'schedule': 'cosine',  # 'cosine' or 'linear_warmdown'
    'warmdown_steps': 700,
    'weight_decay': 0.1, 'group_weight_decay': False, 'grad_clip': 1.0,
    'use_ema': True, 'ema_start_step': 400, 'ema_decay': 0.99,
    'use_muon': True, 'muon_lr': 0.02, 'muon_momentum': 0.95,
    'muon_ns_steps': 5, 'muon_weight_decay': 0.01,
}


class MatrixMuon(torch.optim.Optimizer):
    """Single-device Muon with FP32 Newton-Schulz updates for CPU/CUDA.

    Algorithm reference: https://github.com/KellerJordan/Muon
    Uses EMA momentum, Nesterov blending and aspect-ratio update scaling.
    Embeddings, output weights and gates must be handled by AdamW instead.
    """

    def __init__(self, parameters, options):
        super().__init__(parameters, dict(
            lr=options['muon_lr'], momentum=options['muon_momentum'],
            ns_steps=options['muon_ns_steps'], weight_decay=options['muon_weight_decay'],
        ))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2 or parameter.grad.is_sparse:
                    raise ValueError('MatrixMuon requires dense matrix gradients.')
                gradient = parameter.grad.float()
                state = self.state[parameter]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(gradient)
                momentum = state['momentum']
                momentum.lerp_(gradient, 1 - group['momentum'])
                update = gradient.lerp(momentum, group['momentum'])
                rows, columns = update.shape
                transpose = rows > columns
                with torch.autocast(device_type=parameter.device.type, enabled=False):
                    x = update.T if transpose else update
                    x = x / (x.norm() + 1e-7)
                    for _ in range(group['ns_steps']):
                        gram = x @ x.T
                        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
                    update = (x.T if transpose else x) * math.sqrt(max(1, rows / columns))
                parameter.mul_(1 - group['lr'] * group['weight_decay'])
                parameter.add_(update.to(parameter.dtype), alpha=-group['lr'])
        return loss


def learning_rate(step, options):
    warmup = min(1.0, (step + 1) / max(1, options['warmup_steps']))
    if options['schedule'] == 'cosine':
        decay = 0.5 * (1 + math.cos(math.pi * step / options['steps']))
    else:
        decay = min(1.0, (options['steps'] - step) / options['warmdown_steps'])
    return options['peak_lr'] * warmup * (
        options['min_lr_ratio'] + (1 - options['min_lr_ratio']) * decay
    )


def main():
    process_started = time.perf_counter()
    options = dict(TRAIN_OPTIONS)
    p = argparse.ArgumentParser(description='Train the final model with Muon/AdamW and optional EMA.')
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=None,
                   help='Optional JSON config; default uses student.MODEL_OPTIONS.')
    p.add_argument('--run-dir', type=Path, default=ROOT/options['run_dir'])
    p.add_argument('--device', default=options['device'])
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default=options['precision'])
    p.add_argument('--threads', type=int, default=options['threads'])
    p.add_argument('--seed', type=int, default=options['seed'])
    p.add_argument('--steps', type=int, default=options['steps'])
    p.add_argument('--batch-size', type=int, default=options['batch_size'])
    p.add_argument('--eval-every', type=int, default=options['eval_every'],
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    for key in ('device', 'precision', 'threads', 'seed', 'steps', 'batch_size', 'eval_every'):
        options[key] = getattr(args, key)
    options['run_dir'] = str(args.run_dir)
    for key in ('steps', 'batch_size', 'log_every', 'threads', 'warmdown_steps'):
        if not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if options['eval_every'] < 0:
        p.error('eval-every must be nonnegative.')
    if options['schedule'] not in ('cosine', 'linear_warmdown'):
        raise ValueError('Unknown schedule.')
    if not 0 <= options['min_lr_ratio'] <= 1 or options['warmup_steps'] < 0:
        raise ValueError('Invalid learning-rate schedule settings.')
    for key in ('peak_lr', 'grad_clip'):
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if not math.isfinite(options['weight_decay']) or options['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and nonnegative.')
    if options['use_ema'] and (not 0 <= options['ema_decay'] < 1
            or not isinstance(options['ema_start_step'], int) or options['ema_start_step'] < 1):
        raise ValueError('Invalid EMA decay or start step.')
    if options['use_muon']:
        if not math.isfinite(options['muon_lr']) or options['muon_lr'] <= 0:
            raise ValueError('muon_lr must be finite and positive.')
        if not 0 <= options['muon_momentum'] < 1:
            raise ValueError('muon_momentum must be in [0, 1).')
        if not isinstance(options['muon_ns_steps'], int) or options['muon_ns_steps'] < 1:
            raise ValueError('muon_ns_steps must be a positive integer.')
        if not math.isfinite(options['muon_weight_decay']) or options['muon_weight_decay'] < 0:
            raise ValueError('muon_weight_decay must be finite and nonnegative.')
    run_dir = ROOT / options['run_dir']
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('Use a new run_dir; existing results will not be overwritten.')
    device, precision = setup(options['device'], options['precision'], options['threads'])
    torch.manual_seed(options['seed'])
    prepared = time.perf_counter()
    data = load_data()
    if args.config is not None:
        config = json.loads(args.config.read_text())
    elif args.implementation == 'model':
        config = dict(vocab=2048, width=128, heads=4, depth=4, context=256)
    else:
        config = dict(MODEL_OPTIONS)
    model, implementation_sha = make_model(args.implementation, config, device)
    # model.parameters() deduplicates the tied embedding/output weights.
    parameters = list(model.parameters())
    muon_parameters = []
    if options['use_muon']:
        # Exclude embeddings, the tied output head and learned mixture gates.
        backbone = getattr(model, 'backbone', model)
        protected = {id(backbone.token.weight), id(backbone.head.weight)}
        muon_parameters = [p for name, p in model.named_parameters()
                           if p.ndim == 2 and id(p) not in protected
                           and 'copy_gate' not in name and 'smear_gate' not in name]
        matrix_ids = {id(p) for p in muon_parameters}
        parameters = [p for p in parameters if id(p) not in matrix_ids]
    if options['group_weight_decay']:
        groups = [
            {'params': [p for p in parameters if p.ndim >= 2],
             'weight_decay': options['weight_decay']},
            {'params': [p for p in parameters if p.ndim < 2], 'weight_decay': 0.0},
        ]
    else:
        groups = parameters
    optimizer = torch.optim.AdamW(groups, lr=options['peak_lr'], weight_decay=options['weight_decay'])
    optimizers = [('adamw', optimizer, options['peak_lr'])]
    if options['use_muon'] and muon_parameters:
        optimizers.append(('muon', MatrixMuon(muon_parameters, options), options['muon_lr']))
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(options['seed'])
    run_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha(ROOT / name) for name in (
        'student.py', 'model.py', 'train.py', 'common.py', 'evaluate.py',
    )}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared
    history, validation_history, best = [], [], {}
    final_validation = {}
    ema_model = None
    training_seconds = validation_seconds = checkpoint_seconds = 0.0

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def save(candidate, kind, step, bpb, path):
        nonlocal checkpoint_seconds
        started = time.perf_counter()
        torch.save({
            'protocol': PROTOCOL, 'implementation': args.implementation,
            'config': dict(config), 'model': {k: v.detach().cpu() for k, v in candidate.state_dict().items()},
            'seed': options['seed'], 'step': step,
            'train_tokens': step * options['batch_size'] * config['context'],
            'validation_bpb': bpb, 'weights_kind': kind,
            'training_options': options, 'source_sha256': source_hashes,
        }, path)
        checkpoint_seconds += time.perf_counter() - started

    for step in range(1, options['steps'] + 1):
        sync()
        started = time.perf_counter()
        context = config['context']
        starts = torch.randint(len(tokens) - context - 1, (options['batch_size'],), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(context + 1, device=device)]
        lr = learning_rate(step - 1, options)
        for _, current_optimizer, peak in optimizers:
            for group in current_optimizer.param_groups:
                group['lr'] = peak * lr / options['peak_lr']
            current_optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options['grad_clip'])
        for _, current_optimizer, _ in optimizers:
            current_optimizer.step()
        if options['use_ema'] and step >= options['ema_start_step']:
            if ema_model is None:
                # Deepcopy retains tied weights and the stateless forward hooks.
                ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            else:
                with torch.no_grad():
                    source = dict(model.named_parameters())
                    for name, parameter in ema_model.named_parameters():
                        parameter.lerp_(source[name], 1 - options['ema_decay'])
                    source_buffers = dict(model.named_buffers())
                    for name, buffer in ema_model.named_buffers():
                        buffer.copy_(source_buffers[name])
        sync()
        training_seconds += time.perf_counter() - started
        if step % options['log_every'] == 0 or step == options['steps']:
            row = {'step': step, 'loss': loss.item(), 'lr': lr, 'seconds': training_seconds}
            row['optimizer_lrs'] = {name: opt.param_groups[0]['lr'] for name, opt, _ in optimizers}
            history.append(row)
            print(json.dumps(row), flush=True)
        if (options['eval_every'] > 0 and step % options['eval_every'] == 0) or step == options['steps']:
            candidates = [('raw', model)]
            if ema_model is not None:
                candidates.append(('ema', ema_model))
            for kind, candidate in candidates:
                sync()
                started = time.perf_counter()
                result = score(candidate, *data['validation'], device, 'fp32')
                result.pop('window_nll_nats')
                sync()
                validation_seconds += time.perf_counter() - started
                if not math.isfinite(result['bpb']):
                    raise RuntimeError('Nonfinite validation BPB.')
                validation_history.append({'step': step, 'weights_kind': kind, **result})
                print(json.dumps({'validation': validation_history[-1]}), flush=True)
                if kind not in best or result['bpb'] < best[kind]['bpb']:
                    path = run_dir / f'best_{kind}_checkpoint.pt'
                    save(candidate, kind, step, result['bpb'], path)
                    best[kind] = {'step': step, 'bpb': result['bpb'], 'path': path.name}
                if step == options['steps']:
                    final_validation[kind] = result
                    filename = 'checkpoint.pt' if kind == 'raw' else 'ema_checkpoint.pt'
                    save(candidate, kind, step, result['bpb'], run_dir / filename)

    for item in best.values():
        item['checkpoint_sha256'] = sha(run_dir / item['path'])
    selected_kind = min(best, key=lambda kind: best[kind]['bpb'])
    result = {
        'protocol': PROTOCOL, 'implementation': args.implementation,
        'config': config, 'training_options': options,
        'seed': options['seed'], 'precision': precision, 'threads': options['threads'],
        'parameters': sum(p.numel() for p in model.parameters()),
        'train_tokens': options['steps'] * options['batch_size'] * config['context'],
        'preparation_seconds': preparation_seconds,
        'train_seconds': training_seconds, 'validation_seconds': validation_seconds,
        'validation': final_validation['raw'], 'ema_validation': final_validation.get('ema'),
        'checkpoint_sha256': sha(run_dir / 'checkpoint.pt'),
        'checkpoint_seconds': checkpoint_seconds, 'history': history,
        'validation_history': validation_history, 'best': best,
        'selected_weights_kind': selected_kind, 'selected_checkpoint': best[selected_kind]['path'],
        'source_sha256': source_hashes, 'implementation_sha256': implementation_sha,
        'torch_version': str(torch.__version__), 'process_seconds': time.perf_counter() - process_started,
        **device_metrics(device),
    }
    (run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('history', 'validation_history')}, indent=2))


if __name__ == '__main__':
    main()
