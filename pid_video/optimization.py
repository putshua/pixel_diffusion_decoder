"""Explicit video batch accounting and disjoint learning-rate groups."""
import torch


def batch_plan(cfg, world):
    micro = int(cfg.get('micro_batch', 1))
    total = int(cfg.get('global_batch', world))
    if micro < 1 or total < 1 or total % (world * micro):
        raise ValueError('global_batch must be divisible by world_size * micro_batch')
    return micro, total // (world * micro)


def regroup_optimizer(model, optimizer, rates):
    """Preserve Adam moments while replacing the old single parameter group."""
    from pid_video.model import TemporalBlock
    root = model.module if hasattr(model, 'module') else model
    head = {id(p) for p in root.net.final_layer.linear.parameters()}
    adaptation = {id(p) for p in root.net.lq_proj.parameters()} | {id(root.null_text)}
    for module in root.modules():
        if isinstance(module, TemporalBlock):
            for name in ('norm', 'down', 'conv', 'up'):
                adaptation.update(id(p) for p in getattr(module, name).parameters())
    assert not head & adaptation
    groups = {key: [] for key in ('backbone', 'adapter_temporal', 'head')}
    for p in model.parameters():
        if p.requires_grad:
            groups['head' if id(p) in head else 'adapter_temporal' if id(p) in adaptation else 'backbone'].append(p)
    original = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert original == {id(p) for ps in groups.values() for p in ps}
    template = {k: v for k, v in optimizer.param_groups[0].items() if k not in ('params', 'name', 'base_lr')}
    optimizer.param_groups = [dict(template, params=ps, name=name, lr=float(rates[name]),
                                   base_lr=float(rates[name])) for name, ps in groups.items()]


def set_learning_rates(optimizer, cfg, step):
    scale = min(1., (step + 1) / max(1, cfg['warmup']))
    for group in optimizer.param_groups:
        group['lr'] = float(group.get('base_lr', cfg['lr'])) * scale


def next_batch(pool, cfg, encode, vae, micro):
    # Encode clips individually: temporal VAE peak does not grow with PiD batch.
    xs, zs, records = [], [], []
    for _ in range(micro):
        x, record = pool.next(cfg['bucket'])
        x = x.cuda()
        zs.append(encode(vae, x)); xs.append(x); records.append(record)
    return torch.cat(xs), torch.cat(zs), records
