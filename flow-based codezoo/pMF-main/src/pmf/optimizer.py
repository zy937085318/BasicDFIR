import torch
import torch.optim as optim

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix G.
    Approximates orthogonalization.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, adam_w_params=None, adam_w_kwargs=None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        # TODO: Implement closure support if needed (Muon typically doesn't use it, but good for completeness) [PRIORITY: LOW]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # TODO: Add parameter validation (e.g., ensure lr > 0) [PRIORITY: LOW]
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.ndim != 2:
                    # Warning if non-2D param passed to Muon (which expects 2D)
                    # We skip it here, but it should ideally be handled by the optimizer configuration.
                    # Raising an error might be too strict if user accidentally includes it,
                    # but Muon logic (Newton-Schulz) explicitly fails on non-2D.
                    # The assertion inside zeropower_via_newtonschulz5 will catch it if we proceed,
                    # but better to skip or warn.
                    import warnings
                    warnings.warn(f"Parameter with shape {grad.shape} passed to Muon optimizer. Muon only supports 2D parameters. Skipping.")
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    g = grad + momentum * buf
                else:
                    g = buf

                # Orthogonalize
                update = zeropower_via_newtonschulz5(g, steps=ns_steps)

                # Scale update (learning rate scaling described in blog)
                # "The learning rate should have built-in muP scaling"
                # "gamma <- 0.2 * gamma * sqrt(max(A, B))"
                A, B = p.size()
                scale = 0.2 * lr * max(A, B)**0.5

                p.data.add_(update, alpha=-scale)

        return loss

def configure_optimizers(model, config):
    """
    Splits parameters into hidden (2D) for Muon and others for AdamW.
    """
    hidden_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Heuristic: 2D weights in blocks/layers are hidden.
        # Biases, 1D, embeddings, or final layer might be different.
        # Usually Muon is used for the bulk of transformer weights.
        if p.ndim == 2 and "embed" not in name and "head" not in name:
             hidden_params.append(p)
        else:
             other_params.append(p)

    optim_groups = [
        {'params': hidden_params, 'lr': config.muon_lr, 'momentum': config.muon_momentum, 'optimizer_name': 'muon'},
        {'params': other_params, 'lr': config.adam_lr, 'weight_decay': config.weight_decay, 'optimizer_name': 'adamw'}
    ]

    # We return a custom wrapper or list.
    # Since Muon implementation above is a standalone optimizer, we can't easily mix it in one "Optimizer" object
    # unless we use a wrapper that calls step() on both.

    optimizer1 = Muon(hidden_params, lr=config.muon_lr, momentum=config.muon_momentum)
    optimizer2 = optim.AdamW(other_params, lr=config.adam_lr, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2))

    return [optimizer1, optimizer2]

class CombinedOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, state in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state)
