# pro_sam.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

class SAMHessian(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive_rho=False, max_grad_norm=None, hessian_lambda=0.01, **kwargs):
        """
        SAMHessian: An optimized version of SAM with Hessian information.

        Args:
            params (iterable): Iterable of parameters to optimize.
            base_optimizer (torch.optim.Optimizer): Base optimizer (e.g., SGD, Adam).
            rho (float): Perturbation radius. Default: 0.05.
            adaptive_rho (bool): Whether to dynamically adjust rho during training. Default: False.
            max_grad_norm (float): Maximum gradient norm for clipping. Default: None (no clipping).
            hessian_lambda (float): Weight for Hessian information. Default: 0.01.
            **kwargs: Additional arguments passed to the base optimizer.
        """
        defaults = dict(rho=rho, adaptive_rho=adaptive_rho, max_grad_norm=max_grad_norm, hessian_lambda=hessian_lambda, **kwargs)
        super(SAMHessian, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive_rho = adaptive_rho
        self.max_grad_norm = max_grad_norm
        self.hessian_lambda = hessian_lambda
        self.step_count = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        if self.max_grad_norm is not None:
            # Gradient clipping
            parameters = [p for group in self.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

        for group in self.param_groups:
            if self.adaptive_rho:
                # Adjust rho based on step count
                group['rho'] = group['rho'] * (0.99 ** self.step_count)

            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = p.grad * scale
                p.add_(e_w)  # Update to w + e(w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']  # Restore to original parameters

        # Update parameters using the base optimizer
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

        self.step_count += 1

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def get_lr(self):
        for group in self.param_groups:
            return group['lr']

    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr