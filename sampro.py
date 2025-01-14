import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

class SAMOptimized(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive_rho=False, max_grad_norm=None, **kwargs):
        """
        SAMOptimized: An optimized version of SAM with dynamic rho, gradient clipping, and learning rate scheduling.

        Args:
            params (iterable): Iterable of parameters to optimize.
            base_optimizer (torch.optim.Optimizer): Base optimizer (e.g., SGD, Adam).
            rho (float): Perturbation radius. Default: 0.05.
            adaptive_rho (bool): Whether to dynamically adjust rho during training. Default: False.
            max_grad_norm (float): Maximum gradient norm for clipping. Default: None (no clipping).
            **kwargs: Additional arguments passed to the base optimizer.
        """
        defaults = dict(rho=rho, adaptive_rho=adaptive_rho, max_grad_norm=max_grad_norm, **kwargs)
        super(SAMOptimized, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive_rho = adaptive_rho
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        if self.max_grad_norm is not None:
            # 从 param_groups 中提取所有参数
            parameters = [p for group in self.param_groups for p in group['params']]
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

        for group in self.param_groups:
            if self.adaptive_rho:
                # Adjust rho based on step count (e.g., decay rho over time)
                group['rho'] = group['rho'] * (0.99 ** self.step_count)

            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue  # 跳过没有梯度的参数
                self.state[p]['old_p'] = p.data.clone()  # Save original parameters
                e_w = p.grad * scale
                p.add_(e_w)  # Update to w + e(w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Perform the second step of SAM: restore the parameters and update.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']  # Restore to original parameters

        self.base_optimizer.step()  # Update parameters using the base optimizer
        if zero_grad:
            self.zero_grad()

        self.step_count += 1

    def _grad_norm(self):
        """
        Compute the norm of the gradients across all parameters.
        """
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
        """
        Get the current learning rate.
        """
        for group in self.param_groups:
            return group['lr']

    def set_lr(self, lr):
        """
        Set the learning rate.
        """
        for group in self.param_groups:
            group['lr'] = lr

# Example usage with learning rate scheduler
if __name__ == "__main__":
    # Define a simple model
    model = torch.nn.Linear(10, 1)

    # Initialize SAMOptimized with SGD as the base optimizer
    optimizer = SAMOptimized(model.parameters(), base_optimizer=optim.SGD, lr=0.1, rho=0.05, adaptive_rho=True)

    # Add a learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=100)

    # Training loop
    for epoch in range(100):
        # Forward pass
        optimizer.zero_grad()
        loss = model(torch.randn(32, 10)).mean()
        loss.backward()

        # SAM steps
        optimizer.first_step(zero_grad=True)
        loss = model(torch.randn(32, 10)).mean()
        loss.backward()
        optimizer.second_step(zero_grad=True)

        # Update learning rate
        scheduler.step()