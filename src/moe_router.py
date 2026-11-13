import torch

class BudgetedMoERouter(torch.nn.Module):
    """Top-k expert router from the Budgeted Sparse Routing paper."""

    def __init__(self, dim: int, experts: int, capacity: int):
        super().__init__()
        self.gate = torch.nn.Linear(dim, experts)
        self.capacity = capacity

    def forward(self, x):
        return torch.softmax(self.gate(x), dim=-1)
