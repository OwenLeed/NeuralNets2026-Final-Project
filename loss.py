import torch
import torch.nn as nn
from typing import List


class PinballLoss(nn.Module):

    def __init__(self, quantiles: List[float]):
        super().__init__()

        assert quantiles == sorted(quantiles), \
            "Quantiles must be in ascending order"
        assert all(0 < q < 1 for q in quantiles), \
            "All quantiles must be strictly between 0 and 1"

        self.quantiles = quantiles

        self.register_buffer(
            "quantile_tensor",
            torch.tensor(quantiles, dtype=torch.float32)
        )

    def forward(
        self,
        predictions : torch.Tensor,
        targets     : torch.Tensor,
    ) -> torch.Tensor:
        targets_expanded = targets.expand_as(predictions)
        errors = targets_expanded - predictions

        loss = torch.where(
            errors >= 0,
            self.quantile_tensor * errors,
            (self.quantile_tensor - 1.0) * errors,
        )
        return loss.mean()

    def per_quantile_loss(
        self,
        predictions : torch.Tensor,
        targets     : torch.Tensor,
    ) -> torch.Tensor:
        targets_expanded = targets.expand_as(predictions)
        errors = targets_expanded - predictions

        loss = torch.where(
            errors >= 0,
            self.quantile_tensor * errors,
            (self.quantile_tensor - 1.0) * errors,
        )
        return loss.mean(dim=0)

    def coverage(
        self,
        predictions : torch.Tensor,
        targets     : torch.Tensor,
    ) -> dict:
        targets_flat = targets.squeeze(1)
        n_q = len(self.quantiles)
        results = {}

        for i in range(n_q // 2):
            lower_idx = i
            upper_idx = n_q - 1 - i
            lower_tau = self.quantiles[lower_idx]
            upper_tau = self.quantiles[upper_idx]

            lower_preds = predictions[:, lower_idx]
            upper_preds = predictions[:, upper_idx]

            in_interval = (
                (targets_flat >= lower_preds) &
                (targets_flat <= upper_preds)
            )
            empirical_coverage = in_interval.float().mean().item()
            target_coverage = upper_tau - lower_tau

            label = f"{round(target_coverage * 100)}%"
            results[label] = {
                "target"   : target_coverage,
                "empirical": empirical_coverage,
                "gap"      : empirical_coverage - target_coverage,
            }

        return results