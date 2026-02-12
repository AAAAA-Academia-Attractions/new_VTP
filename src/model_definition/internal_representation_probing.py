import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InternalRepresentationProbingWnorm(nn.Module):
    def __init__(self, input_dim: int = 2048, output_dim: int = 2048):
        super().__init__()
        self.p = nn.Parameter(torch.ones(input_dim))
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # 1. Normalize Input (Good practice)
        # Ensures 'h' magnitude doesn't bias the probe
        h_norm = F.normalize(h, p=2, dim=-1)

        h_weighted = h_norm * self.p

        # 3. Apply Projection with FORCED Unit-Norm (Rotation Only)
        # We normalize the weight matrix rows to length 1 on every pass.
        # This prevents the projection from "scaling up" to hide the shrinking 'p'.
        proj_weight_normalized = F.normalize(self.projection.weight, p=2, dim=1)
        
        # Use functional linear to apply the normalized weights
        z = F.linear(h_weighted, proj_weight_normalized)

        # 4. CRITICAL: Do NOT normalize 'z' here.
        # We want the magnitude of 'z' to depend on 'p'.
        # If 'p' selects useless neurons, 'z' shrinks -> Dot Product drops -> Loss increases.
        # This forces 'p' to stay large only for useful neurons.
        return z
    
    def get_logit_scale(self) -> torch.Tensor:
        """
        Get the logit scale parameter (clamped for numerical stability).
        
        Returns:
            Clamped logit scale tensor
        """
        # Clamp logit scale to prevent numerical issues
        # Common practice: clamp between log(1/100) and log(100)
        return self.logit_scale.clamp(max=np.log(100))

class InternalRepresentationProbing(nn.Module):
    def __init__(self, input_dim: int = 2048):
        super().__init__()
        self.p = nn.Parameter(torch.ones(input_dim))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = F.normalize(h, p=2, dim=-1)

        h_weighted = h_norm * self.p

        z = F.normalize(h_weighted, p=2, dim=-1)
        
        return z
    
    def get_logit_scale(self) -> torch.Tensor:
        """
        Get the logit scale parameter (clamped for numerical stability).
        
        Returns:
            Clamped logit scale tensor
        """
        # Clamp logit scale to prevent numerical issues
        # Common practice: clamp between log(1/100) and log(100)
        return self.logit_scale.clamp(max=np.log(100))