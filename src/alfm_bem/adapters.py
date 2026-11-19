"""Residual PyTorch adapters trained on replayed outcome-labeled embeddings.

Plain SGD clips the joint gradient and projects the live parameters onto a
Euclidean ball at initialization and after each step. For fixed radius and
learning rate, the per-step parameter displacement is at most learning rate
times clipping threshold, up to floating-point error.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
from dataclasses import asdict, dataclass
from math import isfinite
from pickle import UnpicklingError

from .bem import Experience, BidirectionalExperienceMemory


@dataclass
class AdapterConfig:
    """Configuration for bounded adapter."""
    input_dim: int = 256          # From projection layer
    hidden_dim: int = 64          # Small for efficiency
    output_dim: int = 256         # Back to embedding space

    # Joint parameter and gradient norm limits
    max_grad_norm: float = 1.0    # Gradient clipping threshold
    max_param_norm: float = 10.0  # Maximum parameter norm
    learning_rate: float = 1e-4

    # Regularization
    l2_weight: float = 0.01       # Coefficient of the sum of parameter tensor norms
    ema_decay: float = 0.999      # EMA for stable updates


class BoundedAdapter(nn.Module):
    """
    A small MLP adapter with bounded updates, implemented in PyTorch.

    Add the two-layer ReLU correction to the input, then normalize the result.
    """

    def __init__(self, config: AdapterConfig, tenant_id: str = "global"):
        super().__init__()
        if config.input_dim != config.output_dim:
            raise ValueError("Residual adapters require matching input and output dimensions")
        for name in ("learning_rate", "max_grad_norm", "max_param_norm", "l2_weight"):
            value = getattr(config, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= config.ema_decay <= 1:
            raise ValueError("ema_decay must lie in [0, 1]")
        self.config = config
        self.tenant_id = tenant_id

        # Architecture
        self.norm = nn.LayerNorm(config.input_dim)
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)

        # Initialize weights for near-identity start
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

        # Step zero must be feasible before recording either the baseline or the EMA.
        self._project(self.parameters(), config.max_param_norm)
        self.optimizer = optim.SGD(self.parameters(), lr=config.learning_rate)
        self.beta = config.ema_decay
        for name, param in self.named_parameters():
            self.register_buffer(self._shadow_name(name), param.detach().clone(), persistent=False)
        self._reset_tracking()
        self.register_load_state_dict_post_hook(self._on_state_loaded)

    @staticmethod
    def _shadow_name(name):
        return "_ema_" + name.replace(".", "_")

    @property
    def shadow(self):
        return {name: getattr(self, self._shadow_name(name)) for name, _ in self.named_parameters()}

    @staticmethod
    def _vector_norm(vector):
        # Accumulate in double precision even for long float32 parameter vectors.
        if vector.device.type == "mps":
            vector = vector.cpu()
        return torch.linalg.vector_norm(vector, dtype=torch.float64).item()

    @staticmethod
    @torch.no_grad()
    def _project(parameters, radius):
        parameters = list(parameters)
        norm = BoundedAdapter._vector_norm(torch.nn.utils.parameters_to_vector(parameters))
        if not isfinite(norm):
            raise ValueError("Adapter parameters must be finite")
        if norm > radius:
            # Round toward the interior so reloading a feasible checkpoint is idempotent.
            scale = radius / norm * (1.0 - torch.finfo(parameters[0].dtype).eps)
            for param in parameters:
                param.mul_(scale)

    def _reset_tracking(self):
        self.update_count = 0
        self.total_drift = 0.0
        self._initial_param_norm = self._compute_param_norm()

    @torch.no_grad()
    def _on_state_loaded(self, module, incompatible_keys):
        self._project(self.parameters(), self.config.max_param_norm)
        for name, param in self.named_parameters():
            setattr(self, self._shadow_name(name), param.detach().clone())
        # assign=True can replace parameter objects. Rebind plain SGD to the loaded tensors.
        self.optimizer = optim.SGD(self.parameters(), lr=self.config.learning_rate)
        self._reset_tracking()

    @torch.no_grad()
    def _compute_param_norm(self) -> float:
        return self._vector_norm(torch.nn.utils.parameters_to_vector(self.parameters()))

    def forward(self, x, use_ema: bool = False):
        """Return an array for NumPy inference or a differentiable tensor for Torch."""
        if isinstance(x, np.ndarray):
            with torch.no_grad():
                tensor = torch.as_tensor(x, device=self.fc1.weight.device, dtype=self.fc1.weight.dtype)
                return self.forward(tensor, use_ema=use_ema).cpu().numpy()
        if use_ema:
            weights = self.shadow
            normed = nn.functional.layer_norm(
                x, self.norm.normalized_shape, weights["norm.weight"], weights["norm.bias"], self.norm.eps
            )
            hidden = nn.functional.linear(normed, weights["fc1.weight"], weights["fc1.bias"])
            delta = nn.functional.linear(self.relu(hidden), weights["fc2.weight"], weights["fc2.bias"])
            out = x + delta
            return out / (out.norm(dim=-1, keepdim=True) + 1e-10)
        residual = x
        out = self.norm(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + residual

        # Normalize output (embedding space)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-10)
        return out

    def train_step(
        self,
        experiences: List[Experience],
        projection_fn=None # Ignored, we consume embeddings directly
    ) -> float:
        """
        One training step using experience replay.
        """
        if not experiences:
            return 0.0

        self.train()
        self.optimizer.zero_grad()

        # Prepare batch
        embeddings = np.stack([e.embedding for e in experiences])
        outcomes = np.array([e.outcome for e in experiences])

        z = self.fc1.weight.new_tensor(embeddings)
        y = self.fc1.weight.new_tensor(outcomes)

        adapted = self.forward(z)
        target = z / (z.norm(dim=-1, keepdim=True) + 1e-10)
        similarity = torch.sum(adapted * target, dim=-1)
        base_loss = torch.where(y > 0, 1 - similarity, (similarity + 0.5).clamp_min(0)).mean()

        # Clip the gradient of the entire objective before the plain SGD step.
        penalty = sum(param.norm() for param in self.parameters())
        total_loss = base_loss + self.config.l2_weight * penalty
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
        )

        with torch.no_grad():
            old_vec = torch.nn.utils.parameters_to_vector(self.parameters())
            self.optimizer.step()
            self._project(self.parameters(), self.config.max_param_norm)
            new_vec = torch.nn.utils.parameters_to_vector(self.parameters())
            self.total_drift += self._vector_norm(new_vec - old_vec)
            for name, shadow in self.shadow.items():
                shadow.lerp_(self.get_parameter(name), 1.0 - self.beta)

        self.update_count += 1

        return base_loss.item()

    def get_statistics(self) -> Dict[str, Any]:
        """Return adapter statistics."""
        return {
            "tenant_id": self.tenant_id,
            "update_count": self.update_count,
            "total_drift": self.total_drift,
            "current_param_norm": self._compute_param_norm(),
            "initial_param_norm": self._initial_param_norm,
            "relative_drift": self.total_drift / (self._initial_param_norm + 1e-10)
        }

    def save(self, path: str):
        """Save live parameters, moving average, configuration and drift history."""
        torch.save({
            'state_dict': self.state_dict(),
            'shadow': self.shadow,
            'stats': self.get_statistics(),
            'config': asdict(self.config),
            'format_version': 2
        }, path)

    @classmethod
    def load(cls, path: str, config: AdapterConfig) -> 'BoundedAdapter':
        """Restore weights in the configured ball. Legacy imports start a new drift interval."""
        try:
            ckpt = torch.load(path, weights_only=True, map_location="cpu")
        except UnpicklingError:
            # Legacy checkpoints contain the config dataclass and NumPy scalar stats.
            if hasattr(path, "seek"):
                path.seek(0)
            scalar_reducer = np.float64(0).__reduce__()[0]
            legacy_types = [AdapterConfig, np.dtype, type(np.dtype("float64")), scalar_reducer]
            if scalar_reducer.__module__ != "numpy.core.multiarray":
                legacy_types.append((scalar_reducer, "numpy.core.multiarray.scalar"))
            with torch.serialization.safe_globals(legacy_types):
                ckpt = torch.load(path, weights_only=True, map_location="cpu")
        stats = ckpt["stats"]
        adapter = cls(config, tenant_id=stats["tenant_id"])
        adapter.load_state_dict(ckpt["state_dict"])
        with torch.no_grad():
            shadows = adapter.shadow
            for name, shadow in shadows.items():
                shadow.copy_(ckpt["shadow"][name])
            adapter._project(shadows.values(), config.max_param_norm)
        # Legacy history can include a first step from outside the ball. Start
        # a new measurement interval there, or when resuming with new settings.
        if ckpt.get("format_version") == 2 and ckpt["config"] == asdict(config):
            adapter.update_count = stats["update_count"]
            adapter.total_drift = stats["total_drift"]
            adapter._initial_param_norm = stats["initial_param_norm"]
        return adapter


class AdapterManager:
    """
    Manages adapters across tiers: global, domain, tenant.
    """

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.global_adapter = BoundedAdapter(config, tenant_id="global")
        self.domain_adapters: Dict[str, BoundedAdapter] = {}
        self.tenant_adapters: Dict[str, BoundedAdapter] = {}

    def get_adapter_chain(
        self,
        tenant_id: str,
        domain_id: str = "default"
    ) -> List[BoundedAdapter]:
        """Get chain of adapters for a tenant."""
        chain = [self.global_adapter]

        if domain_id in self.domain_adapters:
            chain.append(self.domain_adapters[domain_id])

        if tenant_id in self.tenant_adapters:
            chain.append(self.tenant_adapters[tenant_id])

        return chain

    def forward(
        self,
        z: np.ndarray,
        tenant_id: str,
        domain_id: str = "default",
        use_ema: bool = True
    ) -> np.ndarray:
        """Apply all adapters in chain."""
        chain = self.get_adapter_chain(tenant_id, domain_id)

        for adapter in chain:
            z = adapter(z, use_ema=use_ema)

        return z

    def train_tenant_adapter(
        self,
        tenant_id: str,
        bem: BidirectionalExperienceMemory,
        projection_fn,
        batch_size: int = 32
    ) -> float:
        """Train a tenant's adapter using their BEM experiences."""
        if tenant_id not in self.tenant_adapters:
            self.tenant_adapters[tenant_id] = BoundedAdapter(
                self.config, tenant_id=tenant_id
            )

        experiences = bem.sample_for_training(batch_size)
        loss = self.tenant_adapters[tenant_id].train_step(experiences)

        return loss
