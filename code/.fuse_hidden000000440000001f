"""
model_definition.py — DOMCS-EEG Dual-Space Disentanglement Model
=================================================================
DROP-IN REPLACEMENT for the RESULTS_PIPELINE version.

Architecture matches the locked checkpoints exactly (234,880 params):
  INPUT (B, 64, 256)
    └─ CNN encoder [Conv k=7/5/3 | BN | ELU | AdaptiveAvgPool] → f ∈ R^256
         ├─ identity_branch [Linear(256→128) → LayerNorm → L2-norm] → z_id
         └─ state_branch [f.detach() → Linear(256→128) → LayerNorm → L2-norm] → z_state

Training heads (used during eval pipeline for metric computation only):
  ArcFaceHead:      operates on z_id  (not saved in checkpoint, not needed for EER)
  StateClassifier:  operates on z_state (not saved in checkpoint)

Checkpoint keys loaded: encoder.*, id_branch.*, state_branch.*
Expected param count: 234,880 (backbone only, no heads)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Constants (mirror paths.yaml model section) ──────────────────────────────
# These are overridden by load_model() when cfg is passed.
_DEFAULT_N_SUBJECTS  = 109
_DEFAULT_N_STATES    = 2      # binary rest/task (matches locked training)
_DEFAULT_ID_DIM      = 128
_DEFAULT_STATE_DIM   = 128
_DEFAULT_ENC_OUT     = 256
_DEFAULT_ARC_S       = 32.0
_DEFAULT_ARC_M       = 0.50


# ─── Building blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ELU"""
    def __init__(self, in_ch, out_ch, kernel, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EEGEncoder(nn.Module):
    """
    3-layer 1D CNN.
      Conv1: 64→64,   k=7, pad=3
      Conv2: 64→128,  k=5, pad=2
      Conv3: 128→256, k=3, pad=1
    → AdaptiveAvgPool1d(1) → f ∈ R^{256}
    Works for any temporal length (256 or 128 samples).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(64,  64,  kernel=7, padding=3)
        self.conv2 = ConvBlock(64,  128, kernel=5, padding=2)
        self.conv3 = ConvBlock(128, 256, kernel=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, 64, T)
        x = self.conv1(x)     # (B, 64,  T)
        x = self.conv2(x)     # (B, 128, T)
        x = self.conv3(x)     # (B, 256, T)
        x = self.pool(x)      # (B, 256, 1)
        return x.squeeze(-1)  # (B, 256)


class IdentityBranch(nn.Module):
    """Linear(256→128) → LayerNorm → L2-norm → z_id. Full grad to encoder."""
    def __init__(self, enc_dim=256, id_dim=128):
        super().__init__()
        self.fc   = nn.Linear(enc_dim, id_dim, bias=False)
        self.norm = nn.LayerNorm(id_dim)

    def forward(self, f):
        z = self.norm(self.fc(f))
        return F.normalize(z, p=2, dim=1)


class StateBranch(nn.Module):
    """
    f.detach() → Linear(256→128) → LayerNorm → L2-norm → z_state.
    detach() is CRITICAL: state losses must NOT back-propagate into encoder.
    """
    def __init__(self, enc_dim=256, state_dim=128):
        super().__init__()
        self.fc   = nn.Linear(enc_dim, state_dim, bias=False)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, f):
        z = self.norm(self.fc(f.detach()))   # detach from encoder
        return F.normalize(z, p=2, dim=1)


# ─── Training heads (not in checkpoint; used only if pipeline needs them) ─────

class ArcFaceHead(nn.Module):
    def __init__(self, in_dim=128, n_classes=109, s=32.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z_id, labels):
        W      = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(z_id, W).clamp(-1 + 1e-7, 1 - 1e-7)
        theta  = torch.acos(cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = cosine * (1 - one_hot) + torch.cos(theta + self.m) * one_hot
        return logits * self.s


class StateClassifier(nn.Module):
    def __init__(self, in_dim=128, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z_state):
        return self.fc(z_state)


# ─── Full model ───────────────────────────────────────────────────────────────

class DOMCSEEGModel(nn.Module):
    """
    DOMCS-EEG backbone (234,880 parameters).
    This is the model saved in the locked checkpoints.
    forward() → (z_id, z_state, f)
    """
    def __init__(self):
        super().__init__()
        self.encoder      = EEGEncoder()
        self.id_branch    = IdentityBranch()
        self.state_branch = StateBranch()

    def forward(self, x):
        f       = self.encoder(x)       # (B, 256)
        z_id    = self.id_branch(f)     # (B, 128)
        z_state = self.state_branch(f)  # (B, 128)
        return z_id, z_state, f

    def get_identity_embedding(self, x):
        """Inference-only: z_id without state branch."""
        with torch.no_grad():
            f    = self.encoder(x)
            z_id = self.id_branch(f)
        return z_id


# ─── Pipeline-facing factory (matches original API expected by scripts) ────────

def build_model(cfg=None):
    """Return DOMCSEEGModel. cfg argument accepted but ignored (arch is locked)."""
    return DOMCSEEGModel()


def load_model(checkpoint_path, device='cpu', cfg=None):
    """
    Load DOMCSEEGModel from checkpoint.
    Accepts both 'model_best.pt' and 'best_model.pt' naming.
    Raises ValueError if param count != 234,880 (architecture guard).
    """
    model = DOMCSEEGModel()

    # ── Param count guard ──────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != 234_880:
        raise ValueError(
            f"Architecture mismatch: expected 234,880 params, got {n_params:,}. "
            f"Do NOT modify model_definition.py."
        )

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle various checkpoint formats
    if isinstance(ckpt, dict):
        state_dict = (ckpt.get('model_state_dict')
                      or ckpt.get('state_dict')
                      or ckpt.get('model')
                      or ckpt)
    else:
        state_dict = ckpt

    # Filter to backbone keys only (drop heads if present)
    backbone_keys = {'encoder', 'id_branch', 'state_branch'}
    filtered = {k: v for k, v in state_dict.items()
                if k.split('.')[0] in backbone_keys}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing}")

    model.to(device)
    model.eval()
    return model


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = DOMCSEEGModel()
    total, trainable = count_parameters(model)
    print(f"DOMCSEEGModel  total={total:,}  trainable={trainable:,}")
    assert total == 234_880, f"Param count wrong: {total}"

    x = torch.randn(4, 64, 256)
    z_id, z_state, f = model(x)
    assert f.shape      == (4, 256)
    assert z_id.shape   == (4, 128)
    assert z_state.shape == (4, 128)
    assert torch.allclose(z_id.norm(dim=1),    torch.ones(4), atol=1e-5)
    assert torch.allclose(z_state.norm(dim=1), torch.ones(4), atol=1e-5)

    # Verify state branch does NOT leak gradient into encoder
    z_state.sum().backward()
    enc_grad = model.encoder.conv1.net[0].weight.grad
    assert enc_grad is None or enc_grad.abs().max() < 1e-9
    print("All assertions passed. 234,880 params confirmed.")
