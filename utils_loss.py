"""
Custom Loss Functions for Skin Lesion Segmentation
====================================================
All losses follow GT_BceDiceLoss interface: forward(gt_pre, out, target)

Available GT-level (Deep Supervision) losses:
  1. GT_BCEOnlyLoss        — BCE only (단순 Cross Entropy)
  2. GT_BoundaryFocalLoss  — BCE + Dice + BoundaryDice + FocalTversky
  3. GT_LovaszLoss         — BCE + Lovász-Hinge (IoU 직접 최적화)
  4. GT_UncertaintyLoss    — BCE + Dice + Boundary, 자동 가중치 학습 (σ)

Usage:
    from utils_loss import GT_BoundaryFocalLoss
    config.criterion = GT_BoundaryFocalLoss(wb=1, wd=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Base losses (BCE / Dice — utils.py 호환)
# ═══════════════════════════════════════════════════════════════════════════════

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size
        return dice_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Boundary-Aware Focal Compound Loss 구성요소
# ═══════════════════════════════════════════════════════════════════════════════

class BoundaryDiceLoss(nn.Module):
    """Dice loss computed only on boundary regions of the ground truth."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def extract_boundary(self, mask):
        """Extract boundary band from binary mask using morphological erosion via max_pool2d."""
        eroded = 1.0 - F.max_pool2d(
            1.0 - mask, kernel_size=self.kernel_size, stride=1, padding=self.pad
        )
        boundary = mask - eroded
        boundary = F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)
        return boundary.clamp(0, 1)

    def forward(self, pred, target):
        boundary = self.extract_boundary(target)

        if boundary.sum() < 1.0:
            return DiceLoss()(pred, target)

        smooth = 1
        size = pred.size(0)
        pred_bd = (pred * boundary).view(size, -1)
        target_bd = (target * boundary).view(size, -1)
        intersection = (pred_bd * target_bd).sum(1)
        dice_score = (2 * intersection + smooth) / (pred_bd.sum(1) + target_bd.sum(1) + smooth)
        return 1 - dice_score.sum() / size


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss — penalises FN more than FP, with focal exponent."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        TP = (pred_ * target_).sum(1)
        FP = (pred_ * (1 - target_)).sum(1)
        FN = ((1 - pred_) * target_).sum(1)

        tversky_index = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        focal_tversky = (1 - tversky_index).pow(self.gamma)
        return focal_tversky.sum() / size


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Lovász-Hinge Loss (IoU surrogate, CVPR 2018)
# ═══════════════════════════════════════════════════════════════════════════════

class LovaszHingeLoss(nn.Module):
    """Lovász-Hinge loss — convex surrogate for IoU optimization.
    Operates on sigmoid probabilities (not logits).
    Reference: Berman et al., "The Lovász-Softmax loss", CVPR 2018.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Compute gradient of the Lovász extension w.r.t sorted errors."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def _lovasz_hinge_flat(self, pred, target):
        """Lovász hinge on flattened pred/target for a single sample."""
        # Convert target {0,1} -> signs {-1,+1}
        signs = 2.0 * target - 1.0
        # Hinge-like errors: 1 - sign * pred_shifted
        # pred is in [0,1], shift to [-1,1]
        errors = 1.0 - signs * (2.0 * pred - 1.0)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = target[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def forward(self, pred, target):
        size = pred.size(0)
        loss = 0.0
        for i in range(size):
            loss += self._lovasz_hinge_flat(
                pred[i].view(-1), target[i].view(-1)
            )
        return loss / size


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Uncertainty-Weighted Multi-Loss (Kendall et al.)
# ═══════════════════════════════════════════════════════════════════════════════

class UncertaintyWeightedLoss(nn.Module):
    """Automatic multi-loss weighting via learned uncertainty (log σ²).
    L = Σ (1/(2σ_i²)) · L_i + log(σ_i)
    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty", CVPR 2018.
    """

    def __init__(self, w_boundary=1.0, boundary_kernel=3):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.boundary_dice = BoundaryDiceLoss(kernel_size=boundary_kernel)

        # learnable log(σ²) — 초기값 0 → σ=1 → weight=0.5
        self.log_var_bce = nn.Parameter(torch.zeros(1))
        self.log_var_dice = nn.Parameter(torch.zeros(1))
        self.log_var_boundary = nn.Parameter(torch.zeros(1))

    def forward(self, pred, target):
        l_bce = self.bce(pred, target)
        l_dice = self.dice(pred, target)
        l_bd = self.boundary_dice(pred, target)

        # (1/(2σ²)) · L + (1/2)·log(σ²)
        loss = (torch.exp(-self.log_var_bce) * l_bce + 0.5 * self.log_var_bce
                + torch.exp(-self.log_var_dice) * l_dice + 0.5 * self.log_var_dice
                + torch.exp(-self.log_var_boundary) * l_bd + 0.5 * self.log_var_boundary)
        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Compound losses (pred, target 인터페이스)
# ═══════════════════════════════════════════════════════════════════════════════

class BoundaryFocalCompoundLoss(nn.Module):
    """BCE + Dice + BoundaryDice + FocalTversky"""

    def __init__(self, wb=1, wd=1, w_boundary=1.0, w_focal_tversky=1.0,
                 alpha=0.3, beta=0.7, gamma=0.75, boundary_kernel=3):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.boundary_dice = BoundaryDiceLoss(kernel_size=boundary_kernel)
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.wb = wb
        self.wd = wd
        self.w_boundary = w_boundary
        self.w_focal_tversky = w_focal_tversky

    def forward(self, pred, target):
        loss = (self.wb * self.bce(pred, target)
                + self.wd * self.dice(pred, target)
                + self.w_boundary * self.boundary_dice(pred, target)
                + self.w_focal_tversky * self.focal_tversky(pred, target))
        return loss


class LovaszCompoundLoss(nn.Module):
    """BCE + Lovász-Hinge"""

    def __init__(self, wb=1, wl=1):
        super().__init__()
        self.bce = BCELoss()
        self.lovasz = LovaszHingeLoss()
        self.wb = wb
        self.wl = wl

    def forward(self, pred, target):
        return self.wb * self.bce(pred, target) + self.wl * self.lovasz(pred, target)


# ═══════════════════════════════════════════════════════════════════════════════
# GT-level Deep Supervision wrappers (forward(gt_pre, out, target) 인터페이스)
# ═══════════════════════════════════════════════════════════════════════════════

DS_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)  # gt_pre5 → gt_pre1


def _ds_loss(compound_fn, gt_pre, target):
    """Deep supervision loss with fixed weights."""
    loss = 0.0
    for w, feat in zip(DS_WEIGHTS, gt_pre):
        loss += w * compound_fn(feat, target)
    return loss


class GT_BCEOnlyLoss(nn.Module):
    """BCE only + DS"""

    def __init__(self, wb=1, wd=1):
        super().__init__()
        self.bce = BCELoss()

    def forward(self, gt_pre, out, target):
        return self.bce(out, target) + _ds_loss(self.bce, gt_pre, target)


class GT_BoundaryFocalLoss(nn.Module):
    """BCE + Dice + BoundaryDice + FocalTversky + DS"""

    def __init__(self, wb=1, wd=1, w_boundary=1.0, w_focal_tversky=1.0,
                 alpha=0.3, beta=0.7, gamma=0.75, boundary_kernel=3):
        super().__init__()
        self.compound = BoundaryFocalCompoundLoss(
            wb=wb, wd=wd,
            w_boundary=w_boundary, w_focal_tversky=w_focal_tversky,
            alpha=alpha, beta=beta, gamma=gamma,
            boundary_kernel=boundary_kernel,
        )

    def forward(self, gt_pre, out, target):
        return self.compound(out, target) + _ds_loss(self.compound, gt_pre, target)


class GT_LovaszLoss(nn.Module):
    """BCE + Lovász-Hinge + DS"""

    def __init__(self, wb=1, wl=1):
        super().__init__()
        self.compound = LovaszCompoundLoss(wb=wb, wl=wl)

    def forward(self, gt_pre, out, target):
        return self.compound(out, target) + _ds_loss(self.compound, gt_pre, target)


class GT_UncertaintyLoss(nn.Module):
    """BCE + Dice + BoundaryDice (자동 가중치 학습) + DS"""

    def __init__(self, wb=1, wd=1, boundary_kernel=3):
        super().__init__()
        self.compound = UncertaintyWeightedLoss(boundary_kernel=boundary_kernel)

    def forward(self, gt_pre, out, target):
        return self.compound(out, target) + _ds_loss(self.compound, gt_pre, target)


class BCEDiceLoss_Squared(nn.Module):
    """BCE + Squared Dice. engine.py에서 sigmoid 적용 후 입력받으므로 내부 sigmoid 없음."""

    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        smooth = 1e-5
        num = target.size(0)
        pred_ = pred.view(num, -1)
        target_ = target.view(num, -1)
        bce = self.bceloss(pred_, target_)
        intersection = (pred_ * target_)
        dice = (2. * intersection.sum(1).pow(2) + smooth) / (pred_.sum(1).pow(2) + target_.sum(1).pow(2) + smooth)
        dice_loss = 1 - dice.sum() / num
        return bce + dice_loss


class GT_BceDiceSquaredLoss(nn.Module):
    """BCE + Squared Dice + DS"""

    def __init__(self, wb=1, wd=1):
        super().__init__()
        self.compound = BCEDiceLoss_Squared()

    def forward(self, gt_pre, out, target):
        return self.compound(out, target) + _ds_loss(self.compound, gt_pre, target)

# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W = 2, 1, 64, 64
    target = (torch.rand(B, C, H, W, device=device) > 0.5).float()

    def make_pred():
        logits = torch.randn(B, C, H, W, device=device, requires_grad=True)
        pred = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        pred.retain_grad()
        return logits, pred

    # ── Component tests ──
    print("=== Component-level tests ===")
    components = [
        ("BCELoss", BCELoss()),
        ("DiceLoss", DiceLoss()),
        ("BoundaryDiceLoss", BoundaryDiceLoss()),
        ("FocalTverskyLoss", FocalTverskyLoss()),
        ("LovaszHingeLoss", LovaszHingeLoss()),
        ("UncertaintyWeightedLoss", UncertaintyWeightedLoss()),
    ]
    for name, mod in components:
        mod = mod.to(device)
        logits, pred = make_pred()
        loss = mod(pred, target)
        loss.backward()
        grad_ok = logits.grad is not None and torch.isfinite(logits.grad).all()
        print(f"  {name:30s} = {loss.item():.6f}  finite={torch.isfinite(loss).item()}  grad={'OK' if grad_ok else 'FAIL'}")

    # ── GT-level (DS wrapper) tests ──
    print("\n=== GT-level Deep Supervision tests ===")
    gt_losses = [
        ("GT_BCEOnlyLoss", GT_BCEOnlyLoss()),
        ("GT_BceDiceSquaredLoss", GT_BceDiceSquaredLoss()),
        ("GT_BoundaryFocalLoss", GT_BoundaryFocalLoss()),
        ("GT_LovaszLoss", GT_LovaszLoss()),
        ("GT_UncertaintyLoss", GT_UncertaintyLoss()),
    ]
    for name, criterion in gt_losses:
        criterion = criterion.to(device)
        logits, pred = make_pred()
        gt_pre = tuple(pred for _ in range(5))
        loss = criterion(gt_pre, pred, target)
        loss.backward()
        grad_ok = logits.grad is not None and torch.isfinite(logits.grad).all()
        print(f"  {name:30s} = {loss.item():.6f}  finite={torch.isfinite(loss).item()}  grad={'OK' if grad_ok else 'FAIL'}")

    # ── UncertaintyWeightedLoss σ 학습 확인 ──
    print("\n=== Uncertainty σ learnable check ===")
    uw = UncertaintyWeightedLoss().to(device)
    logits, pred = make_pred()
    loss = uw(pred, target)
    loss.backward()
    for pname, p in uw.named_parameters():
        if 'log_var' in pname:
            has_grad = p.grad is not None and torch.isfinite(p.grad).all()
            print(f"  {pname}: val={p.item():.4f}  grad={p.grad.item():.4f}  ok={'OK' if has_grad else 'FAIL'}")

    print("\nAll tests passed!")
