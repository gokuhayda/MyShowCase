# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
GPT-2 to H-LLM Distillation Module
==================================

Knowledge distillation from pre-trained GPT-2 (Euclidean) to
Hyperbolic Transformer (H-LLM).

Supports:
- Distillation mode (teacher != None, lambda_distill > 0)
- LM-only mode     (teacher == None, lambda_distill == 0)

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Existing loss
from cgt.losses.hyperbolic_lm_losses import TeacherDistillationLoss


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class DistillationConfig:
    """Configuration for distillation training."""

    # Distillation
    alpha: float = 0.5
    temperature: float = 2.0
    lambda_distill: float = 0.7

    # Training
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_steps: int = 20000
    warmup_steps: int = 500
    gradient_clip: float = 1.0

    # Early stopping
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.01

    # Logging / checkpoints
    checkpoint_every: int = 1000
    eval_every: int = 500
    log_every: int = 100
    keep_last_n_checkpoints: int = 3


# ============================================================================
# TEACHER
# ============================================================================

class GPT2TeacherWrapper(nn.Module):
    """Wrapper for GPT-2 teacher model."""

    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        super().__init__()
        from transformers import GPT2LMHeadModel

        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.config = self.model.config

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = True
    ) -> Dict[str, torch.Tensor]:

        outputs = self.model(
            input_ids,
            output_hidden_states=return_hidden,
            return_dict=True
        )

        result = {"logits": outputs.logits}

        if return_hidden and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states[-1]

        return result


# ============================================================================
# TRAINER
# ============================================================================

class DistillationTrainer:
    """
    Trainer for GPT-2 → H-LLM.

    Works in two regimes:
    - Distillation: teacher != None and lambda_distill > 0
    - LM-only:      teacher == None and lambda_distill == 0
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: Optional[GPT2TeacherWrapper],
        config: DistillationConfig,
        tokenizer: Any,
        checkpoint_dir: Path,
        device: str = "cuda"
    ):
        self.student = student
        self.teacher = teacher
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )

        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            progress = (step - config.warmup_steps) / max(
                1, (config.max_steps - config.warmup_steps)
            )
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        self.distill_loss_fn = TeacherDistillationLoss(
            substrate=student.substrate,
            temperature=config.temperature,
            alpha=config.alpha
        )

        self.step = 0
        self.best_val = float("inf")
        self.patience = 0
        self.stop = False

        self.train_hist: List[Dict] = []
        self.val_hist: List[Dict] = []
        self.ckpts: List[Path] = []


    # =========================================================================
    # CHECKPOINT SAVE / LOAD
    # =========================================================================
    def save(self, is_best: bool = False):
        ckpt = {
            "step": self.step,
            "model": self.student.state_dict(),
            "opt": self.optimizer.state_dict(),
            "sched": self.scheduler.state_dict(),
            "best_val": self.best_val,
            "patience": self.patience,
            "train_hist": self.train_hist[-1000:],
            "val_hist": self.val_hist,
            "config": asdict(self.config),
        }

        path = self.checkpoint_dir / f"distill_ckpt_{self.step}.pt"
        torch.save(ckpt, path)
        self.ckpts.append(path)

        torch.save(ckpt, self.checkpoint_dir / "distill_latest.pt")

        if is_best:
            torch.save(ckpt, self.checkpoint_dir / "distill_best.pt")
            print("💾 Best distilled model saved!")

        while len(self.ckpts) > self.config.keep_last_n_checkpoints:
            old = self.ckpts.pop(0)
            if old.exists():
                old.unlink()

    def load(self, path: str):
        """
        Load training checkpoint (resume).
        Compatible with LM-only and distillation.
        """
        ckpt = torch.load(path, map_location=self.device)

        self.student.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["opt"])
        self.scheduler.load_state_dict(ckpt["sched"])

        self.step = ckpt.get("step", 0)
        self.best_val = ckpt.get("best_val", float("inf"))
        self.patience = ckpt.get("patience", 0)
        self.train_hist = ckpt.get("train_hist", [])
        self.val_hist = ckpt.get("val_hist", [])

        print(f"✅ Loaded checkpoint from step {self.step}")


    # =========================================================================
    # TRAIN STEP
    # =========================================================================
    def distillation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:

        self.student.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # ---------- Teacher (optional)
        teacher_logits = None
        teacher_hidden = None

        if self.teacher is not None and self.config.lambda_distill > 0:
            with torch.no_grad():
                t = self.teacher(input_ids, return_hidden=True)
                teacher_logits = t["logits"]
                teacher_hidden = t.get("hidden_states")

        # ---------- Student
        s = self.student(input_ids, labels=labels)
        student_logits = s["logits"]
        student_hidden = s.get("hidden_states")
        lm_loss = s["loss"]

        # ---------- Distill loss
        if teacher_logits is not None:
            d = self.distill_loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden
            )
            distill_loss = d["total"]
        else:
            distill_loss = torch.tensor(0.0, device=self.device)
            d = {}

        total_loss = (
            (1.0 - self.config.lambda_distill) * lm_loss
            + self.config.lambda_distill * distill_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.student.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self.scheduler.step()

        self.step += 1

        with torch.no_grad():
            radius = self.student.radius_statistics(student_hidden)
            fidelity = self.student.manifold_fidelity()

        return {
            "loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "distill_loss": distill_loss.item(),
            "ppl": math.exp(min(lm_loss.item(), 20)),
            "lr": self.scheduler.get_last_lr()[0],
            "nan": False,
            **radius,
            **fidelity
        }


    # =========================================================================
    # EVAL
    # =========================================================================
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:

        self.student.eval()

        total_lm = 0.0
        total_distill = 0.0
        total_tokens = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            s = self.student(input_ids, labels=labels)
            lm_loss = s["loss"]
            student_logits = s["logits"]

            teacher_logits = None
            if self.teacher is not None and self.config.lambda_distill > 0:
                t = self.teacher(input_ids, return_hidden=False)
                teacher_logits = t["logits"]

            if not torch.isnan(lm_loss):
                total_lm += lm_loss.item() * input_ids.numel()

                if teacher_logits is not None:
                    d = self.distill_loss_fn(
                        student_logits, teacher_logits, labels
                    )
                    total_distill += d["total"].item() * input_ids.numel()

                total_tokens += input_ids.numel()

        lm = total_lm / max(total_tokens, 1)
        dist = total_distill / max(total_tokens, 1)

        return {
            "val_loss": lm,
            "val_distill": dist,
            "val_ppl": math.exp(min(lm, 20))
        }


    # =========================================================================
    # TRAIN LOOP
    # =========================================================================
    def train(self, train_loader: DataLoader, val_loader: DataLoader):

        print("\n" + "=" * 60)
        print("🎓 STARTING TRAINING")
        print("=" * 60)
        print(f"Teacher: {'ON' if self.teacher is not None else 'OFF'}")
        print(f"Lambda distill: {self.config.lambda_distill}")
        print("=" * 60)

        pbar = tqdm(total=self.config.max_steps - self.step)
        data_iter = iter(train_loader)

        while self.step < self.config.max_steps and not self.stop:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            m = self.distillation_step(batch)
            self.train_hist.append(m)

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{m['loss']:.3f}",
                ppl=f"{m['ppl']:.1f}"
            )

            if self.step % self.config.eval_every == 0:
                vm = self.evaluate(val_loader)
                self.val_hist.append(vm)
                print(
                    f"\n📊 Val: loss={vm['val_loss']:.4f} "
                    f"ppl={vm['val_ppl']:.1f}"
                )

        pbar.close()
        return self.train_hist, self.val_hist


# ============================================================================
# PLOT (MODULE-LEVEL, IMPORTABLE)
# ============================================================================

def plot_distillation_analysis(
    train_hist: List[Dict],
    val_hist: List[Dict],
    save_path: Optional[Path] = None
):
    """
    Plot distillation training curves.
    """
    import matplotlib.pyplot as plt

    losses = [h["loss"] for h in train_hist]
    ppl = [h["ppl"] for h in train_hist]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.semilogy(ppl)
    plt.title("Perplexity")

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.tight_layout()
    plt.show()
