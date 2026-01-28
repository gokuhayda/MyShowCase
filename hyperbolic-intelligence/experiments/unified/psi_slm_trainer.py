"""
PSI-SLM Full Trainer for Unified Pipeline
==========================================

Minimal wrapper to integrate PSI-SLM components into the unified experiment pipeline.

This trainer uses:
- CGTStudentHardened as the base model (unchanged)
- MultiObjectiveLoss (standard CGT loss, proven to work)
- Optional: H-AKOrN, TopologicalField, GW loss when available

The trainer maintains full compatibility with the UnifiedTrainer interface
so PSI_SLM_FULL can participate in the model × dataset × seed grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from scipy.stats import spearmanr

# Core CGT imports (these are stable and unchanged)
from cgt.models.cgt_hardened import CGTStudentHardened
from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened
from cgt.losses import MultiObjectiveLoss

# Try importing psi_extensions components
HAS_PSI_EXTENSIONS = False
HAS_AKORN = False
HAS_TOPO = False
HAS_GW = False

try:
    from cgt.psi_extensions.binding import HyperbolicAKORN
    HAS_AKORN = True
except ImportError:
    pass

try:
    from cgt.psi_extensions.topology import TopologicalConstraintField
    HAS_TOPO = True
except ImportError:
    pass

try:
    from cgt.psi_extensions.transfer import GromovWassersteinLoss
    HAS_GW = True
except ImportError:
    pass

HAS_PSI_EXTENSIONS = HAS_AKORN or HAS_TOPO or HAS_GW

logger = logging.getLogger(__name__)


class PsiSlmFullTrainer:
    """
    Full PSI-SLM trainer for unified pipeline.
    
    Uses the standard MultiObjectiveLoss (proven to work) with optional
    PSI extensions when available.
    
    Compatible with UnifiedTrainer interface for grid evaluation.
    """
    
    def __init__(
        self,
        model_type: Any,  # ModelType enum
        output_dir: Path,
        student_dim: int = 128,   # PSI_SLM uses 128 (not 32)
        teacher_dim: int = 768,   # PSI_SLM uses 768D teacher (not 384)
        hidden_dim: int = 1024,   # PSI_SLM uses 1024 (not 256)
        num_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        # Loss weights (passed to MultiObjectiveLoss)
        lambda_contrastive: float = 1.0,
        lambda_distill: float = 0.7,
        lambda_spectral: float = 0.2,
        lambda_topo: float = 0.02,
        lambda_lipschitz: float = 0.01,
        # Early stopping
        early_stopping_patience: int = 25,
        early_stopping_min_delta: float = 0.0001,
        # Device
        device: Optional[str] = None,
    ):
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Loss weights
        self.lambda_contrastive = lambda_contrastive
        self.lambda_distill = lambda_distill
        self.lambda_spectral = lambda_spectral
        self.lambda_topo = lambda_topo
        self.lambda_lipschitz = lambda_lipschitz
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Will be initialized in train()
        self.model = None
        self.substrate = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        
        logger.info(f"PsiSlmFullTrainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  PSI extensions: AKORN={HAS_AKORN}, TOPO={HAS_TOPO}, GW={HAS_GW}")
    
    def _create_model(self) -> Tuple[CGTStudentHardened, LorentzSubstrateHardened]:
        """Create CGT student model and substrate."""
        model = CGTStudentHardened(
            teacher_dim=self.teacher_dim,
            student_dim=self.student_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        
        # Use the substrate from the model
        substrate = model.substrate
        
        return model, substrate
    
    def _create_loss(self) -> MultiObjectiveLoss:
        """Create standard MultiObjectiveLoss."""
        return MultiObjectiveLoss(
            lambda_contrastive=self.lambda_contrastive,
            lambda_distill=self.lambda_distill,
            lambda_spectral=self.lambda_spectral,
            lambda_topo=self.lambda_topo,
            lambda_lipschitz=self.lambda_lipschitz,
            num_epochs=self.num_epochs,
        )
    
    def _create_optimizer(self) -> Tuple[AdamW, CosineAnnealingLR]:
        """Create optimizer and scheduler."""
        params = list(self.model.parameters())
        
        optimizer = AdamW(params, lr=self.learning_rate, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        return optimizer, scheduler
    
    def train(
        self,
        train_emb1: torch.Tensor,
        train_emb2: torch.Tensor,
        train_scores: torch.Tensor,
        val_emb1: torch.Tensor,
        val_emb2: torch.Tensor,
        val_scores: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Train the PSI-SLM model.
        
        Args:
            train_emb1: First sentence teacher embeddings (train)
            train_emb2: Second sentence teacher embeddings (train)
            train_scores: Similarity scores (train)
            val_emb1: First sentence teacher embeddings (val)
            val_emb2: Second sentence teacher embeddings (val)
            val_scores: Similarity scores (val)
            
        Returns:
            Dictionary with training results
        """
        logger.info("=" * 60)
        logger.info("PSI_SLM_FULL TRAINING")
        logger.info("=" * 60)
        
        # Initialize components
        self.model, self.substrate = self._create_model()
        self.loss_fn = self._create_loss()
        self.optimizer, self.scheduler = self._create_optimizer()
        
        # Move data to device
        train_emb1 = train_emb1.to(self.device)
        train_emb2 = train_emb2.to(self.device)
        train_scores = train_scores.to(self.device)
        val_emb1 = val_emb1.to(self.device)
        val_emb2 = val_emb2.to(self.device)
        val_scores = val_scores.to(self.device)
        
        n_train = len(train_emb1)
        n_batches = (n_train + self.batch_size - 1) // self.batch_size
        
        best_val_rho = -1.0
        best_epoch = 0
        epochs_no_improve = 0
        history = []
        
        # Early stopping config
        patience = getattr(self, 'early_stopping_patience', 25)
        min_delta = getattr(self, 'early_stopping_min_delta', 0.0001)
        
        logger.info(f"Early stopping: patience={patience}, min_delta={min_delta}")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            
            epoch_loss = 0.0
            
            # Shuffle
            perm = torch.randperm(n_train, device=self.device)
            
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, n_train)
                idx = perm[start:end]
                
                batch_t1 = train_emb1[idx]
                batch_t2 = train_emb2[idx]
                
                # Concatenate for batch processing
                # MultiObjectiveLoss expects single batch of embeddings
                teacher_batch = torch.cat([batch_t1, batch_t2], dim=0)
                
                self.optimizer.zero_grad()
                
                # Forward through student (returns tensor directly)
                student_batch = self.model(teacher_batch)
                
                # Compute loss using MultiObjectiveLoss interface
                loss_dict = self.loss_fn(
                    student_emb=student_batch,
                    teacher_emb=teacher_batch,
                    model=self.model,
                    current_epoch=epoch,
                )
                
                loss = loss_dict["total"]
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            self.scheduler.step()
            
            avg_loss = epoch_loss / n_batches
            
            # Validation
            val_rho = self._evaluate(val_emb1, val_emb2, val_scores)
            
            # Check for improvement
            if val_rho > best_val_rho + min_delta:
                best_val_rho = val_rho
                best_epoch = epoch
                epochs_no_improve = 0
                # Save best model
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "psi_slm_full_best.pt"
                )
            else:
                epochs_no_improve += 1
            
            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "val_rho": val_rho,
            })
            
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val ρ: {val_rho:.4f}"
                )
            
            # Early stopping check
            if epochs_no_improve >= patience:
                logger.info(f"\n⏹️ Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        logger.info("=" * 60)
        logger.info(f"Training complete. Best Val ρ: {best_val_rho:.4f} @ epoch {best_epoch}")
        logger.info("=" * 60)
        
        return {
            "best_val_rho": best_val_rho,
            "best_epoch": best_epoch,
            "history": history,
            "model_path": str(self.output_dir / "psi_slm_full_best.pt"),
        }
    
    def _evaluate(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        scores: torch.Tensor,
    ) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        
        with torch.no_grad():
            # Model returns tensor directly
            student_emb1 = self.model(emb1)
            student_emb2 = self.model(emb2)
            
            # Compute geodesic distances
            dists = self.substrate.dist(student_emb1, student_emb2)
            
            # Convert to similarities (inverse distance)
            sims = 1.0 / (1.0 + dists)
            
            # Spearman correlation
            rho, _ = spearmanr(
                sims.cpu().numpy(),
                scores.cpu().numpy()
            )
        
        self.model.train()
        return rho if not (rho != rho) else 0.0  # Check for NaN
