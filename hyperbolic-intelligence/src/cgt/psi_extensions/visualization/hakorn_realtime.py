# ==============================================================================
# H-AKORN REAL-TIME VISUALIZER - Dual Mode with MTEB Integration
# ==============================================================================
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
#
# Connects real STS/MTEB datasets to H-AKORN visualization.
# Supports both Poincaré disk (2D) and Lorentz hyperboloid (3D) modes.
#
# Supported dataset types:
#   - STS (Semantic Textual Similarity): sentence pairs with similarity scores
#   - Reranking: queries with positive/negative examples
#   - Clustering: sentences grouped by cluster labels
#
# New datasets added (v2):
#   - AskUbuntuDupQuestions (Reranking)
#   - SciDocsRR (Reranking)
#   - StackOverflowDupQuestions (Reranking)
#   - TwentyNewsgroupsClustering (Clustering)
#   - RedditClustering (Clustering)
# ==============================================================================

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F

# Optional imports
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle
    from matplotlib.collections import LineCollection
    from IPython.display import HTML, display
    HAS_DISPLAY = True
except ImportError:
    HAS_DISPLAY = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


# ==============================================================================
# MTEB DATASET CONFIGURATIONS
# ==============================================================================

MTEB_DATASETS = {
    # ========== Core STS Datasets (Original) ==========
    'STS12': ('mteb/sts12-sts', 'test', 'sentence1', 'sentence2', 'score', 'sts'),
    'STS13': ('mteb/sts13-sts', 'test', 'sentence1', 'sentence2', 'score', 'sts'),
    'STS14': ('mteb/sts14-sts', 'test', 'sentence1', 'sentence2', 'score', 'sts'),
    'STS15': ('mteb/sts15-sts', 'test', 'sentence1', 'sentence2', 'score', 'sts'),
    'STS16': ('mteb/sts16-sts', 'test', 'sentence1', 'sentence2', 'score', 'sts'),
    'STSBenchmark': ('mteb/stsbenchmark-sts', 'test', 'sentence1', 'sentence2', 'score', 'sts'),
    'SICK-R': ('mteb/sickr-sts', 'test', 'sentence1', 'sentence2', 'score', 'sts'),
    'BIOSSES': ('mteb/biosses-sts', 'test', 'sentence1', 'sentence2', 'score', 'sts'),
    
    # ========== Extended STS Datasets (Cross-lingual & Multilingual) ==========
    'STS17': ('mteb/sts17-crosslingual-sts', 'test', 'sentence1', 'sentence2', 'score', 'sts'),
    'STS22': ('mteb/sts22-crosslingual-sts', 'test', 'sentence1', 'sentence2', 'score', 'sts'),
    
    # ========== Reranking Datasets ==========
    'AskUbuntuDupQuestions': ('mteb/askubuntudupquestions-reranking', 'test', 'query', 'positive', None, 'reranking'),
    'SciDocsRR': ('mteb/scidocs-reranking', 'test', 'query', 'positive', None, 'reranking'),
    'StackOverflowDupQuestions': ('mteb/stackoverflowdupquestions-reranking', 'test', 'query', 'positive', None, 'reranking'),
    
    # ========== Clustering Datasets ==========
    'TwentyNewsgroupsClustering': ('mteb/twentynewsgroups-clustering', 'test', 'sentences', 'labels', None, 'clustering'),
    'RedditClustering': ('mteb/reddit-clustering', 'test', 'sentences', 'labels', None, 'clustering'),
    
    # ========== Add Your Own Datasets Here ==========
    # Format: 'DisplayName': ('org/huggingface-path', 'split', 'col1', 'col2', 'score_col', 'type')
    # Example STS:
    # 'MyDataset': ('myorg/my-sts-dataset', 'test', 'text1', 'text2', 'similarity', 'sts'),
    # Example Reranking:
    # 'MyRerank': ('myorg/my-reranking', 'test', 'query', 'positive', None, 'reranking'),
    # Example Clustering:
    # 'MyCluster': ('myorg/my-clustering', 'test', 'sentences', 'labels', None, 'clustering'),
}


# ==============================================================================
# DATA LOADER
# ==============================================================================

class MTEBDataLoader:
    """Load and prepare MTEB datasets for H-AKORN visualization."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._encoder = None
        self._encoder_name = None
        self._cache = {}
    
    def get_encoder(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Lazy load sentence transformer."""
        if self._encoder is None or self._encoder_name != model_name:
            if not HAS_ST:
                raise ImportError("sentence-transformers not installed")
            print(f"Loading encoder: {model_name}")
            self._encoder = SentenceTransformer(model_name, device=str(self.device))
            self._encoder_name = model_name
        return self._encoder
    
    def load_dataset(
        self,
        dataset_name: str = 'STSBenchmark',
        max_samples: int = 100,
        min_score: float = 0.0,
        max_score: float = 5.0,
    ) -> Dict:
        """Load MTEB dataset and return sentences with scores."""
        if not HAS_DATASETS:
            raise ImportError("datasets library not installed")
        
        if dataset_name not in MTEB_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(MTEB_DATASETS.keys())}")
        
        cache_key = f"{dataset_name}_{max_samples}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        config = MTEB_DATASETS[dataset_name]
        hub_name, split, col1, col2, score_col, dataset_type = config
        
        print(f"Loading {dataset_name} ({dataset_type}) from {hub_name}...")
        dataset = load_dataset(hub_name, split=split)
        
        if dataset_type == 'sts':
            return self._load_sts_dataset(dataset, col1, col2, score_col, max_samples, max_score, dataset_name)
        elif dataset_type == 'reranking':
            return self._load_reranking_dataset(dataset, col1, col2, max_samples, dataset_name)
        elif dataset_type == 'clustering':
            return self._load_clustering_dataset(dataset, col1, col2, max_samples, dataset_name)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _load_sts_dataset(self, dataset, s1_col, s2_col, score_col, max_samples, max_score, dataset_name):
        """Load STS-style dataset with sentence pairs and similarity scores."""
        sentences = []
        sentence_to_idx = {}
        pairs = []
        
        for i, row in enumerate(dataset):
            if i >= max_samples * 2:
                break
            
            s1 = row[s1_col]
            s2 = row[s2_col]
            score = row[score_col]
            
            if max_score > 1:
                score = score / max_score
            
            if s1 not in sentence_to_idx:
                sentence_to_idx[s1] = len(sentences)
                sentences.append(s1)
            if s2 not in sentence_to_idx:
                sentence_to_idx[s2] = len(sentences)
                sentences.append(s2)
            
            idx1 = sentence_to_idx[s1]
            idx2 = sentence_to_idx[s2]
            pairs.append((idx1, idx2, score))
            
            if len(sentences) >= max_samples:
                break
        
        result = {
            'sentences': sentences[:max_samples],
            'pairs': pairs,
            'dataset_name': dataset_name,
        }
        
        self._cache[f"{dataset_name}_{max_samples}"] = result
        print(f"Loaded {len(result['sentences'])} sentences, {len(pairs)} pairs")
        return result
    
    def _load_reranking_dataset(self, dataset, query_col, positive_col, max_samples, dataset_name):
        """Load reranking dataset with queries and positive/negative examples."""
        sentences = []
        sentence_to_idx = {}
        pairs = []
        
        for i, row in enumerate(dataset):
            if i >= max_samples:
                break
            
            query = row[query_col]
            positives = row[positive_col] if isinstance(row[positive_col], list) else [row[positive_col]]
            
            # Add query to sentences
            if query not in sentence_to_idx:
                sentence_to_idx[query] = len(sentences)
                sentences.append(query)
            query_idx = sentence_to_idx[query]
            
            # Add positive examples and create pairs with high similarity
            for positive in positives[:3]:  # Limit to 3 positives per query
                if positive not in sentence_to_idx:
                    sentence_to_idx[positive] = len(sentences)
                    sentences.append(positive)
                pos_idx = sentence_to_idx[positive]
                pairs.append((query_idx, pos_idx, 1.0))  # High similarity for positives
                
                if len(sentences) >= max_samples:
                    break
            
            if len(sentences) >= max_samples:
                break
        
        result = {
            'sentences': sentences[:max_samples],
            'pairs': pairs,
            'dataset_name': dataset_name,
        }
        
        self._cache[f"{dataset_name}_{max_samples}"] = result
        print(f"Loaded {len(result['sentences'])} sentences, {len(pairs)} pairs (reranking)")
        return result
    
    def _load_clustering_dataset(self, dataset, sentences_col, labels_col, max_samples, dataset_name):
        """Load clustering dataset with sentences and cluster labels."""
        sentences = []
        sentence_to_idx = {}
        pairs = []
        labels_dict = {}
        
        for i, row in enumerate(dataset):
            if i >= max_samples:
                break
            
            sentence_list = row[sentences_col] if isinstance(row[sentences_col], list) else [row[sentences_col]]
            label_list = row[labels_col] if isinstance(row[labels_col], list) else [row[labels_col]]
            
            # Process sentences and their labels
            for sentence, label in zip(sentence_list, label_list):
                if sentence not in sentence_to_idx and len(sentences) < max_samples:
                    sentence_to_idx[sentence] = len(sentences)
                    sentences.append(sentence)
                    labels_dict[len(sentences) - 1] = label
            
            if len(sentences) >= max_samples:
                break
        
        # Create pairs based on cluster membership
        # Sentences in the same cluster have high similarity (0.8)
        # Sentences in different clusters have low similarity (0.2)
        for idx1 in range(len(sentences)):
            for idx2 in range(idx1 + 1, min(idx1 + 10, len(sentences))):  # Limit pairs per sentence
                label1 = labels_dict.get(idx1)
                label2 = labels_dict.get(idx2)
                
                if label1 is not None and label2 is not None:
                    similarity = 0.8 if label1 == label2 else 0.2
                    pairs.append((idx1, idx2, similarity))
        
        result = {
            'sentences': sentences[:max_samples],
            'pairs': pairs,
            'dataset_name': dataset_name,
        }
        
        self._cache[f"{dataset_name}_{max_samples}"] = result
        print(f"Loaded {len(result['sentences'])} sentences, {len(pairs)} pairs (clustering)")
        return result
    
    def encode_sentences(
        self,
        sentences: List[str],
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode sentences to embeddings."""
        encoder = self.get_encoder(model_name)
        
        print(f"Encoding {len(sentences)} sentences...")
        embeddings = encoder.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=str(self.device),
        )
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings.to(self.device)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class RealtimeConfig:
    """Configuration for real-time visualization."""
    # Kuramoto
    K: float = 3.0
    dt: float = 0.05
    curvature: float = 1.0
    frequency_spread: float = 0.2
    
    # Dynamics
    drift_rate: float = 0.02
    topological_feedback: bool = True
    feedback_strength: float = 0.15
    
    # Graph
    k_neighbors: int = 6
    use_similarity_graph: bool = True
    similarity_threshold: float = 0.5
    
    # Visualization mode: 'lorentz' (3D) or 'poincare' (2D)
    mode: Literal['lorentz', 'poincare'] = 'lorentz'
    
    # Display
    update_interval: int = 50
    trail_length: int = 10
    show_labels: bool = True
    max_labels: int = 20
    show_surface: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# REAL-TIME H-AKORN SIMULATOR
# ==============================================================================

class RealtimeHAKORN:
    """
    Real-time H-AKORN visualization with MTEB dataset integration.
    Supports both Poincaré (2D) and Lorentz (3D) visualization modes.
    """
    
    def __init__(
        self,
        config: Optional[RealtimeConfig] = None,
        sentences: Optional[List[str]] = None,
        embeddings: Optional[torch.Tensor] = None,
        similarity_pairs: Optional[List[Tuple[int, int, float]]] = None,
    ):
        self.config = config or RealtimeConfig()
        self.device = torch.device(self.config.device)
        
        self.sentences = sentences or []
        self.N = len(self.sentences)
        
        if embeddings is not None:
            self.embeddings = embeddings.to(self.device)
        else:
            self.embeddings = None
        
        self.similarity_pairs = similarity_pairs or []
        
        # State
        self.positions = None
        self.phases = None
        self.omega = None
        self.target_positions = None
        self.target_spatial = None
        self.adjacency = None
        
        # History
        self.history = {
            'positions': [],
            'positions_3d': [],
            'positions_2d': [],
            'phases': [],
            'R': [],
        }
        self.step_count = 0
    
    def initialize(self):
        """Initialize simulation state from embeddings."""
        if self.embeddings is None:
            raise ValueError("No embeddings provided")
        
        self.N = self.embeddings.shape[0]
        
        # Project embeddings to 2D spatial
        emb = self.embeddings
        if emb.shape[1] > 2:
            U, S, V = torch.linalg.svd(emb, full_matrices=False)
            emb_2d = U[:, :2] * S[:2]
        else:
            emb_2d = emb[:, :2] if emb.shape[1] >= 2 else F.pad(emb, (0, 2 - emb.shape[1]))
        
        # Normalize and scale
        emb_2d = F.normalize(emb_2d, p=2, dim=-1) * 1.2
        self.target_spatial = emb_2d
        
        # Project to hyperboloid
        t = torch.sqrt(1 + (emb_2d ** 2).sum(dim=-1, keepdim=True))
        self.target_positions = torch.cat([t, emb_2d], dim=-1)
        
        # Random initial chaos state
        init_spatial = (torch.rand(self.N, 2, device=self.device) - 0.5) * 2.5
        t_init = torch.sqrt(1 + (init_spatial ** 2).sum(dim=-1, keepdim=True))
        self.positions = torch.cat([t_init, init_spatial], dim=-1)
        
        # Random phases
        self.phases = torch.rand(self.N, device=self.device) * 2 * math.pi
        
        # Natural frequencies
        self.omega = torch.randn(self.N, device=self.device) * self.config.frequency_spread
        
        # Build adjacency
        self._build_adjacency()
        
        # Clear history
        self.history = {'positions': [], 'positions_3d': [], 'positions_2d': [], 'phases': [], 'R': []}
        self.step_count = 0
        
        print(f"Initialized H-AKORN with {self.N} sentences (mode: {self.config.mode})")
    
    def _build_adjacency(self):
        """Build adjacency matrix."""
        self.adjacency = torch.zeros(self.N, self.N, device=self.device)
        
        if self.config.use_similarity_graph and self.similarity_pairs:
            for i, j, score in self.similarity_pairs:
                if i < self.N and j < self.N and score >= self.config.similarity_threshold:
                    self.adjacency[i, j] = score
                    self.adjacency[j, i] = score
        
        # Add kNN connections
        if self.target_positions is not None:
            inner = -self.target_positions[:, 0:1] @ self.target_positions[:, 0:1].T + \
                    self.target_positions[:, 1:] @ self.target_positions[:, 1:].T
            dist = torch.acosh(torch.clamp(-inner, min=1.0 + 1e-7))
            
            k = min(self.config.k_neighbors, self.N - 1)
            _, indices = torch.topk(dist, k + 1, largest=False, dim=1)
            
            for i in range(self.N):
                for j in indices[i, 1:k+1]:
                    if self.adjacency[i, j] == 0:
                        self.adjacency[i, j] = 0.5
                        self.adjacency[j, i] = 0.5
    
    def to_lorentz_3d(self, h: torch.Tensor) -> torch.Tensor:
        """Return 3D Lorentz coordinates (t, x, y)."""
        return h[..., :3]
    
    def to_poincare(self, h: torch.Tensor) -> torch.Tensor:
        """Convert to Poincaré disk."""
        return h[..., 1:3] / (1 + h[..., 0:1])
    
    def compute_order_parameter(self) -> float:
        """Compute Kuramoto order parameter Γ."""
        z = torch.exp(1j * self.phases.to(torch.complex64))
        return z.mean().abs().item()
    
    def get_phase_state(self) -> str:
        """Get current phase state name."""
        R = self.compute_order_parameter()
        if R < 0.3:
            return "CHAOS"
        elif R < 0.5:
            return "DRIFT"
        elif R < 0.7:
            return "METASTABLE"
        else:
            return "EMERGENCE"
    
    @torch.no_grad()
    def step(self, n_steps: int = 1):
        """Run Kuramoto dynamics for n steps."""
        K = self.config.K
        dt = self.config.dt
        kappa = self.config.curvature
        
        for _ in range(n_steps):
            # Hyperbolic distance
            inner = -self.positions[:, 0:1] @ self.positions[:, 0:1].T + \
                    self.positions[:, 1:] @ self.positions[:, 1:].T
            dist = torch.acosh(torch.clamp(-inner, min=1.0 + 1e-7))
            
            # Coupling
            coupling = self.adjacency * torch.exp(-dist * kappa)
            
            # Phase dynamics
            phase_diff = self.phases.unsqueeze(0) - self.phases.unsqueeze(1)
            interaction = (coupling * torch.sin(phase_diff)).sum(dim=1)
            
            R = self.compute_order_parameter()
            
            dtheta = self.omega + (K / max(1, self.N)) * interaction
            
            if self.config.topological_feedback:
                dtheta *= (1 + self.config.feedback_strength * (R - 0.5))
            
            # RK2
            k1 = dtheta
            phases_mid = (self.phases + 0.5 * dt * k1) % (2 * math.pi)
            phase_diff_mid = phases_mid.unsqueeze(0) - phases_mid.unsqueeze(1)
            k2 = self.omega + (K / max(1, self.N)) * (coupling * torch.sin(phase_diff_mid)).sum(dim=1)
            if self.config.topological_feedback:
                k2 *= (1 + self.config.feedback_strength * (R - 0.5))
            
            self.phases = (self.phases + dt * k2) % (2 * math.pi)
            
            # Drift toward targets
            alpha = self.config.drift_rate * (0.5 + R * 0.5)
            spatial = self.positions[:, 1:]
            spatial = spatial + alpha * (self.target_spatial - spatial)
            t = torch.sqrt(1 + (spatial ** 2).sum(dim=-1, keepdim=True))
            self.positions = torch.cat([t, spatial], dim=-1)
            
            self.step_count += 1
            
            if self.step_count % 2 == 0:
                pos_3d = self.to_lorentz_3d(self.positions).cpu().numpy()
                pos_2d = self.to_poincare(self.positions).cpu().numpy()
                
                self.history['positions_3d'].append(pos_3d.copy())
                self.history['positions_2d'].append(pos_2d.copy())
                
                if self.config.mode == 'lorentz':
                    self.history['positions'].append(pos_3d.copy())
                else:
                    self.history['positions'].append(pos_2d.copy())
                
                self.history['phases'].append(self.phases.cpu().numpy().copy())
                self.history['R'].append(R)
                
                if len(self.history['positions']) > self.config.trail_length:
                    self.history['positions'].pop(0)
                    self.history['positions_3d'].pop(0)
                    self.history['positions_2d'].pop(0)


# ==============================================================================
# HYPERBOLOID MESH
# ==============================================================================

def create_hyperboloid_mesh(resolution: int = 30, t_max: float = 2.5):
    """Create mesh for Lorentz hyperboloid surface."""
    r = np.linspace(0, np.arccosh(t_max), resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    R, Theta = np.meshgrid(r, theta)
    
    T = np.cosh(R)
    X = np.sinh(R) * np.cos(Theta)
    Y = np.sinh(R) * np.sin(Theta)
    
    return T, X, Y


# ==============================================================================
# DEMO FUNCTIONS
# ==============================================================================

def run_realtime_demo(
    dataset_name: str = 'STSBenchmark',
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    max_samples: int = 40,
    steps: int = 300,
    K: float = 3.0,
    mode: str = 'lorentz',  # 'lorentz' or 'poincare'
    device: str = "auto",
):
    """
    Run H-AKORN visualization with real MTEB dataset.
    
    Parameters
    ----------
    dataset_name : str
        MTEB dataset name
    model_name : str
        Sentence transformer model
    max_samples : int
        Maximum number of sentences
    steps : int
        Simulation steps
    K : float
        Coupling strength
    mode : str
        'lorentz' for 3D hyperboloid, 'poincare' for 2D disk
    device : str
        'auto', 'cuda', or 'cpu'
    
    Returns
    -------
    anim, sim : FuncAnimation, RealtimeHAKORN
    
    Example
    -------
    ```python
    # 3D Lorentz (default)
    anim, sim = run_realtime_demo('STSBenchmark', mode='lorentz')
    
    # 2D Poincaré
    anim, sim = run_realtime_demo('STSBenchmark', mode='poincare')
    
    HTML(anim.to_jshtml())
    ```
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mode_name = "3D Lorentz Hyperboloid" if mode == 'lorentz' else "2D Poincaré Disk"
    
    print("=" * 60)
    print(f"H-AKORN: REAL-TIME SEMANTIC RESONANCE ({mode_name})")
    print(f"Dataset: {dataset_name}")
    print(f"Encoder: {model_name}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load data
    loader = MTEBDataLoader(device=device)
    data = loader.load_dataset(dataset_name, max_samples=max_samples)
    embeddings = loader.encode_sentences(data['sentences'], model_name)
    
    # Create simulator
    config = RealtimeConfig(K=K, device=device, mode=mode, show_labels=True, max_labels=15)
    sim = RealtimeHAKORN(
        config=config,
        sentences=data['sentences'],
        embeddings=embeddings,
        similarity_pairs=data['pairs'],
    )
    sim.initialize()
    
    # Run simulation
    print(f"\nRunning {steps} steps...")
    start = time.time()
    
    for i in range(0, steps, 10):
        sim.step(10)
        if i % 50 == 0:
            R = sim.compute_order_parameter()
            print(f"  t={sim.step_count}: Γ={R:.3f} [{sim.get_phase_state()}]")
    
    elapsed = time.time() - start
    print(f"\n✓ Completed in {elapsed:.2f}s")
    print(f"Final Γ = {sim.compute_order_parameter():.4f}")
    
    if not HAS_DISPLAY:
        return None, sim
    
    print(f"\n🎬 Creating {mode_name} animation...")
    
    # Reset for animation
    sim.initialize()
    
    if mode == 'lorentz':
        anim = _create_animation_3d(sim, data['sentences'])
    else:
        anim = _create_animation_2d(sim, data['sentences'])
    
    return anim, sim


def _create_animation_3d(sim, sentences):
    """Create 3D Lorentz animation."""
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a12')
    
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_3d.set_facecolor('#0a0a12')
    
    ax_gamma = fig.add_subplot(122)
    ax_gamma.set_facecolor('#0a0a12')
    
    # Draw hyperboloid surface
    T, X, Y = create_hyperboloid_mesh(resolution=25, t_max=2.5)
    ax_3d.plot_surface(X, Y, T, alpha=0.1, color='cyan', edgecolor='none')
    
    pos_init = sim.to_lorentz_3d(sim.positions).cpu().numpy()
    scatter = ax_3d.scatter(pos_init[:, 1], pos_init[:, 2], pos_init[:, 0],
                            c='cyan', s=80, alpha=0.9,
                            edgecolors='white', linewidth=1)
    
    ax_3d.set_xlabel('X', color='white')
    ax_3d.set_ylabel('Y', color='white')
    ax_3d.set_zlabel('t', color='white')
    ax_3d.tick_params(colors='white')
    title = ax_3d.set_title('', color='white', fontsize=12, fontweight='bold')
    
    # Pane styling
    for pane in [ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('white')
        pane.set_alpha(0.1)
    
    # Gamma plot
    ax_gamma.set_xlim(0, 200)
    ax_gamma.set_ylim(0, 1)
    ax_gamma.set_xlabel('Time', color='white')
    ax_gamma.set_ylabel('Γ', color='white', fontsize=14)
    ax_gamma.tick_params(colors='white')
    ax_gamma.axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
    ax_gamma.axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
    ax_gamma.axhspan(0.3, 0.7, alpha=0.1, color='cyan')
    
    line_gamma, = ax_gamma.plot([], [], color='cyan', linewidth=2)
    
    R_history = []
    connection_lines = []
    
    def update(frame):
        nonlocal connection_lines
        
        sim.step(5)
        
        pos = sim.to_lorentz_3d(sim.positions).cpu().numpy()
        phases = sim.phases.cpu().numpy()
        R = sim.compute_order_parameter()
        state = sim.get_phase_state()
        
        R_history.append(R)
        
        scatter._offsets3d = (pos[:, 1], pos[:, 2], pos[:, 0])
        colors = plt.cm.hsv(phases / (2 * np.pi))
        scatter.set_facecolors(colors)
        
        for line in connection_lines:
            line.remove()
        connection_lines = []
        
        adj = sim.adjacency.cpu().numpy()
        for i in range(sim.N):
            for j in range(i + 1, sim.N):
                if adj[i, j] > 0:
                    coh = (1 + np.cos(phases[i] - phases[j])) / 2
                    line, = ax_3d.plot([pos[i, 1], pos[j, 1]],
                                       [pos[i, 2], pos[j, 2]],
                                       [pos[i, 0], pos[j, 0]],
                                       color='cyan', alpha=0.05 + 0.2 * coh,
                                       linewidth=0.5)
                    connection_lines.append(line)
        
        title.set_text(f'Lorentz Hyperboloid | t={sim.step_count} | Γ={R:.3f} | {state}')
        
        line_gamma.set_data(range(len(R_history)), R_history)
        ax_gamma.set_xlim(0, max(200, len(R_history)))
        
        ax_3d.view_init(elev=25, azim=30 + frame * 0.8)
        
        return scatter, line_gamma, title
    
    anim = FuncAnimation(fig, update, frames=150, interval=50, blit=False)
    plt.tight_layout()
    
    return anim


def _create_animation_2d(sim, sentences):
    """Create 2D Poincaré animation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a12')
    
    ax_poincare = axes[0]
    ax_gamma = axes[1]
    
    ax_poincare.set_facecolor('#0a0a12')
    ax_poincare.set_aspect('equal')
    ax_poincare.set_xlim(-1.15, 1.15)
    ax_poincare.set_ylim(-1.15, 1.15)
    
    circle = Circle((0, 0), 1, fill=False, color='white', linewidth=2, alpha=0.3)
    ax_poincare.add_patch(circle)
    ax_poincare.axis('off')
    
    pos_init = sim.to_poincare(sim.positions).cpu().numpy()
    scatter = ax_poincare.scatter(pos_init[:, 0], pos_init[:, 1],
                                   c='cyan', s=80, alpha=0.9,
                                   edgecolors='white', linewidth=1)
    
    lines = LineCollection([], colors='cyan', alpha=0.2, linewidths=0.5)
    ax_poincare.add_collection(lines)
    
    title = ax_poincare.text(0, 1.08, '', ha='center', va='bottom',
                              fontsize=12, color='white', fontweight='bold',
                              transform=ax_poincare.transAxes)
    
    # Gamma plot
    ax_gamma.set_facecolor('#0a0a12')
    ax_gamma.set_xlim(0, 200)
    ax_gamma.set_ylim(0, 1)
    ax_gamma.set_xlabel('Time', color='white')
    ax_gamma.set_ylabel('Γ', color='white', fontsize=14)
    ax_gamma.tick_params(colors='white')
    ax_gamma.axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
    ax_gamma.axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
    ax_gamma.axhspan(0.3, 0.7, alpha=0.1, color='cyan')
    
    line_gamma, = ax_gamma.plot([], [], color='cyan', linewidth=2)
    
    R_history = []
    
    def update(frame):
        sim.step(5)
        
        pos = sim.to_poincare(sim.positions).cpu().numpy()
        phases = sim.phases.cpu().numpy()
        R = sim.compute_order_parameter()
        state = sim.get_phase_state()
        
        R_history.append(R)
        
        scatter.set_offsets(pos)
        colors = plt.cm.hsv(phases / (2 * np.pi))
        scatter.set_facecolors(colors)
        
        segments = []
        adj = sim.adjacency.cpu().numpy()
        for i in range(sim.N):
            for j in range(i + 1, sim.N):
                if adj[i, j] > 0:
                    segments.append([pos[i], pos[j]])
        lines.set_segments(segments)
        
        title.set_text(f'Poincaré Disk | t={sim.step_count} | Γ={R:.3f} | {state}')
        
        line_gamma.set_data(range(len(R_history)), R_history)
        ax_gamma.set_xlim(0, max(200, len(R_history)))
        
        return scatter, line_gamma, title, lines
    
    anim = FuncAnimation(fig, update, frames=150, interval=50, blit=False)
    plt.tight_layout()
    
    return anim


def visualize_sts_resonance(
    sentences: List[str],
    embeddings: torch.Tensor,
    pairs: Optional[List[Tuple[int, int, float]]] = None,
    K: float = 3.0,
    steps: int = 200,
    mode: str = 'lorentz',
    device: str = "auto",
):
    """Visualize semantic resonance for custom sentences."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = RealtimeConfig(K=K, device=device, mode=mode)
    sim = RealtimeHAKORN(
        config=config,
        sentences=sentences,
        embeddings=embeddings,
        similarity_pairs=pairs or [],
    )
    sim.initialize()
    
    for _ in range(steps):
        sim.step(1)
    
    return sim


def create_interactive_hakorn(*args, **kwargs):
    """Placeholder for interactive widget."""
    print("Interactive widgets require ipywidgets. Use run_realtime_demo() instead.")
    return None


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'RealtimeConfig',
    'RealtimeHAKORN',
    'MTEBDataLoader',
    'MTEB_DATASETS',
    'run_realtime_demo',
    'visualize_sts_resonance',
    'create_interactive_hakorn',
    'create_hyperboloid_mesh',
]
