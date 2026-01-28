# ==============================================================================
# H-AKORN VIDEO RECORDER - Dual Mode: Poincaré 2D & Lorentz 3D
# ==============================================================================
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
#
# Records H-AKORN simulations as videos with:
# - Timelapse acceleration
# - Sentence/intent overlays
# - Both Poincaré (2D) and Lorentz (3D) rendering modes
# ==============================================================================

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Circle
    from matplotlib.collections import LineCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class VideoConfig:
    """Configuration for video recording."""
    # Output
    output_path: str = "hakorn_resonance.mp4"
    format: str = "mp4"
    
    # Resolution
    width: int = 1920
    height: int = 1080
    dpi: int = 150
    
    # Timing
    fps: int = 30
    duration_seconds: float = 15.0
    timelapse_factor: int = 3
    
    # Visualization mode: 'lorentz' (3D) or 'poincare' (2D)
    mode: Literal['lorentz', 'poincare'] = 'lorentz'
    
    # Content
    show_labels: bool = True
    max_labels: int = 12
    show_surface: bool = True
    show_metrics: bool = True
    rotate_camera: bool = True
    rotation_speed: float = 0.8
    
    # Style
    background_color: str = "#0a0a12"
    surface_color: str = "cyan"
    surface_alpha: float = 0.12
    
    # Camera (3D only)
    initial_elev: float = 25
    initial_azim: float = 30
    
    # Cinematic
    fade_in_frames: int = 15
    fade_out_frames: int = 15
    intro_text: Optional[str] = "H-AKORN: Semantic Resonance"


# ==============================================================================
# HYPERBOLOID MESH (3D)
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
# FRAME RENDERER - DUAL MODE
# ==============================================================================

class FrameRenderer:
    """Renders frames for video in both 2D and 3D modes."""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.fig_width = config.width / config.dpi
        self.fig_height = config.height / config.dpi
        
        if config.mode == 'lorentz':
            self.T_mesh, self.X_mesh, self.Y_mesh = create_hyperboloid_mesh(
                resolution=25, t_max=2.5
            )
        
        self._setup_figure()
    
    def _setup_figure(self):
        """Create figure with appropriate layout."""
        self.fig = plt.figure(figsize=(self.fig_width, self.fig_height),
                               dpi=self.config.dpi)
        self.fig.patch.set_facecolor(self.config.background_color)
        
        if self.config.mode == 'lorentz':
            self.ax_main = self.fig.add_subplot(121, projection='3d')
        else:
            self.ax_main = self.fig.add_subplot(121)
        
        self.ax_gamma = self.fig.add_subplot(122)
        
        self.ax_main.set_facecolor(self.config.background_color)
        self._setup_gamma_axis()
    
    def _setup_gamma_axis(self):
        """Configure the order parameter plot."""
        ax = self.ax_gamma
        ax.set_facecolor(self.config.background_color)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.tick_params(colors='white', labelsize=10)
        ax.set_xlabel('Time', color='white', fontsize=12)
        ax.set_ylabel('Γ', color='white', fontsize=16)
        ax.set_title('Order Parameter', color='white', fontsize=12)
        
        ax.axhspan(0, 0.3, alpha=0.1, color='red')
        ax.axhspan(0.3, 0.7, alpha=0.1, color='cyan')
        ax.axhspan(0.7, 1.0, alpha=0.1, color='green')
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
        
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)
    
    def render_frame(
        self,
        positions_3d: np.ndarray,   # (N, 3) Lorentz (t, x, y)
        positions_2d: np.ndarray,   # (N, 2) Poincaré
        phases: np.ndarray,
        adjacency: np.ndarray,
        gamma: float,
        gamma_history: List[float],
        step: int,
        total_steps: int,
        sentences: Optional[List[str]] = None,
        state: str = "CHAOS",
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Render a single frame."""
        self.ax_main.clear()
        self.ax_gamma.clear()
        self._setup_gamma_axis()
        
        self.ax_main.set_facecolor(self.config.background_color)
        
        N = len(positions_3d)
        colors = plt.cm.hsv(phases / (2 * np.pi))
        
        if self.config.mode == 'lorentz':
            self._render_3d(positions_3d, phases, adjacency, colors, sentences, frame_idx)
        else:
            self._render_2d(positions_2d, phases, adjacency, colors, sentences)
        
        # Title
        state_colors = {'CHAOS': '#ff4444', 'DRIFT': '#ffaa00', 
                       'METASTABLE': '#00aaff', 'EMERGENCE': '#00ff88'}
        title_color = state_colors.get(state, 'white')
        
        mode_name = "Lorentz Hyperboloid" if self.config.mode == 'lorentz' else "Poincaré Disk"
        self.ax_main.set_title(
            f'{self.config.intro_text or "H-AKORN"}\nt={step} | Γ={gamma:.3f} | {state}',
            color=title_color, fontsize=12, fontweight='bold'
        )
        
        # Gamma plot
        if gamma_history:
            self.ax_gamma.plot(range(len(gamma_history)), gamma_history,
                              color='cyan', linewidth=2)
            self.ax_gamma.scatter([len(gamma_history)-1], [gamma_history[-1]],
                                  color='cyan', s=50, zorder=10)
            self.ax_gamma.set_xlim(0, max(100, len(gamma_history)))
        
        plt.tight_layout()
        
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        rgb = img[:, :, :3]
        
        return rgb
    
    def _render_3d(self, positions, phases, adjacency, colors, sentences, frame_idx):
        """Render 3D Lorentz frame."""
        ax = self.ax_main
        
        if self.config.show_surface:
            ax.plot_surface(
                self.X_mesh, self.Y_mesh, self.T_mesh,
                alpha=self.config.surface_alpha,
                color=self.config.surface_color,
                edgecolor='none'
            )
        
        # Plot points (x, y, t)
        ax.scatter(
            positions[:, 1], positions[:, 2], positions[:, 0],
            c=colors, s=100, alpha=0.9,
            edgecolors='white', linewidth=1.5, depthshade=True
        )
        
        # Connections
        N = len(positions)
        for i in range(N):
            for j in range(i + 1, N):
                if adjacency[i, j] > 0:
                    coh = (1 + np.cos(phases[i] - phases[j])) / 2
                    ax.plot(
                        [positions[i, 1], positions[j, 1]],
                        [positions[i, 2], positions[j, 2]],
                        [positions[i, 0], positions[j, 0]],
                        color='cyan', alpha=0.05 + 0.25 * coh, linewidth=0.6
                    )
        
        # Labels
        if self.config.show_labels and sentences:
            n_labels = min(self.config.max_labels, len(sentences))
            for i in range(n_labels):
                label = sentences[i][:18] + '...' if len(sentences[i]) > 18 else sentences[i]
                ax.text(positions[i, 1], positions[i, 2], positions[i, 0] + 0.1,
                       label, fontsize=7, color='white', alpha=0.7,
                       ha='center', va='bottom')
        
        ax.set_xlabel('X', color='white', fontsize=10)
        ax.set_ylabel('Y', color='white', fontsize=10)
        ax.set_zlabel('t', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('white')
            pane.set_alpha(0.1)
        
        if self.config.rotate_camera:
            azim = self.config.initial_azim + frame_idx * self.config.rotation_speed
            ax.view_init(elev=self.config.initial_elev, azim=azim)
        else:
            ax.view_init(elev=self.config.initial_elev, azim=self.config.initial_azim)
    
    def _render_2d(self, positions, phases, adjacency, colors, sentences):
        """Render 2D Poincaré frame."""
        ax = self.ax_main
        
        ax.set_aspect('equal')
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        
        # Poincaré disk boundary
        circle = Circle((0, 0), 1, fill=False, color='white', linewidth=2, alpha=0.3)
        ax.add_patch(circle)
        
        # Inner glow
        for r, alpha in [(0.97, 0.1), (0.94, 0.05)]:
            ring = Circle((0, 0), r, fill=False, color='cyan', linewidth=1, alpha=alpha)
            ax.add_patch(ring)
        
        ax.axis('off')
        
        # Connections
        N = len(positions)
        segments = []
        line_colors = []
        for i in range(N):
            for j in range(i + 1, N):
                if adjacency[i, j] > 0:
                    segments.append([positions[i], positions[j]])
                    coh = (1 + np.cos(phases[i] - phases[j])) / 2
                    line_colors.append((0, 1, 1, 0.05 + 0.3 * coh))
        
        if segments:
            lines = LineCollection(segments, colors=line_colors, linewidths=0.8)
            ax.add_collection(lines)
        
        # Points
        ax.scatter(positions[:, 0], positions[:, 1],
                   c=colors, s=100, alpha=0.9,
                   edgecolors='white', linewidth=1.5, zorder=10)
        
        # Labels
        if self.config.show_labels and sentences:
            n_labels = min(self.config.max_labels, len(sentences))
            for i in range(n_labels):
                label = sentences[i][:18] + '...' if len(sentences[i]) > 18 else sentences[i]
                offset_x = 0.03 if positions[i, 0] < 0 else -0.03
                ha = 'left' if positions[i, 0] < 0 else 'right'
                ax.annotate(label, (positions[i, 0], positions[i, 1]),
                           xytext=(offset_x, 0.02), textcoords='offset fontsize',
                           fontsize=7, color='white', alpha=0.7, ha=ha, va='bottom')
    
    def apply_fade(self, frame: np.ndarray, alpha: float) -> np.ndarray:
        """Apply fade effect."""
        bg_color = np.array([int(self.config.background_color[i:i+2], 16) 
                            for i in (1, 3, 5)])
        faded = (frame.astype(float) * alpha + 
                 bg_color.astype(float) * (1 - alpha))
        return faded.astype(np.uint8)
    
    def close(self):
        """Close the figure."""
        plt.close(self.fig)


# ==============================================================================
# VIDEO RECORDER
# ==============================================================================

class HAKORNVideoRecorder:
    """Records H-AKORN simulations as videos."""
    
    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self.renderer = None
        
        if not HAS_MPL:
            raise ImportError("matplotlib required for video recording")
        if self.config.format == 'mp4' and not HAS_CV2:
            import warnings
            warnings.warn("OpenCV not found, falling back to GIF")
            self.config.format = 'gif'
    
    def record(
        self,
        simulator,
        output_path: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Record simulation to video file."""
        output_path = output_path or self.config.output_path
        
        total_frames = int(self.config.fps * self.config.duration_seconds)
        steps_per_frame = self.config.timelapse_factor
        total_steps = total_frames * steps_per_frame
        
        mode_name = "3D Lorentz" if self.config.mode == 'lorentz' else "2D Poincaré"
        
        print(f"🎬 Recording H-AKORN Video ({mode_name})")
        print(f"   Resolution: {self.config.width}x{self.config.height}")
        print(f"   Duration: {self.config.duration_seconds}s @ {self.config.fps}fps")
        print(f"   Timelapse: {steps_per_frame}x")
        print(f"   Output: {output_path}")
        
        simulator.initialize()
        self.renderer = FrameRenderer(self.config)
        
        if self.config.format == 'gif':
            frames = []
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path, fourcc, self.config.fps,
                (self.config.width, self.config.height)
            )
        
        gamma_history = []
        
        start_time = time.time()
        
        for frame_idx in range(total_frames):
            simulator.step(steps_per_frame)
            
            pos_3d = simulator.to_lorentz_3d(simulator.positions).cpu().numpy()
            pos_2d = simulator.to_poincare(simulator.positions).cpu().numpy()
            phases = simulator.phases.cpu().numpy()
            adjacency = simulator.adjacency.cpu().numpy()
            gamma = simulator.compute_order_parameter()
            state = simulator.get_phase_state()
            
            gamma_history.append(gamma)
            
            frame = self.renderer.render_frame(
                positions_3d=pos_3d,
                positions_2d=pos_2d,
                phases=phases,
                adjacency=adjacency,
                gamma=gamma,
                gamma_history=gamma_history,
                step=simulator.step_count,
                total_steps=total_steps,
                sentences=getattr(simulator, 'sentences', None),
                state=state,
                frame_idx=frame_idx,
            )
            
            # Fades
            if frame_idx < self.config.fade_in_frames:
                alpha = frame_idx / self.config.fade_in_frames
                frame = self.renderer.apply_fade(frame, alpha)
            elif frame_idx > total_frames - self.config.fade_out_frames:
                remaining = total_frames - frame_idx
                alpha = remaining / self.config.fade_out_frames
                frame = self.renderer.apply_fade(frame, alpha)
            
            if self.config.format == 'gif':
                frames.append(Image.fromarray(frame))
            else:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            
            if progress_callback:
                progress_callback(frame_idx + 1, total_frames)
            elif (frame_idx + 1) % 30 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (frame_idx + 1) * (total_frames - frame_idx - 1)
                print(f"   Frame {frame_idx + 1}/{total_frames} (Γ={gamma:.3f}, ETA: {eta:.0f}s)")
        
        if self.config.format == 'gif':
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // self.config.fps,
                loop=0,
                optimize=True,
            )
        else:
            writer.release()
        
        self.renderer.close()
        
        elapsed = time.time() - start_time
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"\n✅ Video saved: {output_path}")
        print(f"   Size: {file_size:.1f} MB")
        print(f"   Time: {elapsed:.1f}s")
        
        return output_path


# ==============================================================================
# QUICK FUNCTIONS
# ==============================================================================

def record_hakorn_video(
    simulator,
    output_path: str = "hakorn_resonance.mp4",
    duration: float = 15.0,
    fps: int = 30,
    timelapse: int = 3,
    resolution: Tuple[int, int] = (1920, 1080),
    mode: str = 'lorentz',  # 'lorentz' or 'poincare'
    format: str = "mp4",
) -> str:
    """Quick function to record H-AKORN video."""
    config = VideoConfig(
        output_path=output_path,
        format=format,
        width=resolution[0],
        height=resolution[1],
        fps=fps,
        duration_seconds=duration,
        timelapse_factor=timelapse,
        mode=mode,
    )
    
    recorder = HAKORNVideoRecorder(config)
    return recorder.record(simulator, output_path)


def record_mteb_video(
    dataset_name: str = 'STSBenchmark',
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    max_samples: int = 35,
    output_path: str = "hakorn_sts.mp4",
    duration: float = 12.0,
    K: float = 3.5,
    mode: str = 'lorentz',  # 'lorentz' or 'poincare'
    device: str = "auto",
) -> str:
    """Record H-AKORN video from MTEB dataset."""
    from .hakorn_realtime import MTEBDataLoader, RealtimeHAKORN, RealtimeConfig
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    loader = MTEBDataLoader(device=device)
    data = loader.load_dataset(dataset_name, max_samples=max_samples)
    embeddings = loader.encode_sentences(data['sentences'], model_name)
    
    sim_config = RealtimeConfig(K=K, device=device, mode=mode, show_labels=True)
    simulator = RealtimeHAKORN(
        config=sim_config,
        sentences=data['sentences'],
        embeddings=embeddings,
        similarity_pairs=data['pairs'],
    )
    simulator.initialize()
    
    mode_suffix = "lorentz" if mode == 'lorentz' else "poincare"
    if output_path == "hakorn_sts.mp4":
        output_path = f"hakorn_sts_{mode_suffix}.mp4"
    
    video_config = VideoConfig(
        output_path=output_path,
        duration_seconds=duration,
        mode=mode,
        show_labels=True,
        max_labels=12,
        intro_text=f"H-AKORN: {dataset_name}",
    )
    
    recorder = HAKORNVideoRecorder(video_config)
    return recorder.record(simulator, output_path)


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'VideoConfig',
    'FrameRenderer',
    'HAKORNVideoRecorder',
    'record_hakorn_video',
    'record_mteb_video',
    'create_hyperboloid_mesh',
]
