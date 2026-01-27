# Technical Architecture

This document describes the technical implementation of the UGFT Simulator.

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Structures](#data-structures)
3. [Algorithms](#algorithms)
4. [Rendering Pipeline](#rendering-pipeline)
5. [Performance Considerations](#performance-considerations)
6. [WebGPU Integration](#webgpu-integration)
7. [Future Improvements](#future-improvements)

---

## 1. System Overview

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        UGFT Simulator                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Config    │  │    State    │  │        Metrics          │ │
│  │  K, ε, κ    │  │   cells[]   │  │  R, Φ, β₀, β₁, β₂     │ │
│  │  layers     │  │  stepCount  │  │  clusters, histogram    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Core Modules                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Hyperbolic  │  │  Dynamics   │  │      Topology           │ │
│  │  Geometry   │  │  (H-AKORN)  │  │  (Clusters, Betti)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Rendering                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Canvas 2D Renderer (with geodesic arcs)                    ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ ││
│  │  │  Cells      │  │  Geodesics  │  │  Overlays (tooltip) │ ││
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                         UI Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Controls   │  │   Charts    │  │    UGFT Panel           │ │
│  │  (sliders)  │  │   (R(t))    │  │    (S, L_i, state)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Core Logic | Vanilla JavaScript ES6+ | Zero dependencies |
| Rendering | Canvas 2D API | Wide browser support |
| Styling | CSS Custom Properties | Theming, animations |
| Future | WebGPU Compute | Parallel dynamics |

---

## 2. Data Structures

### 2.1 Cell Object

```javascript
{
    id: number,              // Unique identifier
    position: [t, x, y],     // Hyperboloid coordinates (Lorentz model)
    poincare: [u, v],        // Poincaré disk projection
    phase: number,           // θ ∈ [0, 2π)
    omega: number,           // Natural frequency ω
    layer: number,           // Tessellation layer (0 = center)
    neighbors: number[],     // Adjacent cell IDs
    localR: number           // Local order parameter (cached)
}
```

### 2.2 Configuration Object

```javascript
const config = {
    K: 2.5,          // Coupling strength
    dt: 0.05,        // Time step (ε)
    kappa: -1.0,     // Hyperbolic curvature
    layers: 5,       // Tessellation depth
    vizMode: 'phase', // Visualization mode
    topologicalFeedback: true,  // Enable downward causation
    feedbackStrength: 0.1       // Feedback coefficient (α)
};
```

### 2.3 Spatial Hash Grid

For O(1) neighbor lookup during tessellation:

```javascript
// Grid structure
Map<string, Cell[]>

// Key format: "gx,gy" where
gx = floor(poincare[0] / gridSize)
gy = floor(poincare[1] / gridSize)
```

---

## 3. Algorithms

### 3.1 Tessellation Generation

```
generateTessellation(layers):
    cells = []
    grid = new Map()  // Spatial hash
    
    // 1. Generate center cell
    cells.push(createCell(id=0, position=[1,0,0], layer=0))
    addToGrid(grid, cells[0])
    
    // 2. Generate concentric layers
    for layer in 1..layers:
        count = 5 * 2^(layer-1)  // Exponential growth
        offset = (layer % 2) * π/count  // Stagger alternating layers
        
        for i in 0..count:
            angle = 2πi/count + offset
            r = tanh(layer * 0.4)  // Poincaré radius
            poincare = [r*cos(angle), r*sin(angle)]
            position = poincareToHyperboloid(poincare)
            
            cell = createCell(nextId++, position, layer)
            cells.push(cell)
            addToGrid(grid, cell)
    
    // 3. Build adjacency using spatial hash
    threshold = 1.2  // Hyperbolic distance threshold
    for cell in cells:
        gx, gy = getGridCoords(cell)
        for dx in -1..1:
            for dy in -1..1:
                bucket = grid.get(key(gx+dx, gy+dy))
                for other in bucket:
                    if other.id > cell.id:
                        d = poincareDistance(cell, other)
                        if d < threshold:
                            cell.neighbors.push(other.id)
                            other.neighbors.push(cell.id)
    
    return cells
```

**Complexity**: O(N) average case (with spatial hashing), vs O(N²) naive approach.

### 3.2 H-AKORN Step

```
runStep(cells, withFeedback):
    K = config.K
    dt = config.dt
    kappa = config.kappa
    
    // Compute global order (for feedback)
    R = computeOrderParameter(cells)
    
    newPhases = []
    
    for i in 0..N:
        cell = cells[i]
        coupling = 0
        localCos = 0, localSin = 0
        
        // Sum over neighbors with hyperbolic attention
        for j in cell.neighbors:
            neighbor = cells[j]
            d = geodesicDistance(cell.position, neighbor.position)
            A_ij = exp(-d * |kappa|)  // Attention weight
            coupling += A_ij * sin(neighbor.phase - cell.phase)
            
            // Track local coherence
            localCos += cos(neighbor.phase)
            localSin += sin(neighbor.phase)
        
        // Update local R
        cell.localR = sqrt(localCos² + localSin²) / |neighbors|
        
        // Kuramoto dynamics
        dtheta = cell.omega + (K / |neighbors|) * coupling
        
        // Optional topological feedback
        if withFeedback:
            dtheta *= (1 + α * (R - 0.5))
        
        // Euler integration
        newPhase = (cell.phase + dt * dtheta) mod 2π
        newPhases.push(newPhase)
    
    // Apply updates
    for i in 0..N:
        cells[i].phase = newPhases[i]
```

**Complexity**: O(N × k) where k = average neighbor count ≈ 4.

### 3.3 Cluster Detection

```
detectClusters(cells, threshold=0.3):
    clusters = []
    visited = Set()
    
    for i in 0..N:
        if i in visited: continue
        
        cluster = [i]
        visited.add(i)
        queue = [i]
        
        // BFS for phase-coherent neighbors
        while queue not empty:
            curr = queue.pop()
            for n in cells[curr].neighbors:
                if n in visited: continue
                
                phaseDiff = |cells[curr].phase - cells[n].phase|
                minDiff = min(phaseDiff, 2π - phaseDiff)
                
                if minDiff < threshold:
                    visited.add(n)
                    cluster.push(n)
                    queue.push(n)
        
        clusters.push(cluster)
    
    return clusters
```

**Complexity**: O(N + E) where E = number of edges.

### 3.4 Geodesic Arc Calculation

```
geodesicArc(p, q):
    // Check for near-collinearity with origin
    cross = p[0]*q[1] - p[1]*q[0]
    if |cross| < 0.001:
        return null  // Draw straight line
    
    // Find orthogonal circle through p and q
    px, py = p
    qx, qy = q
    
    d = 2 * (px*qy - qx*py)
    if |d| < 0.0001: return null
    
    p2 = px² + py²
    q2 = qx² + qy²
    
    cx = ((p2 + 1)*qy - (q2 + 1)*py) / d
    cy = ((q2 + 1)*px - (p2 + 1)*qx) / d
    r = sqrt((cx - px)² + (cy - py)²)
    
    angle1 = atan2(py - cy, px - cx)
    angle2 = atan2(qy - cy, qx - cx)
    
    return {cx, cy, r, angle1, angle2}
```

---

## 4. Rendering Pipeline

### 4.1 Frame Rendering Order

```
render(cells, clusters, mode, highlightId):
    1. Clear canvas (dark background)
    2. Draw disk glow (radial gradient)
    3. Draw boundary circle
    4. Draw geodesic arcs (on separate canvas layer)
    5. Draw cell glows (larger, transparent)
    6. Draw cell bodies (smaller, opaque)
    7. Draw highlight effects (if hovering)
```

### 4.2 Color Mapping

| Mode | Color Function |
|------|----------------|
| Phase | `hsl(θ/2π × 360, 85%, 58%)` |
| Cluster | `hsl(idx × 137.5 mod 360, 70%, 55%)` |
| Attention | `rgb(localR×255, localR×100, 255-localR×255)` |

### 4.3 Coordinate Transformation

```javascript
class Renderer {
    toScreen(poincare) {
        return [
            this.cx + poincare[0] * this.radius,
            this.cy - poincare[1] * this.radius  // Y inverted
        ];
    }
    
    fromScreen(sx, sy) {
        return [
            (sx - this.cx) / this.radius,
            -(sy - this.cy) / this.radius
        ];
    }
}
```

---

## 5. Performance Considerations

### 5.1 Current Bottlenecks

| Operation | Complexity | Time (1000 cells) |
|-----------|------------|-------------------|
| Tessellation | O(N) | ~50ms (once) |
| Dynamics step | O(N×k) | ~2ms |
| Cluster detection | O(N+E) | ~1ms |
| Rendering | O(N+E) | ~5ms |
| **Total frame** | - | **~8ms** (~120 FPS) |

### 5.2 Optimization Techniques Used

1. **Spatial hashing**: O(N) adjacency building
2. **Cached local R**: Avoid recomputation
3. **Topology update throttling**: Every 8 frames
4. **Histogram throttling**: Every 15 frames
5. **Separate geodesics canvas**: Avoid full redraw

### 5.3 Scaling Limits

| Cells | FPS (Canvas 2D) | FPS (WebGPU target) |
|-------|-----------------|---------------------|
| 500 | 120 | 120 |
| 2,000 | 60 | 120 |
| 10,000 | 15 | 60+ |
| 100,000 | <1 | 30+ |

---

## 6. WebGPU Integration

### 6.1 Compute Shader (Available in versions/webgpu-compute.html)

```wgsl
struct Cell {
    position: vec3<f32>,
    poincare: vec2<f32>,
    phase: f32,
    omega: f32,
    magnitude: f32,
    layer: u32,
    neighborStart: u32,
    neighborCount: u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.N) { return; }
    
    let cell = cellsIn[i];
    var coupling: f32 = 0.0;
    
    for (var j: u32 = 0u; j < cell.neighborCount; j++) {
        let neighborIdx = adjacency[cell.neighborStart + j];
        let neighbor = cellsIn[neighborIdx];
        
        let d = geodesicDist(cell.position, neighbor.position);
        let A_ij = exp(-d * abs(params.kappa));
        coupling += A_ij * sin(neighbor.phase - cell.phase);
    }
    
    let dtheta = cell.omega + (params.K / f32(cell.neighborCount)) * coupling;
    let newPhase = (cell.phase + params.dt * dtheta) % TAU;
    
    cellsOut[i].phase = newPhase;
}
```

### 6.2 Future: Render Pipeline

To eliminate CPU readback, implement a vertex/fragment shader pipeline:

```wgsl
@vertex
fn vs_main(@builtin(instance_index) i: u32) -> VertexOutput {
    let cell = cells[i];
    // Create billboard quad at cell.poincare
    // Pass phase to fragment for coloring
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Draw circle using SDF
    let dist = length(in.uv);
    if (dist > 1.0) { discard; }
    
    // Phase-based color
    let hue = in.phase / TAU;
    return hsv_to_rgb(hue, 0.85, 0.6);
}
```

---

## 7. Future Improvements

### 7.1 Planned Features

| Feature | Priority | Complexity | Status |
|---------|----------|------------|--------|
| WebGPU render pipeline | High | High | Designed |
| Lyapunov exponents | Medium | Medium | Planned |
| {7,3}, {4,5} tessellations | Medium | Low | Planned |
| Export/import state | Low | Low | Planned |
| WebXR visualization | Low | High | Concept |

### 7.2 Architecture Refactoring

Potential module extraction:

```
src/
├── geometry/
│   ├── hyperbolic.js      # Lorentz, Poincaré operations
│   └── tessellation.js    # Pentagrid, heptagrid generators
├── dynamics/
│   ├── kuramoto.js        # H-AKORN step function
│   └── topology.js        # Clusters, Betti numbers
├── rendering/
│   ├── canvas-renderer.js # Current 2D renderer
│   └── webgpu-renderer.js # Future GPU renderer
├── ui/
│   ├── controls.js        # Sliders, buttons
│   └── charts.js          # R(t) plot, histogram
└── main.js                # Entry point, orchestration
```

---

## Appendix: Browser Compatibility

### WebGPU Support Matrix

| Browser | Version | Compute | Render | Notes |
|---------|---------|---------|--------|-------|
| Chrome | 113+ | ✅ | ✅ | Full support |
| Edge | 113+ | ✅ | ✅ | Chromium-based |
| Firefox | Nightly | ⚠️ | ⚠️ | Flag required |
| Safari | 18+ | ⚠️ | ⚠️ | Limited |

### Fallback Strategy

```javascript
async function init() {
    if (navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
            return initWebGPU(adapter);
        }
    }
    return initCanvas2DFallback();
}
```

---

*Document version: 1.0.0 | Last updated: January 2026*
