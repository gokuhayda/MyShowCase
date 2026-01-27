# JavaScript API Reference

This document provides a complete reference for the UGFT Simulator's JavaScript API.

## Table of Contents

1. [Hyperbolic Geometry](#hyperbolic-geometry)
2. [Tessellation](#tessellation)
3. [Dynamics](#dynamics)
4. [Topology](#topology)
5. [UGFT Metrics](#ugft-metrics)
6. [Renderer](#renderer)
7. [Configuration](#configuration)

---

## 1. Hyperbolic Geometry

### `Hyperbolic` Object

Static utility methods for hyperbolic geometry operations.

#### `Hyperbolic.lorentzInner(u, v)`

Compute the Lorentz (Minkowski) inner product.

```javascript
const u = [1.5, 0.5, 0.5];
const v = [1.2, 0.3, 0.4];
const inner = Hyperbolic.lorentzInner(u, v);
// inner = -u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
```

**Parameters:**
- `u` (number[3]): First vector in ℝ^{2,1}
- `v` (number[3]): Second vector in ℝ^{2,1}

**Returns:** `number` - The Lorentz inner product ⟨u,v⟩_L

---

#### `Hyperbolic.toHyperboloid(p)`

Convert Poincaré disk coordinates to hyperboloid model.

```javascript
const poincare = [0.3, 0.4];
const hyperboloid = Hyperbolic.toHyperboloid(poincare);
// hyperboloid = [t, x, y] satisfying -t² + x² + y² = -1
```

**Parameters:**
- `p` (number[2]): Point in Poincaré disk [u, v] with u² + v² < 1

**Returns:** `number[3]` - Point on hyperboloid [t, x, y]

---

#### `Hyperbolic.toPoincare(h)`

Convert hyperboloid coordinates to Poincaré disk.

```javascript
const hyperboloid = [1.5, 0.6, 0.8];
const poincare = Hyperbolic.toPoincare(hyperboloid);
// poincare = [x/(1+t), y/(1+t)]
```

**Parameters:**
- `h` (number[3]): Point on hyperboloid [t, x, y]

**Returns:** `number[2]` - Point in Poincaré disk [u, v]

---

#### `Hyperbolic.poincareDistance(p, q)`

Compute hyperbolic distance between two points in Poincaré disk.

```javascript
const d = Hyperbolic.poincareDistance([0.1, 0.2], [0.5, 0.3]);
// d = acosh(1 + 2|p-q|²/((1-|p|²)(1-|q|²)))
```

**Parameters:**
- `p` (number[2]): First point in Poincaré disk
- `q` (number[2]): Second point in Poincaré disk

**Returns:** `number` - Hyperbolic distance

---

#### `Hyperbolic.geodesicDistance(p, q)`

Compute geodesic distance between points on hyperboloid.

```javascript
const p = [1.2, 0.3, 0.4];
const q = [1.5, 0.6, 0.8];
const d = Hyperbolic.geodesicDistance(p, q);
// d = acosh(-⟨p,q⟩_L)
```

**Parameters:**
- `p` (number[3]): First point on hyperboloid
- `q` (number[3]): Second point on hyperboloid

**Returns:** `number` - Geodesic distance

---

#### `Hyperbolic.geodesicArc(p, q)`

Compute parameters for drawing geodesic arc in Poincaré disk.

```javascript
const arc = Hyperbolic.geodesicArc([0.2, 0.3], [0.6, 0.1]);
// arc = { cx, cy, r, angle1, angle2 } or null if collinear with origin
```

**Parameters:**
- `p` (number[2]): First point in Poincaré disk
- `q` (number[2]): Second point in Poincaré disk

**Returns:** `Object|null`
- `cx` (number): Arc center x-coordinate
- `cy` (number): Arc center y-coordinate  
- `r` (number): Arc radius
- `angle1` (number): Start angle (radians)
- `angle2` (number): End angle (radians)

Returns `null` if points are collinear with origin (draw straight line).

---

## 2. Tessellation

### `generateTessellation(layers)`

Generate a {5,4} pentagrid tessellation of the hyperbolic plane.

```javascript
const cells = generateTessellation(5);
// cells.length ≈ 156 for 5 layers
```

**Parameters:**
- `layers` (number): Number of concentric layers (2-7 recommended)

**Returns:** `Cell[]` - Array of cell objects

**Cell Object Structure:**
```javascript
{
    id: number,           // Unique identifier (0 = center)
    position: number[3],  // Hyperboloid coordinates [t, x, y]
    poincare: number[2],  // Poincaré disk coordinates [u, v]
    phase: number,        // Initial phase θ ∈ [0, 2π)
    omega: number,        // Natural frequency ω
    layer: number,        // Tessellation layer
    neighbors: number[],  // Adjacent cell IDs
    localR: number        // Local order parameter (initialized to 0)
}
```

**Cell Count by Layer:**
| Layers | Approximate Cell Count |
|--------|----------------------|
| 2 | 11 |
| 3 | 26 |
| 4 | 56 |
| 5 | 156 |
| 6 | 312 |
| 7 | 632 |

---

## 3. Dynamics

### `runStep(cells, withFeedback)`

Execute one time step of H-AKORN dynamics.

```javascript
runStep(cells, true);  // With topological feedback
runStep(cells, false); // Without feedback (counterfactual)
```

**Parameters:**
- `cells` (Cell[]): Array of cell objects (modified in place)
- `withFeedback` (boolean): Enable topological downward causation

**Side Effects:**
- Updates `cell.phase` for all cells
- Updates `cell.localR` for all cells

**Algorithm:**
1. Compute global order parameter R
2. For each cell:
   - Sum hyperbolic-attention-weighted phase differences
   - Apply Kuramoto dynamics with optional feedback
   - Euler integration

---

### `computeOrderParameter(cells)`

Compute the Kuramoto order parameter R.

```javascript
const R = computeOrderParameter(cells);
// R ∈ [0, 1], where 0 = incoherent, 1 = synchronized
```

**Parameters:**
- `cells` (Cell[]): Array of cell objects

**Returns:** `number` - Order parameter R = |1/N Σ exp(iθ_k)|

---

### `computeMeanPhase(cells)`

Compute the mean phase Ψ of all oscillators.

```javascript
const psi = computeMeanPhase(cells);
// psi ∈ (-π, π]
```

**Parameters:**
- `cells` (Cell[]): Array of cell objects

**Returns:** `number` - Mean phase in radians

---

## 4. Topology

### `detectClusters(cells, threshold)`

Detect phase-coherent clusters via BFS.

```javascript
const clusters = detectClusters(cells, 0.3);
// clusters = [[0,1,3,5], [2,4,6], ...]  // Arrays of cell IDs
```

**Parameters:**
- `cells` (Cell[]): Array of cell objects
- `threshold` (number, optional): Maximum phase difference for clustering. Default: 0.3 radians

**Returns:** `number[][]` - Array of clusters, each containing cell IDs

**Clustering Criteria:**
Two cells are in the same cluster if:
1. They are adjacent in the tessellation graph
2. `min(|θ_i - θ_j|, 2π - |θ_i - θ_j|) < threshold`

---

### `computeBetti(cells, clusters)`

Estimate Betti numbers for topological analysis.

```javascript
const betti = computeBetti(cells, clusters);
// betti = { beta0: 3, beta1: 2, beta2: 0 }
```

**Parameters:**
- `cells` (Cell[]): Array of cell objects
- `clusters` (number[][]): Cluster array from `detectClusters`

**Returns:** `Object`
- `beta0` (number): Number of connected components
- `beta1` (number): Number of cycles (holes)
- `beta2` (number): Always 0 for 2D manifolds

---

## 5. UGFT Metrics

### `computeUGFTAction(cells)`

Compute the full UGFT action functional and system state.

```javascript
const action = computeUGFTAction(cells);
// action = {
//   L_task: 0.35,
//   L_geometry: 0.22,
//   L_topology: 0.15,
//   S_total: 0.72,
//   state: 'transition'
// }
```

**Parameters:**
- `cells` (Cell[]): Array of cell objects

**Returns:** `Object`
- `L_task` (number): Task loss (1 - R), clamped to [0, 1]
- `L_geometry` (number): Geometric frustration, clamped to [0, 1]
- `L_topology` (number): Topological complexity
- `S_total` (number): Total action S = L_task + L_geometry + L_topology
- `state` (string): System state classification

**State Classification:**
| State | Conditions |
|-------|------------|
| `'stable'` | R > 0.8 AND β₀ ≤ 2 |
| `'subcritical'` | R < 0.3 OR β₀ > 5 |
| `'transition'` | Otherwise |

---

## 6. Renderer

### `class Renderer`

Canvas 2D renderer with geodesic arc support.

#### Constructor

```javascript
const canvas = document.getElementById('mainCanvas');
const geoCanvas = document.getElementById('geodesicsCanvas');
const renderer = new Renderer(canvas, geoCanvas);
```

**Parameters:**
- `canvas` (HTMLCanvasElement): Main rendering canvas
- `geodesicsCanvas` (HTMLCanvasElement, optional): Overlay for geodesic lines

---

#### `renderer.resize(size)`

Resize the renderer canvases.

```javascript
renderer.resize(600);  // Set to 600x600 pixels
renderer.resize();     // Auto-size based on viewport
```

**Parameters:**
- `size` (number, optional): Canvas size in pixels. If omitted, auto-calculates.

---

#### `renderer.toScreen(poincare)`

Convert Poincaré disk coordinates to screen coordinates.

```javascript
const [sx, sy] = renderer.toScreen([0.3, -0.4]);
```

**Parameters:**
- `poincare` (number[2]): Point in Poincaré disk

**Returns:** `number[2]` - Screen coordinates [x, y]

---

#### `renderer.fromScreen(sx, sy)`

Convert screen coordinates to Poincaré disk coordinates.

```javascript
const [u, v] = renderer.fromScreen(350, 280);
```

**Parameters:**
- `sx` (number): Screen x-coordinate
- `sy` (number): Screen y-coordinate

**Returns:** `number[2]` - Poincaré disk coordinates

---

#### `renderer.render(cells, clusters, mode, highlightId)`

Render the current state to canvas.

```javascript
renderer.render(cells, clusters, 'phase', hoveredCellId);
```

**Parameters:**
- `cells` (Cell[]): Array of cell objects
- `clusters` (number[][]): Cluster array from `detectClusters`
- `mode` (string): Visualization mode
- `highlightId` (number|null, optional): Cell ID to highlight

**Visualization Modes:**
| Mode | Description |
|------|-------------|
| `'phase'` | Color by phase angle (HSL color wheel) |
| `'cluster'` | Color by cluster membership |
| `'attention'` | Color by local order parameter |
| `'geodesic'` | Phase coloring with geodesic arcs |

---

## 7. Configuration

### Global `config` Object

```javascript
const config = {
    K: 2.5,                    // Coupling strength
    dt: 0.05,                  // Time step (epsilon)
    kappa: -1.0,               // Hyperbolic curvature
    layers: 5,                 // Tessellation layers
    vizMode: 'phase',          // Visualization mode
    topologicalFeedback: true, // Enable feedback
    feedbackStrength: 0.1      // Feedback coefficient (alpha)
};
```

### Parameter Ranges

| Parameter | Min | Max | Default | Description |
|-----------|-----|-----|---------|-------------|
| `K` | 0 | 10 | 2.5 | Coupling strength |
| `dt` | 0.01 | 0.2 | 0.05 | Integration time step |
| `kappa` | -2 | -0.1 | -1.0 | Curvature (always negative) |
| `layers` | 2 | 7 | 5 | Tessellation depth |
| `feedbackStrength` | 0 | 1 | 0.1 | Feedback intensity |

---

## Usage Examples

### Basic Simulation Loop

```javascript
// Initialize
const cells = generateTessellation(5);
const renderer = new Renderer(canvas, geoCanvas);

// Animation loop
function animate() {
    runStep(cells, config.topologicalFeedback);
    
    const clusters = detectClusters(cells);
    renderer.render(cells, clusters, config.vizMode);
    
    const R = computeOrderParameter(cells);
    console.log(`Order parameter: ${R.toFixed(3)}`);
    
    requestAnimationFrame(animate);
}

animate();
```

### Counterfactual Comparison

```javascript
// Clone state
const cellsA = JSON.parse(JSON.stringify(cells));
const cellsB = JSON.parse(JSON.stringify(cells));

// Run 100 steps
for (let i = 0; i < 100; i++) {
    runStep(cellsA, true);   // With feedback
    runStep(cellsB, false);  // Without feedback
}

// Compare
const R_A = computeOrderParameter(cellsA);
const R_B = computeOrderParameter(cellsB);
console.log(`ΔR = ${(R_A - R_B).toFixed(3)}`);
```

### Custom Tessellation Analysis

```javascript
for (let layers = 2; layers <= 7; layers++) {
    const cells = generateTessellation(layers);
    
    let totalNeighbors = 0;
    for (const c of cells) {
        totalNeighbors += c.neighbors.length;
    }
    
    const avgDegree = totalNeighbors / cells.length;
    console.log(`Layers: ${layers}, Cells: ${cells.length}, Avg degree: ${avgDegree.toFixed(2)}`);
}
```

---

*Document version: 1.0.0 | Last updated: January 2026*
