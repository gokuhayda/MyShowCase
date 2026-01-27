# Contributing to UGFT Simulator

Thank you for your interest in contributing to the UGFT Simulator! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## 📜 Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- **Be respectful** of differing viewpoints and experiences
- **Be constructive** in criticism and feedback
- **Focus on what is best** for the scientific community
- **Show empathy** towards other contributors

---

## 🚀 Getting Started

### Prerequisites

- Modern browser with WebGPU support (Chrome 113+, Edge 113+)
- Basic understanding of:
  - JavaScript/ES6+
  - HTML5 Canvas API
  - WebGPU (for compute shader contributions)
  - Hyperbolic geometry (for mathematical contributions)

### Local Development

```bash
# Clone the repository
git clone https://github.com/erickreis/ugft-simulator.git
cd ugft-simulator

# Start a local server (choose one)
python -m http.server 8000
# or
npx serve .

# Open http://localhost:8000 in your browser
```

---

## 🤝 How to Contribute

### Areas Where We Need Help

#### 🔴 High Priority
- [ ] **WebGPU Render Pipeline**: Eliminate CPU readback by rendering directly from GPU buffers
- [ ] **Performance Optimization**: Handle 100k+ cells at 60 FPS
- [ ] **Accessibility**: Screen reader support, keyboard navigation

#### 🟡 Medium Priority
- [ ] **Additional Tessellations**: Implement {7,3} heptagrid and {4,5} square tessellation
- [ ] **Lyapunov Exponents**: Add chaos/stability classification
- [ ] **Export Features**: Save/load configurations, export data as CSV/JSON
- [ ] **Mobile Support**: Touch interactions, responsive layout improvements

#### 🟢 Nice to Have
- [ ] **VR/WebXR**: Immersive visualization mode
- [ ] **Multi-language**: i18n support
- [ ] **Tutorials**: Interactive guided tours
- [ ] **Benchmarks**: Performance comparison suite

### Types of Contributions

1. **Bug Reports**: Found something broken? Open an issue!
2. **Feature Requests**: Have an idea? We'd love to hear it!
3. **Code Contributions**: Bug fixes, new features, optimizations
4. **Documentation**: Improve README, add examples, write tutorials
5. **Scientific Review**: Mathematical correctness, theoretical suggestions

---

## 💻 Development Setup

### Project Structure

```
ugft-simulator/
├── index.html              # Main entry point
├── versions/               # Alternative implementations
├── assets/                 # Images, GIFs, previews
├── docs/                   # Documentation
│   ├── THEORY.md          # Mathematical foundations
│   ├── ARCHITECTURE.md    # Technical details
│   └── API.md             # JavaScript API reference
└── tests/                  # Test files (future)
```

### Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| `Hyperbolic` | index.html | Geometry utilities (Lorentz, Poincaré) |
| `generateTessellation()` | index.html | Spatial-hashed pentagrid generation |
| `runStep()` | index.html | H-AKORN dynamics integration |
| `computeUGFTAction()` | index.html | Action functional computation |
| `Renderer` | index.html | Canvas 2D rendering with geodesics |

---

## 📝 Coding Standards

### JavaScript Style

```javascript
// Use const/let, never var
const CONSTANT_VALUE = 42;
let mutableVariable = 0;

// Use descriptive names
function computeKuramotoOrderParameter(cells) { ... }

// Document complex functions
/**
 * Compute geodesic arc parameters in Poincaré disk
 * @param {number[]} p - First point [x, y]
 * @param {number[]} q - Second point [x, y]
 * @returns {Object|null} Arc parameters {cx, cy, r, angle1, angle2}
 */
function geodesicArc(p, q) { ... }

// Use early returns
function processCell(cell) {
    if (!cell) return null;
    if (cell.neighbors.length === 0) return cell;
    // ... main logic
}
```

### WGSL Shader Style (for WebGPU contributions)

```wgsl
// Use descriptive struct names
struct Cell {
    position: vec3<f32>,
    phase: f32,
    // ...
};

// Comment complex operations
fn lorentzInner(u: vec3<f32>, v: vec3<f32>) -> f32 {
    // Minkowski metric: η = diag(-1, 1, 1)
    return -u.x * v.x + u.y * v.y + u.z * v.z;
}
```

### CSS Guidelines

- Use CSS custom properties (variables) for theming
- Follow BEM-like naming for complex components
- Mobile-first responsive design

---

## 🔄 Pull Request Process

### Before Submitting

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/amazing-feature`
3. **Make your changes** following coding standards
4. **Test thoroughly** in multiple browsers
5. **Update documentation** if needed

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-reviewed my own code
- [ ] Commented complex/non-obvious code
- [ ] Made corresponding documentation changes
- [ ] No new warnings in browser console
- [ ] Tested in Chrome and at least one other browser

### Commit Messages

Use conventional commits:

```
feat: add {7,3} heptagrid tessellation
fix: correct geodesic arc calculation near origin
docs: add API reference for Renderer class
perf: optimize spatial hashing for large cell counts
refactor: extract topology computation to separate module
```

### Review Process

1. Submit PR with clear description of changes
2. Maintainers will review within 1-2 weeks
3. Address any requested changes
4. Once approved, maintainers will merge

---

## 🐛 Reporting Issues

### Bug Reports

Please include:

1. **Browser & version** (e.g., Chrome 120)
2. **Operating system** (e.g., Windows 11, macOS 14)
3. **Steps to reproduce**
4. **Expected behavior**
5. **Actual behavior**
6. **Screenshots** if applicable
7. **Console errors** (F12 → Console tab)

### Feature Requests

Please include:

1. **Clear description** of the feature
2. **Use case** - why is this needed?
3. **Proposed implementation** (optional)
4. **Alternatives considered** (optional)

---

## 🔬 Scientific Contributions

For contributions involving mathematical or theoretical changes:

1. **Document the theory** in `docs/THEORY.md`
2. **Provide references** to relevant papers
3. **Include derivations** for non-trivial equations
4. **Add visual explanations** where helpful

### Mathematical Notation

Use LaTeX in markdown:

```markdown
The Kuramoto order parameter is defined as:

$$R = \left|\frac{1}{N}\sum_{k=1}^{N} e^{i\theta_k}\right|$$
```

---

## 📫 Contact

- **GitHub Issues**: For bugs and feature requests
- **Email**: eireikreisena@gmail.com (for sensitive matters)

---

## 🙏 Recognition

All contributors will be acknowledged in:

- README.md contributors section
- Release notes
- Academic publications using this software (where appropriate)

---

Thank you for contributing to open science! 🧠✨
