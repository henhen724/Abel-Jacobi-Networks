# Gradient Computation in InverseAbelJacobiNetwork

## Current Implementation: Limited Gradient Flow

The current `InverseAbelJacobiNetwork` has **very limited gradient flow** due to the forward pass design:

### What Breaks Gradients

1. **Detaching input** (line 341-344):
   ```python
   u_np = u.detach().cpu().numpy()  # ← .detach() breaks gradient graph
   ```
   The input `u` is detached from the computation graph, so gradients cannot flow back through it.

2. **Numpy/mpmath computation** (lines 356-401):
   - All inverse Abel-Jacobi computation happens in numpy/mpmath
   - `inverse_abel_jacobi_newton()` uses mpmath for integration (not differentiable)
   - Newton's method iterations are not differentiable
   - Theta function evaluation uses numpy (not part of torch autograd)

3. **Conversion back to torch** (line 409):
   ```python
   torch.from_numpy(sigma.real).float()  # ← New tensor, no gradient connection
   ```

### What Gradients DO Flow

**Only through `self.coeffs`** (line 414):
```python
coeffs_out = coeffs_out + self.coeffs.unsqueeze(0)
```

This means:
- ✅ Gradients can flow **into** `self.coeffs` (the learnable parameters)
- ❌ Gradients **cannot** flow **through** the inverse Abel-Jacobi computation
- ❌ Gradients **cannot** flow **back** to the input `u`

### Gradient Flow Diagram

```
Input u (torch.Tensor)
    ↓ [.detach()] ← GRADIENT BREAK
    ↓
Numpy/mpmath computation:
  - inverse_abel_jacobi_newton()
  - mpmath integration
  - theta evaluation
    ↓
x_coords (numpy array)
    ↓
symmetric_polynomials()
    ↓
sigma (numpy array)
    ↓ [torch.from_numpy()] ← NEW TENSOR (no grad)
    ↓
coeffs_out (torch.Tensor, requires_grad=False)
    ↓ [+ self.coeffs] ← ONLY GRADIENT PATH
    ↓
Output (torch.Tensor)
```

## Implications

### What Works
- You can train `self.coeffs` to adjust the output coefficients
- The network can learn an additive correction to the inverse Abel-Jacobi result

### What Doesn't Work
- End-to-end training where gradients flow from loss → output → inverse AJ → input `u`
- Differentiable inverse Abel-Jacobi map (the core computation is non-differentiable)
- Gradient-based optimization of branch points or base point (they're fixed)

## Making It Differentiable

To enable full gradient flow, you would need:

1. **Replace mpmath with torch-compatible integration**:
   - Use `torch.quad` or approximate integrals with torch operations
   - Implement theta functions in PyTorch (already possible with `riemann_theta`)

2. **Differentiable Newton solver**:
   - Use implicit differentiation or unrolled iterations
   - Or use a learned approximation network instead of exact Newton

3. **Keep everything in torch**:
   - Don't detach inputs
   - Use torch operations throughout
   - Convert complex numbers to real pairs `(real, imag)` for torch compatibility

4. **Example differentiable approach**:
   ```python
   def forward(self, u: torch.Tensor) -> torch.Tensor:
       # Keep u in torch, don't detach
       u_complex = u[..., 0] + 1j * u[..., 1]  # Still breaks grad
       
       # Instead: work with real/imag pairs
       u_real = u[..., 0]  # shape (..., g)
       u_imag = u[..., 1]  # shape (..., g)
       
       # Implement inverse AJ using torch operations
       # (requires torch-compatible theta and integration)
   ```

## Current Use Case

The current implementation is suitable for:
- **Fixed inverse mapping**: Given `u`, compute approximate divisor coefficients
- **Learning corrections**: Train `self.coeffs` to refine the output
- **Non-differentiable pipeline**: When you don't need gradients through the inverse AJ step

If you need full differentiability, the architecture needs significant refactoring to use torch-native operations throughout.
