# wMPS

Wavelet matrix product states for continuum quantum field theories, built on [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl).

A wavelet MPS (wMPS) is a matrix product state whose sites are the modes of a sufficiently regular (N ≥ 6) Daubechies scaling function basis at resolution r. The resulting states live in the continuum Fock space with finite energy density, and can be optimized with standard MPS algorithms (VUMPS, DMRG, Grassmann gradient descent). Because wavelets form a multi-resolution analysis, a state at resolution r can be refined into a state at r+1 with a small quantum circuit, so that fine length scales can be reached iteratively. The package implements this for the Lieb-Liniger model, see [the paper](https://arxiv.org/abs/2606.23823).

The package includes:

- MPO Hamiltonians for the Lieb-Liniger model in the Daubechies D6 and D8 wavelet bases (`LL_D6`, `LL_D8`), exact minimal-bond-dimension versions of these (`LL_D6_minimal`, `LL_D8_minimal`, `LL_wavelet_minimal`), the D6 model in a harmonic trap V(x) = ½ω²(x - x₀)² on a finite chain (`LL_D6_trap`, `LL_D6_trap_minimal`), and a quadratic test model (`quad_model`, `quad_model_D8`)
- MPS refinement from one resolution to the next (`MPSrefine`, `single_site`, `refinement_alg`, `refine_optim`), and its finite chain version (`MPSrefine_finite`)
- Correlation functions: the one-body density matrix ⟨ψ⁺(x)ψ(0)⟩ of an infinite state and the density of a finite chain state (`continuum_density`), and the density-density correlation function ⟨ψ⁺(x)ψ⁺(0)ψ(x)ψ(0)⟩ of an infinite state (`nn_correlator`)
- Ground states at fixed particle number on a finite chain, refined from one resolution to the next (`finite_groundstates`), and the Tonks-Girardeau limit to compare them with (`box_levels`, `tg_box_density`, `trap_levels`, `tg_trap_density`)
- Versions of the VUMPS and GradientGrassmann ground-state solvers that record timing (`find_groundstate_t`, `add_histories`, `add_histories_kf`)

## Installation

```julia
] dev /path/to/wMPS
```

wMPS uses the `dense-blas` branch of [AntoineTilloy/MPSKit.jl](https://github.com/AntoineTilloy/MPSKit.jl/tree/dense-blas), which makes contractions with large MPOs much faster. With Julia ≥ 1.11 the package's own environment and `examples/` pick it up from their `[sources]` entries. In any other environment where you `dev` wMPS, or on Julia 1.10, add it there with `] add https://github.com/AntoineTilloy/MPSKit.jl.git#dense-blas`.

## The model

At resolution r, with ψₙ the annihilation operator of the scaling function mode at site n, the wavelet Lieb-Liniger Hamiltonian is

```
H = 2³ʳ Σₙₘ ψₙ⁺ Kₙₘ ψₘ  +  2²ʳ c Σₙₘₗₖ Γ₄[m-n, l-n, k-n] ψₙ⁺ ψₘ⁺ ψₗ ψₖ  -  2ʳ μ Σₙ ψₙ⁺ ψₙ
```

This is 2ʳ times the continuum Hamiltonian ∫ [∂ₓψ⁺∂ₓψ + c ψ⁺ψ⁺ψψ - μ ψ⁺ψ] dx, restricted to the span of the scaling functions. The overall 2ʳ compensates for the lattice spacing, so `expectation_value(ψ, H)` is directly the continuum energy density, and can be compared across resolutions. K are the kinetic coefficients and Γ₄ the four-point overlaps of the scaling functions (stored in `data/`).

All model constructors take the keyword arguments

| keyword  | meaning                                               | default |
|----------|-------------------------------------------------------|---------|
| `μ`      | chemical potential                                    | 2.0     |
| `c`      | interaction strength                                  | 1.0     |
| `r`      | resolution.                                           | 0       |
| `cutoff` | maximum number of bosons per site (physical dimension `cutoff + 1`) | 2 |

and the positional arguments `(elt=ComplexF64, symmetry=Trivial, lattice=InfiniteChain(1))`, as in MPSKitModels. `lattice` can be an `InfiniteChain(1)` or a `FiniteChain(L)`. `symmetry` can be `Trivial` or, since the Lieb-Liniger Hamiltonian conserves the particle number, `U1Irrep` (not for `quad_model`). With `U1Irrep` the physical space is `U1Space(n => 1 for n in 0:cutoff)`, and on a finite chain DMRG can look for the ground state directly in the sector with N particles, e.g. starting from `FiniteMPS(L, P, V; right=U1Space(N => 1))` (see `examples/finite_chain.jl`). The minimal MPOs have the same bond dimension with either symmetry. On an `InfiniteChain` the U(1) MPO can be built, but VUMPS at a fixed density would need a unit cell with a shifted charge, which is not implemented here.

### Which constructor to use

| wavelet | `@mpoham` MPO | bond dimension | exact minimal MPO | bond dimension |
|---------|---------------|----------------|-------------------|----------------|
| D6      | `LL_D6`       | 562            | `LL_D6_minimal`   | 45             |
| D8      | `LL_D8`       | 2466           | `LL_D8_minimal`   | 84             |

The minimal MPOs are the same operator, reproduced to floating point round-off, and are much faster both to build and to optimize with (VUMPS is about 9× faster for D6). `LL_D6` and `LL_D8` build the MPO term by term and are kept as an independent reference and to reproduce the original computations; building them is quick, but optimizing with them is slow, `LL_D8` in particular. `LL_wavelet_minimal(K, Γ4, ...)` builds the same Hamiltonian for any wavelet order, given its coefficients.

**Reporting energies.** To keep every reported energy exactly variational, the examples optimize with the minimal MPOs (VUMPS, DMRG, GradientGrassmann) but always evaluate the final energies with the corresponding `@mpoham` MPO, which is built term by term and does not depend on the compression used to build the minimal one. `exact_model(model)` gives that constructor (`exact_model(LL_D6_minimal) === LL_D6`), and `refinement_alg` and `refine_optim` use it for the energies they print. Energies recorded *during* an optimization, e.g. in the `history` returned by `find_groundstate_t`, are those of the MPO being optimized. With the `dense-blas` branch of [AntoineTilloy/MPSKit.jl](https://github.com/AntoineTilloy/MPSKit.jl/tree/dense-blas), evaluating one energy at bond dimension 12 takes about 0.5 s with `LL_D6` and 12 s with `LL_D8`; with the registered MPSKit, `LL_D8` takes over half an hour.

## Usage

```julia
using wMPS, MPSKit, TensorKit, OptimKit

μ, c, dmax = 1.0, 8.0, 2

# ground state at resolution r = 2
H = LL_D6_minimal(; μ, c, r=2, cutoff=dmax)
ψ, = find_groundstate(InfiniteMPS(ℂ^(dmax + 1), ℂ^12), H, VUMPS(; tol=1e-6))
E = real(expectation_value(ψ, LL_D6(; μ, c, r=2, cutoff=dmax)))     # energy density, with the exact MPO

# correlation functions of a state at resolution 2
xs, g1 = continuum_density(ψ, 2, 10.0)             # ⟨ψ⁺(x)ψ(0)⟩ at x = t / 2^(r+J), 0 ≤ x ≤ 10 (J = 5 by default)
xs, g2 = nn_correlator(ψ, 2, 10.0)                 # ⟨ψ⁺(x)ψ⁺(0)ψ(x)ψ(0)⟩ on the same grid

# refine from r = 2 to r = 5, with 100 L-BFGS iterations at each new resolution
states, history = refine_optim(ψ, 2, 3, μ, c, dmax, dmax + 1, 12, 100; model=LL_D6_minimal)
#                              ψ  r_start n_scale  dmax  d       D  L-BFGS iterations
```

### When to refine

At low resolution, optimizing directly at the target r with VUMPS or L-BFGS is fast, and refinement brings nothing. It pays off at high resolution (r ≈ 4-5 and above), where direct optimization from a random state becomes slow. Even there, it is better to run some optimization in between refinement steps, as `refine_optim` does, rather than to refine repeatedly with `refinement_alg` alone.

### Finite chains

`MPSrefine_finite(ψ, D)` refines a `FiniteMPS` on the L sites of a `FiniteChain(L)` at resolution r into one on the 2L sites of a `FiniteChain(2L)` at resolution r+1, the chains used by the finite chain models. A translation invariant state is not needed, so there is no `single_site` step. The circuit sends the scaling mode n onto the modes 2n, …, 2n+5 at the next resolution, so the two modes that fall past the right end of the chain are projected onto the vacuum. It returns `(ψf, p)`, with p the probability kept by that projection, which is close to 1 when the density at the wall is small. The circuit conserves the particle number, and works with `U1Irrep` states. Since the space at resolution r is contained in that at r+1, the refined state has the same continuum energy, up to that projection, and is a good starting point for DMRG at r+1.

`finite_groundstates(model, l, N, rs; D, cutoff, kwargs...)` does this for a box of length l at the resolutions r in `rs`: it looks for the ground state with N particles with DMRG, starting from a random state at the lowest resolution, and from the refined previous ground state at each following one. It returns, for each r, the state, the occupations of the modes and the continuum energy (evaluated with `exact_model(model)`).

`continuum_density(ψ, r; J)` gives the density ρ(x) = ⟨ψ⁺(x)ψ(x)⟩ of a `FiniteMPS` at resolution r, on the grid x = t / 2^(r+J) over the whole chain, from the correlations ⟨aₙ⁺aₘ⟩ of the modes that overlap.

### Return values

- `find_groundstate_t(ψ, H, ::VUMPS)` returns `(ψ, envs, ϵ, history)`, and `find_groundstate_t(ψ, H, ::GradientGrassmann)` returns `(ψ, envs, history)`. The GradientGrassmann version only supports `OptimKit.LBFGS` as its method.
- A `history` array has one row per iteration and the columns (wall time in s, energy density, error), where the error is the Galerkin error for VUMPS and the gradient norm for L-BFGS. Histories of successive runs can be concatenated with `add_histories` (drops the first row of the second history) or `add_histories_kf` (keeps it).
- `refine_optim` returns `(states, history)`: `states[i, :, :, :]` is the AL tensor at resolution `r_start + i`, and `history` is the combined history of all L-BFGS runs, with the refinement time included. The energies it prints at the end of each resolution are evaluated with `exact_model(model)`, while those in `history` are recorded with `model` during the optimization.

### Limitations

- The refinement circuit (`MPSrefine`, `MPSrefine_finite`, `refinement_alg`, `refine_optim`) is only implemented for D6 wavelets, so `model` should be a D6 model.
- `continuum_density` and `nn_correlator` are only valid for D6 states, whose scaling function they compute exactly on a dyadic grid (`wMPS.scaling_function`). `nn_correlator` only takes states with `Trivial` symmetry.

## Examples

`examples/` has its own environment with plotting packages, which finds wMPS through a `[sources]` entry and so needs Julia ≥ 1.11. From the package root,

```julia
] activate examples
] instantiate
```

then open the worksheet `examples/wMPS_worksheet.ipynb`, which goes through ground state optimization, correlators and refinement, or run one of the scripts:

| script                          | contents                                                        |
|---------------------------------|-----------------------------------------------------------------|
| `examples/groundstate.jl`       | D6 and D8 ground states, compared with the exact energy          |
| `examples/finite_chain.jl`      | DMRG in an infinite well and a harmonic trap at fixed N (U(1)), refining between resolutions, compared with the Tonks-Girardeau limit |
| `examples/custom_wavelet.jl`    | building the minimal MPO from your own coefficients              |
| `examples/timing_histories.jl`  | convergence history of VUMPS followed by L-BFGS, versus time     |

e.g. `julia --project=examples examples/groundstate.jl`.

## Tests

```julia
] test wMPS
```

The tests that build `LL_D8` (bond dimension 2466) are slow and skipped by default; run them with

```
WMPS_SLOW_TESTS=true julia --project -e 'using Pkg; Pkg.test()'
```

## Data

`data/` holds the scaling function coefficients that are not known in closed form, as HDF5 files:

| file                     | dataset    | contents                                      |
|--------------------------|------------|-----------------------------------------------|
| `gamma4_D6_save_arr.h5`  | `Γ4arr`    | four-point overlaps Γ₄ for D6, 9 × 9 × 9      |
| `gamma4_D8_save_arr.h5`  | `Γ4_D8arr` | four-point overlaps Γ₄ for D8, 13 × 13 × 13   |
| `K_D8_save_arr.h5`       | `K8arr`    | kinetic coefficients K for D8, offsets 0 to 6 |

The D6 kinetic coefficients are rational and defined in `src/models.jl`.

## Citation

If you use this package, please cite

> M. Kaplan and A. Tilloy, *Wavelet Matrix Product States for Quantum Fields*, [arXiv:2606.23823](https://arxiv.org/abs/2606.23823) (2026).

```bibtex
@misc{kaplan2026wavelet,
  title         = {Wavelet Matrix Product States for Quantum Fields},
  author        = {Kaplan, Molly and Tilloy, Antoine},
  year          = {2026},
  eprint        = {2606.23823},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2606.23823}
}
```
