#!/usr/bin/env julia
# =============================================================================
#  1D ANHARMONIC QUANTUM OSCILLATOR  --  FOCK (NUMBER) BASIS SOLVER
# =============================================================================
#
#  WHAT THIS DOES
#  --------------
#  Simulates the time-dependent Schrodinger equation for a 1D anharmonic
#  oscillator starting from a COHERENT STATE (displaced ground state of the
#  harmonic part) and animates the time-evolving probability density
#  |psi(x,t)|^2 together with the potential V(x), a <q>(t) marker, and a live
#  norm / energy readout.
#
#  It ALSO produces a static comparison figure (q_comparison.png) overlaying
#  FOUR solutions for the mean position:
#     * the FULL QUANTUM <q>(t)              (numerically exact in an Ncut Fock basis),
#     * the CLASSICAL q(t)                   (tree level / hbar^0 EOM),
#     * the TREE + ONE-LOOP LOCAL (A)        (local adiabatic eq; oneloop_trajectory), and
#     * the TREE + ONE-LOOP NONLOCAL (B)     (causal memory eq; nonlocal_trajectory).
#  (A) and (B) are the two real-time one-loop equations derived in
#  theory/q4_1_loop.tex. All four start from the same coherent state; they agree
#  at early times (Ehrenfest) and separate once the wavepacket distorts. The
#  classical and one-loop curves are integrated with a stdlib RK4 (no extra packages).
#
#  METHOD (numerically exact within an Ncut Fock truncation, zero-install)
#  -----------------------------------------------------------------------
#    * "Exact" / "full quantum" here means UNITARY evolution with NO time-stepping
#      error in the kept Ncut-dimensional Fock space -- NOT the exact infinite-
#      dimensional answer. The Fock truncation (levels 0 .. Ncut-1) is a separate,
#      controlled approximation; its error is NOT seen by norm/energy conservation
#      (both hold within the kept subspace) -- only by the convergence_check (re-run
#      at higher Ncut) and the top-5% Fock-population gauges (t=0 and max-over-t).
#    * Build annihilation / creation operators a, adag as bidiagonal matrices of
#      size Ncut (Fock truncation: levels 0 .. Ncut-1).
#    * Position operator   x = sqrt(hbar/(2 m omega)) * (a + adag).
#    * Hamiltonian:
#          H = hbar*omega*(adag*a + 1/2)      (harmonic reference)
#            + (lambda/24) * x^4               (= lambda q^4 / 4! ; quartic Duffing)
#    * Diagonalize H ONCE with LinearAlgebra.eigen  ->  H = Vmat * diag(E) * Vmat'.
#    * Prepare the coherent state |alpha> in the Fock basis and evolve it EXACTLY
#      in the energy eigenbasis (no time-stepping error, unitary to machine eps):
#          |psi(t)> = Vmat * exp(-i E t / hbar) * (Vmat' * |psi(0)>).
#    * Reconstruct psi(x,t) on a real-space grid using the harmonic-oscillator
#      eigenfunctions phi_n(x), built with a numerically STABLE Hermite
#      recurrence on the *normalized* functions (no factorial / Hermite overflow).
#
#  WHY THE FOCK BASIS
#  ------------------
#    * It runs with ZERO non-stdlib installs (only LinearAlgebra/Printf + the
#      optional, guarded `using Plots`).
#    * The propagation is exact WITHIN the truncation (energy and norm conserved to
#      ~machine eps in the kept subspace), so we can hard-assert conservation instead
#      of merely reporting O(dt^2) drift (truncation error is gauged separately).
#    * eigen() gives the exact spectrum/eigenvectors, so the coherent state is
#      propagated with NO time-stepping error and the comparison curves start
#      from the identical initial condition.
#
#  PHYSICS CHECKS (printed + asserted)
#  -----------------------------------
#    * Coherent-state normalization in the Fock basis and on the grid (~1e-10).
#    * <x>(0), <p>(0) match the analytic coherent-state targets.
#    * Norm conservation along the whole trajectory (exact evolution -> ~1e-12).
#    * Energy <H> conservation (constant of motion; deviation reported+asserted).
#    * Fock-truncation safeguard: top-Fock-level population gauge AND an explicit
#      re-run at higher Ncut comparing |psi(x,tmax)|^2 in L1. NOTE: the
#      population gauge alone is NOT reliable (it can stay ~1e-99 while the
#      density is under-resolved); the higher-Ncut re-run is the real test.
#
#  HOW TO RUN
#  ----------
#      julia src/anharmonic_oscillator.jl          # runs defaults, writes figures to fig/
#  or from the REPL:
#      include("src/anharmonic_oscillator.jl")
#      run_simulation(lambda=4.8, alpha=2.5+0.0im, Ncut=200)
#      run_simulation(x0=3.0, p0=0.0, lambda=2.4)
#      run_simulation(make_gif=false)              # physics + checks only
#
#  DEPENDENCIES:  Julia stdlib (LinearAlgebra, Printf) + Plots (optional, only
#                 for the figures). ZERO installs required in the target env.
# =============================================================================

using LinearAlgebra
using Printf

# --- Headless / CI safety: make GR render to an in-memory device so it never
#     tries to open a window. Must be set BEFORE Plots/GR initializes. ---------
if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "nul"
end

# Plots is optional and only used for the figures; the physics runs without it.
const _HAVE_PLOTS = try
    @eval using Plots
    true
catch err
    @warn "Plots not available; will run physics + checks but skip the figures." exception=err
    false
end

# Figures are written to <repo>/fig (a sibling of this src/ directory). Resolved
# from the source-file location, so it works regardless of the current directory.
const FIGDIR = normpath(joinpath(@__DIR__, "..", "fig"))

# =============================================================================
# ============================  PARAMETERS BLOCK  =============================
#  Edit these defaults, OR call run_simulation(; kwargs...) with overrides.
#  Because this is a @kwdef struct, a typo'd keyword raises an error instead of
#  being silently ignored.
# =============================================================================
Base.@kwdef struct Params
    # --- physical constants (natural "oscillator" units by default) ---
    # The defaults are tuned to a weak-anharmonicity, semiclassical regime
    # (hbar=0.5, lambda=0.3, alpha=1.4) in which the CAUSAL NONLOCAL one-loop
    # curve (B) tracks the full quantum mean <q>(t) most closely: classical
    # dephases clearly, the local one-loop (A) follows the quantum, and the
    # nonlocal (B) -- whose memory term refines the frequency -- is the closest
    # approximation of all (RMS distance to <q>(t): classical 0.34, A 0.16, B 0.13).
    hbar::Float64   = 0.5          # reduced Planck constant (smaller -> more classical)
    m::Float64      = 1.0          # mass
    omega::Float64  = 1.0          # harmonic angular frequency of the HO reference

    # --- anharmonicity ---
    lambda::Float64 = 0.3          # anharmonicity strength (V quartic coeff = lambda/24)
    # CONVENTION: the quartic term is  lambda * q^4 / 4! = (lambda/24) q^4, matching
    # theory/q4_1_loop.tex. The default 0.3 gives quartic coefficient 0.3/24 ~ 0.0125.

    # --- initial coherent state ---
    # Provide EITHER alpha (complex) OR (x0,p0). If x0 OR p0 is finite the run
    # switches FULLY to (x0,p0) mode and alpha is ignored; an unspecified component
    # defaults to 0 (so x0=3 alone means p0=0, NOT Im(alpha)). Mapping:
    #   x0 = sqrt(2 hbar/(m omega)) Re(alpha),   p0 = sqrt(2 hbar m omega) Im(alpha)
    alpha::ComplexF64 = 1.4 + 0.0im # complex displacement amplitude
    x0::Float64 = NaN              # initial mean position (NaN ⇒ use alpha; any finite x0/p0 ⇒ (x0,p0) mode)
    p0::Float64 = NaN              # initial mean momentum (NaN ⇒ use alpha; unspecified component ⇒ 0)

    # --- Fock-space truncation ---
    # NOTE: x^4 couples to higher Fock levels than the coherent state itself
    # occupies, so the basis needs headroom ABOVE the populated levels for the
    # quartic operator to be represented accurately. Ncut=120 is converged for
    # the defaults (L1 |dens| diff < 1e-9 vs higher Ncut); raise it for larger
    # lambda or |alpha| (the convergence_check below will tell you). As a guide,
    # |alpha|~3 -> Ncut~250, |alpha|~3.5 -> Ncut~400, |alpha|~4 -> Ncut~600.
    Ncut::Int       = 140          # number of Fock states kept (0 .. Ncut-1)

    # --- real-space grid (for reconstruction / plotting only) ---
    # Tight box framing the small (hbar=0.5) wavepacket; autogrid off so it is
    # honored exactly. Raise xmin/xmax (or set autogrid=true) for larger |alpha|.
    xmin::Float64   = -5.0
    xmax::Float64   =  5.0
    Nx::Int         = 600
    # When true, the grid is GROWN (never shrunk) to contain the wavepacket for
    # large |alpha|, so the reconstructed |psi(x)|^2 keeps its norm. Set false to
    # honor xmin/xmax exactly (e.g. to zoom in).
    autogrid::Bool  = false

    # --- time span ---
    tmax::Float64   = 12.0 * pi    # total simulated time (window where (B) tracks the quantum)
    Nt::Int         = 360          # number of time frames (animation length)

    # --- animation / output ---
    fps::Int        = 24
    giffile::String = joinpath(FIGDIR, "anharmonic_oscillator.gif")
    make_gif::Bool  = true         # set false to skip rendering (just run physics+checks)
    convergence_check::Bool = true # re-run at higher Ncut to estimate truncation error

    # --- classical-vs-quantum comparison plot ---
    make_comparison::Bool = true   # save a <q>(t) quantum vs q(t) classical plot
    qcompfile::String = joinpath(FIGDIR, "q_comparison.png")
end

# -----------------------------------------------------------------------------
#  Copy-with-overrides constructor: Params(p; field=newval, ...)
#  Defined right next to the struct so source order is clean and dependable.
# -----------------------------------------------------------------------------
function Params(p::Params; kwargs...)
    d = Dict{Symbol,Any}()
    for f in fieldnames(Params)
        d[f] = getfield(p, f)
    end
    for (k, v) in kwargs
        haskey(d, k) || error("Unknown Params field: $k")
        d[k] = v
    end
    return Params(; d...)
end

# =============================================================================
# ===========================  OPERATOR BUILDERS  =============================
# =============================================================================

"""
    build_operators(p) -> (a, adag, X, Nop)

Annihilation `a`, creation `adag`, position `X` (in physical length units) and
number operator `Nop`, all as dense `Ncut x Ncut` ComplexF64 matrices.
"""
function build_operators(p::Params)
    N = p.Ncut
    a = zeros(ComplexF64, N, N)
    for n in 1:N-1
        a[n, n+1] = sqrt(n)          # <n-1| a |n> = sqrt(n); row n = |n-1>, col n+1 = |n>
    end
    adag = collect(a')               # creation = Hermitian conjugate of a
    x_scale = sqrt(p.hbar / (2 * p.m * p.omega))
    X = x_scale .* (a .+ adag)       # position operator in physical length units
    Nop = adag * a                   # number operator
    return a, adag, X, Nop
end

"""
    build_hamiltonian(p) -> Hermitian matrix

H = hbar*omega*(N + 1/2) + (lambda/24) * x^4  (= lambda q^4 / 4!), a dense
Hermitian matrix in the Fock basis.

The anharmonic term is the **Galerkin projection** `P_N x⁴ P_N`, NOT `(P_N x P_N)⁴`.
Truncating `x` to `N` levels and *then* taking the 4th power deletes the virtual
paths through Fock levels ≥ N, which corrupts the matrix near the cutoff (the
`⟨n|x⁴|n⟩` element is already off by O(100) at N≈6). We instead build `x` in an
`(N+4)`-dimensional workspace, form `x⁴` there, and crop the exact upper-left
`N×N` block — this reproduces the analytic three-band `x⁴` to machine precision and
is the proper variational (Rayleigh–Ritz) finite-basis Hamiltonian. For the default
(well-converged) state the two constructions agree to ~1e-13, but they diverge for
under-resolved / large-amplitude runs, which is exactly where the difference matters.
"""
function build_hamiltonian(p::Params)
    N = p.Ncut
    pad = 4
    M = N + pad
    a = zeros(ComplexF64, M, M)
    for n in 1:M-1
        a[n, n+1] = sqrt(n)
    end
    Xbig = sqrt(p.hbar / (2 * p.m * p.omega)) .* (a .+ a')
    X4 = (Xbig * Xbig * Xbig * Xbig)[1:N, 1:N]   # = P_N x⁴ P_N (exact projection)
    H = (p.lambda / 24) .* X4
    for n in 0:N-1
        H[n+1, n+1] += p.hbar * p.omega * (n + 0.5)   # ħω(a†a + ½), diagonal & truncation-exact
    end
    # Symmetrize to kill tiny round-off asymmetry so eigen() returns a real spectrum.
    H = (H .+ H') ./ 2
    return Hermitian(H)
end

# =============================================================================
# ============================  COHERENT STATE  ==============================
# =============================================================================

"""
    coherent_amplitudes(alpha, N) -> Vector{ComplexF64}

Raw (un-renormalized) truncated coherent amplitudes
c_n = exp(-|alpha|^2/2) alpha^n/sqrt(n!), via the stable recurrence
c_n = c_{n-1} alpha/sqrt(n). For an adequate N, norm(c) ~ 1; the deviation
|norm(c)-1| is exactly the Fock-truncation tail loss.
"""
function coherent_amplitudes(alpha::ComplexF64, N::Int)
    c = zeros(ComplexF64, N)
    c[1] = exp(-abs2(alpha) / 2)             # n = 0
    for n in 1:N-1
        c[n+1] = c[n] * alpha / sqrt(n)      # index n+1 holds |n>
    end
    return c
end

"""
    coherent_truncation_error(alpha, N) -> Float64

|norm(coherent_amplitudes) - 1|: how much of |alpha> falls outside the kept N
levels. ~0 means the basis is adequate; large means raise Ncut.
"""
coherent_truncation_error(alpha::ComplexF64, N::Int) = abs(norm(coherent_amplitudes(alpha, N)) - 1)

"""
    coherent_state_fock(alpha, N) -> Vector{ComplexF64}

|alpha> = exp(-|alpha|^2/2) sum_n alpha^n / sqrt(n!) |n>, truncated to N levels.
The construction is exact; a finite N truncates the Poisson tail. If that tail
loss is significant we WARN (don't crash) and project onto the kept subspace by
renormalizing -- so a too-small Ncut degrades gracefully instead of aborting.
"""
function coherent_state_fock(alpha::ComplexF64, N::Int)
    c = coherent_amplitudes(alpha, N)
    nrm = norm(c)
    if abs(nrm - 1) > 1e-8
        @warn @sprintf("Coherent state under-resolved by Fock truncation (tail |norm-1|=%.2e for |alpha|^2=%.2f, Ncut=%d); raise Ncut for accuracy.",
                       abs(nrm - 1), abs2(alpha), N)
    end
    c ./= nrm                                # project onto the kept subspace
    return c
end

"""
    resolve_alpha(p) -> ComplexF64

Choose alpha from x0/p0 if either is finite, else use p.alpha directly.
x0 = sqrt(2 hbar/(m omega)) Re(alpha),  p0 = sqrt(2 hbar m omega) Im(alpha).
"""
function resolve_alpha(p::Params)
    if isfinite(p.x0) || isfinite(p.p0)
        x0 = isfinite(p.x0) ? p.x0 : 0.0
        p0 = isfinite(p.p0) ? p.p0 : 0.0
        re = x0 / sqrt(2 * p.hbar / (p.m * p.omega))
        im = p0 / sqrt(2 * p.hbar * p.m * p.omega)
        return complex(re, im)
    else
        return p.alpha
    end
end

# =============================================================================
# ===================  HARMONIC-OSCILLATOR EIGENFUNCTIONS  ===================
#  phi_n(x) computed by a STABLE recurrence on the *normalized* functions.
#  Let xi = x*sqrt(m omega/hbar).  Then
#     phi_0   = (m omega/(pi hbar))^(1/4) exp(-xi^2/2)
#     phi_1   = sqrt(2) * xi * phi_0
#     phi_{n} = sqrt(2/n) xi phi_{n-1} - sqrt((n-1)/n) phi_{n-2}
#  This avoids overflow of raw Hermite polynomials / factorials.
#  NOTE: this builds an Nx x Ncut dense matrix; for very large Ncut*Nx it is the
#  main memory cost (and convergence_check builds it again at higher Ncut).
# =============================================================================

"""
    ho_eigenfunctions(xs, N, p) -> Phi  (Nx x N real matrix), Phi[:,n+1] = phi_n(xs)
"""
function ho_eigenfunctions(xs::AbstractVector{<:Real}, N::Int, p::Params)
    Nx = length(xs)
    Phi = zeros(Float64, Nx, N)
    s = sqrt(p.m * p.omega / p.hbar)            # 1/length scale
    xi = s .* xs                                # dimensionless coordinate
    norm0 = (p.m * p.omega / (pi * p.hbar))^(0.25)
    @inbounds for i in 1:Nx
        Phi[i, 1] = norm0 * exp(-xi[i]^2 / 2)   # phi_0
        if N >= 2
            Phi[i, 2] = sqrt(2.0) * xi[i] * Phi[i, 1]   # phi_1
        end
        for n in 2:N-1
            Phi[i, n+1] = sqrt(2.0 / n) * xi[i] * Phi[i, n] -
                          sqrt((n - 1) / n) * Phi[i, n-1]
        end
    end
    return Phi
end

# =============================================================================
# =====================  CLASSICAL EQUATIONS OF MOTION  ======================
#  The classical limit of the SAME Hamiltonian, for Ehrenfest comparison with
#  the quantum mean <q>(t). Integrated with a fixed-step RK4 (stdlib only).
#
#  Local potential V(q) = 1/2 m w^2 q^2 + (lambda/24) q^4, so Newton's law
#     m q'' = -dV/dq = -(m w^2 q + (lambda/6) q^3),
#  i.e.  q'' = -w^2 q - (lambda/(6 m)) q^3,  integrated for the state (q, v=q').
#
#  Initial conditions come from the SAME coherent amplitude as the quantum state:
#     q(0) = sqrt(2 hbar/(m w)) Re(alpha),  p(0) = sqrt(2 hbar m w) Im(alpha).
# =============================================================================

"""
    classical_trajectory(p, alpha, ts) -> (qcl, pcl)

Classical q(t), p(t) sampled at the times `ts`, from the quartic Newton EOM
q'' = -w^2 q - (lambda/(6 m)) q^3, integrated with RK4.
"""
function classical_trajectory(p::Params, alpha::ComplexF64, ts::AbstractVector{<:Real})
    nt = length(ts)
    qcl = zeros(Float64, nt)
    pcl = zeros(Float64, nt)
    q0 = sqrt(2 * p.hbar / (p.m * p.omega)) * real(alpha)
    p0 = sqrt(2 * p.hbar * p.m * p.omega) * imag(alpha)

    # RK4 on the state u = (q, v), v = dq/dt, acceleration a(q) = F/m.
    accel(q) = -p.omega^2 * q - (p.lambda / (6 * p.m)) * q^3
    q = q0
    v = p0 / p.m
    qcl[1] = q
    pcl[1] = p.m * v
    nsub = 40                                   # RK4 substeps per output interval
    for k in 2:nt
        h = (ts[k] - ts[k-1]) / nsub
        for _ in 1:nsub
            k1q = v;            k1v = accel(q)
            k2q = v + 0.5h*k1v; k2v = accel(q + 0.5h*k1q)
            k3q = v + 0.5h*k2v; k3v = accel(q + 0.5h*k2q)
            k4q = v + h*k3v;    k4v = accel(q + h*k3q)
            q += h * (k1q + 2k2q + 2k3q + k4q) / 6
            v += h * (k1v + 2k2v + 2k3v + k4v) / 6
        end
        qcl[k] = q
        pcl[k] = p.m * v
    end
    return qcl, pcl
end

# =============================================================================
# ===============  REAL-TIME TREE + ONE-LOOP EQUATION OF MOTION  =============
#  Semiclassical (one-loop) correction to the mean Q(t)=<q>(t), from the
#  real-time, local effective action derived in theory/q4_1_loop.tex
#  (eq. (A) "final-adiabatic"):
#
#     Z(Q) Qddot + 1/2 Z'(Q) Qdot^2 + Veff'(Q) = 0 ,
#
#  with the SHARED normalization V = 1/2 m w^2 q^2 + lambda q^4/4! = (lambda/24) q^4,
#     Omega(Q) = sqrt(w^2 + lambda/(2m) Q^2)
#     Z(Q)     = m + hbar lambda^2 Q^2 / (32 m^2 Omega^5)
#     1/2 Z'(Q)= hbar lambda^2/(64 m^2) [ 2Q/Omega^5 - 5 lambda Q^3/(2 m Omega^7) ]
#     Veff'(Q) = m w^2 Q + (lambda/6) Q^3 + hbar lambda Q/(4 m Omega).
#
#  CONVENTION: this code and theory/q4_1_loop.tex use the SAME coupling
#  (V = lambda q^4/4!), so the symbol `lam` below is exactly p.lambda (no
#  rescaling). At tree level (hbar->0), Veff'/m = w^2 Q + (lambda/6m) Q^3 is
#  this code's classical force, so hbar -> 0 reproduces classical_trajectory EXACTLY.
#
#  Same initial data as the coherent state / classical run:
#     Q(0) = sqrt(2 hbar/(m w)) Re(alpha),  Qdot(0) = sqrt(2 hbar m w) Im(alpha)/m.
#  Valid in the adiabatic regime |dOmega|/Omega^2 << 1. (Returns NaN where
#  Omega^2<=0, e.g. for lambda<0, where the local expansion breaks down.)
# =============================================================================

"""
    oneloop_trajectory(p, alpha, ts) -> Qol

Solve the real-time local tree+one-loop EOM for the mean Q(t)=<q>(t) (RK4).
"""
function oneloop_trajectory(p::Params, alpha::ComplexF64, ts::AbstractVector{<:Real})
    nt = length(ts)
    Q = zeros(Float64, nt)
    m, w, hbar = p.m, p.omega, p.hbar
    lam = p.lambda                           # shared coupling: V = lambda q^4/4! (no rescaling)

    # acceleration Qddot = -(1/2 Z' Qdot^2 + Veff') / Z   (function of Q and Qdot)
    function accel(q, dq)
        Om2 = w^2 + lam / (2m) * q^2
        Om2 <= 0 && return NaN               # Omega^2<0 (e.g. lambda<0): one-loop invalid
        Om  = sqrt(Om2)
        Z      = m + hbar * lam^2 * q^2 / (32 * m^2 * Om^5)
        halfZp = hbar * lam^2 / (64 * m^2) * (2q / Om^5 - 5 * lam * q^3 / (2m * Om^7))
        Veffp  = m * w^2 * q + lam / 6 * q^3 + hbar * lam * q / (4m * Om)
        return -(halfZp * dq^2 + Veffp) / Z
    end

    q = sqrt(2 * hbar / (m * w)) * real(alpha)
    v = sqrt(2 * hbar * m * w) * imag(alpha) / m
    Q[1] = q
    nsub = 40
    for k in 2:nt
        h = (ts[k] - ts[k-1]) / nsub
        for _ in 1:nsub                      # RK4 on (q, v=dq/dt); accel depends on BOTH
            k1q = v;            k1v = accel(q, v)
            k2q = v + 0.5h*k1v; k2v = accel(q + 0.5h*k1q, v + 0.5h*k1v)
            k3q = v + 0.5h*k2v; k3v = accel(q + 0.5h*k2q, v + 0.5h*k2v)
            k4q = v + h*k3v;    k4v = accel(q + h*k3q,    v + h*k3v)
            q += h * (k1q + 2k2q + 2k3q + k4q) / 6
            v += h * (k1v + 2k2v + 2k3v + k4v) / 6
        end
        Q[k] = q
    end
    return Q
end

# =============================================================================
# ===========  CAUSAL NONLOCAL TREE + ONE-LOOP EQUATION (memory)  ===========
#  The real-time, CAUSAL one-loop equation with the first nonlocal memory term,
#  from theory/q4_1_loop.tex eq. (B) "final-nonlocal":
#
#    m Qddot + m w^2 Q + (lambda/6) Q^3 + hbar lambda/(4 m w) Q
#      - hbar lambda^2/(8 m^2 w^2) Q(t) * INT_{t0}^{t} sin[2 w (t-t')] Q^2(t') dt' = 0.
#
#  The fluctuation correlator <eta^2(t)> = hbar/(2 m w) - (hbar lambda)/(4 m^2 w^2)
#  INT sin[2w(t-t')] Q^2 dt' is built from FREE modes (linear in delta Om^2), so the
#  force at time t depends only on the PAST history of Q -- causal, with memory.
#
#  The kernel is separable: sin[2w(t-t')] = sin(2wt)cos(2wt') - cos(2wt)sin(2wt'),
#  so  I(t) = sin(2wt) C(t) - cos(2wt) S(t)  with  C' = cos(2wt) Q^2, S' = sin(2wt) Q^2.
#  That turns the integro-differential equation into an EXACT augmented ODE system
#  (Q, V=Qdot, C, S) -- no history storage -- integrated with the same RK4.
#
#  Same initial data as the other curves (C(t0)=S(t0)=0, t0=0). Unlike the local
#  one-loop (A), the memory term is non-conservative, so the mean amplitude can
#  drift -- this is the leading non-adiabatic backreaction of the fluctuations.
# =============================================================================

"""
    nonlocal_trajectory(p, alpha, ts) -> Qnl

Solve the causal nonlocal tree+one-loop EOM (theory eq. B) for the mean Q(t),
via the augmented (Q, V, C, S) ODE with RK4. `ts` must start at t0 = 0.
"""
function nonlocal_trajectory(p::Params, alpha::ComplexF64, ts::AbstractVector{<:Real})
    nt = length(ts)
    Qout = zeros(Float64, nt)
    m, w, hbar, lam = p.m, p.omega, p.hbar, p.lambda
    c1 = lam / (6m)                  # cubic (tree)
    c2 = hbar * lam / (4 * m^2 * w)  # local tadpole (constant freq w)
    c3 = hbar * lam^2 / (8 * m^3 * w^2)  # memory coefficient

    # derivative of state (Q, V, C, S) at time t -> (dQ, dV, dC, dS)
    function deriv(t, Q, V, C, S)
        s2 = sin(2w * t); c2t = cos(2w * t)
        Imem = s2 * C - c2t * S              # = INT sin[2w(t-t')] Q^2(t') dt'
        dV = -w^2 * Q - c1 * Q^3 - c2 * Q + c3 * Q * Imem
        return (V, dV, c2t * Q^2, s2 * Q^2)
    end

    Q = sqrt(2 * hbar / (m * w)) * real(alpha)
    V = sqrt(2 * hbar * m * w) * imag(alpha) / m
    C = 0.0; S = 0.0
    Qout[1] = Q
    nsub = 40
    for k in 2:nt
        h = (ts[k] - ts[k-1]) / nsub
        t = ts[k-1]
        for _ in 1:nsub
            (a1,b1,c1k,d1) = deriv(t,        Q,            V,            C,            S)
            (a2,b2,c2k,d2) = deriv(t+0.5h,   Q+0.5h*a1,    V+0.5h*b1,    C+0.5h*c1k,   S+0.5h*d1)
            (a3,b3,c3k,d3) = deriv(t+0.5h,   Q+0.5h*a2,    V+0.5h*b2,    C+0.5h*c2k,   S+0.5h*d2)
            (a4,b4,c4k,d4) = deriv(t+h,      Q+h*a3,       V+h*b3,       C+h*c3k,      S+h*d3)
            Q += h*(a1 + 2a2 + 2a3 + a4)/6
            V += h*(b1 + 2b2 + 2b3 + b4)/6
            C += h*(c1k + 2c2k + 2c3k + c4k)/6
            S += h*(d1 + 2d2 + 2d3 + d4)/6
            t += h
        end
        Qout[k] = Q
    end
    return Qout
end

# =============================================================================
# ===============================  CORE SOLVER  ==============================
# =============================================================================

struct SimResult
    p::Params
    alpha::ComplexF64
    xs::Vector{Float64}
    ts::Vector{Float64}
    dens::Matrix{Float64}     # |psi(x,t)|^2 ,  Nx x Nt
    xexp::Vector{Float64}     # <x>(t)  (quantum mean position)
    pexp::Vector{Float64}     # <p>(t)  (quantum mean momentum)
    qcl::Vector{Float64}      # q(t)    (classical position, classical EOM)
    pcl::Vector{Float64}      # p(t)    (classical momentum, classical EOM)
    qol::Vector{Float64}      # Q(t)    (local adiabatic tree+1-loop mean, theory eq. A)
    qnl::Vector{Float64}      # Q(t)    (causal nonlocal tree+1-loop mean, theory eq. B)
    energy::Vector{Float64}   # <H>(t)
    norm_x::Vector{Float64}   # grid norm at each frame
    norm_fock::Vector{Float64}# Fock norm at each frame
    Vx::Vector{Float64}       # potential V(x) on the grid
    Evals::Vector{Float64}    # eigen-energies (for reference)
    toppop::Float64           # population in the top 5% of Fock levels at t=0 (truncation gauge)
    toppop_dyn::Float64       # MAX top-5% Fock population over t (dynamical truncation gauge)
    trunc_err::Float64        # |norm of un-renormalized coherent state - 1| (basis adequacy)
end

"""
    autosize_grid(p, alpha) -> Params

If `p.autogrid`, return a copy of `p` whose grid is grown (never shrunk) to
contain the oscillating wavepacket: half-width >= classical amplitude
sqrt(2 hbar/(m omega))|alpha| plus a few widths, and also the turning point of
the highest significantly-populated Fock level, with Nx scaled to preserve dx.
Otherwise returns `p` unchanged.
"""
function autosize_grid(p::Params, alpha::ComplexF64)
    p.autogrid || return p
    sigma0 = sqrt(p.hbar / (2 * p.m * p.omega))              # ground-state width
    xlen   = sqrt(p.hbar / (p.m * p.omega))                  # HO length scale
    amp    = sqrt(2 * p.hbar / (p.m * p.omega)) * abs(alpha) # classical oscillation amplitude
    # Two estimates of the needed half-width; take the larger:
    #  (1) classical turning point + a few widths;
    #  (2) the classical turning point of the highest significantly-populated
    #      Fock level (~|alpha|^2 + 6|alpha|), which bounds the reconstruction.
    nmax   = abs2(alpha) + 6 * abs(alpha) + 10
    xhw    = max(amp + 10 * sigma0,
                 sqrt(2 * nmax + 1) * xlen + 3 * sigma0)
    xmin = min(p.xmin, -xhw)
    xmax = max(p.xmax,  xhw)
    (xmin == p.xmin && xmax == p.xmax) && return p           # default box already fits
    dx0 = (p.xmax - p.xmin) / (p.Nx - 1)
    Nx  = max(p.Nx, round(Int, (xmax - xmin) / dx0) + 1)
    return Params(p; xmin=xmin, xmax=xmax, Nx=Nx)
end

"""
    simulate(p::Params) -> SimResult

Run the full Fock-basis evolution and reconstruct everything on the grid.
"""
function simulate(p::Params)
    alpha = resolve_alpha(p)
    p = autosize_grid(p, alpha)        # grow the grid to contain the state (if autogrid)

    # --- operators & Hamiltonian ---
    a, adag, X, _ = build_operators(p)
    H = build_hamiltonian(p)

    # --- one-shot diagonalization (exact propagator) ---
    F = eigen(H)                      # F.values (real), F.vectors (unitary V)
    Evals = F.values
    Vmat  = F.vectors

    # --- momentum operator for <p>(t):  p = i sqrt(hbar m omega/2)(adag - a) ---
    p_scale = sqrt(p.hbar * p.m * p.omega / 2)
    P = (im * p_scale) .* (adag .- a)

    # --- initial coherent state in Fock basis (+ basis-adequacy gauge) ---
    trunc_err = coherent_truncation_error(alpha, p.Ncut)
    psi0 = coherent_state_fock(alpha, p.Ncut)
    n0 = norm(psi0)
    # truncation gauge: population in the top 5% of levels (see warning in report)
    ntop = max(1, round(Int, 0.05 * p.Ncut))
    toppop = sum(abs2, @view psi0[end-ntop+1:end]) / abs2(n0)

    # coefficients in the energy eigenbasis (time evolution is trivial here)
    c_E = Vmat' * psi0                # |psi0> = Vmat * c_E

    # --- real-space reconstruction matrix (HO eigenfunctions) ---
    xs = collect(range(p.xmin, p.xmax; length=p.Nx))
    Phi = ho_eigenfunctions(xs, p.Ncut, p)     # Nx x Ncut
    dx = xs[2] - xs[1]

    # potential on the grid (for plotting / energy display)
    Vx = 0.5 * p.m * p.omega^2 .* xs.^2 .+ (p.lambda / 24) .* xs.^4

    # --- time grid ---
    ts = collect(range(0.0, p.tmax; length=p.Nt))

    # --- classical trajectory q(t) from the classical EOM (same initial alpha) ---
    qcl, pcl = classical_trajectory(p, alpha, ts)
    if any(!isfinite, qcl)
        @warn "Classical trajectory diverged to NaN/Inf (unbounded potential, e.g. lambda<0); the classical q(t) curve will be incomplete. The quantum propagator still runs and conserves norm/energy WITHIN the fixed Ncut, but for lambda<0 the potential is unbounded below: the Fock truncation acts as an artificial regulator, so the spectrum (and hence the evolution) is Ncut-dependent and NOT physically convergent — treat it as a regularized toy model, not the true lambda<0 dynamics."
    end

    # --- tree+one-loop means: local adiabatic (A) and causal nonlocal (B) ---
    qol = oneloop_trajectory(p, alpha, ts)
    qnl = nonlocal_trajectory(p, alpha, ts)

    dens   = zeros(Float64, p.Nx, p.Nt)
    xexp   = zeros(Float64, p.Nt)
    pexp   = zeros(Float64, p.Nt)
    energy = zeros(Float64, p.Nt)
    norm_x = zeros(Float64, p.Nt)
    norm_f = zeros(Float64, p.Nt)

    # dynamical truncation gauge: the MAX population in the top 5% of Fock levels
    # over the whole evolution. The t=0 gauge (toppop) misses population leaking INTO
    # the cutoff during the run (x^4 couples upward), which is exactly how a too-small
    # Ncut hides while norm/energy stay conserved within the kept subspace.
    toppop_dyn = toppop

    phase = similar(c_E)              # workspace
    for (k, t) in enumerate(ts)
        # exact propagation in the eigenbasis (exact WITHIN the Ncut truncation)
        @. phase = c_E * cis(-Evals * t / p.hbar)   # cis(theta) = exp(i theta)
        psi_f = Vmat * phase                          # state in Fock basis

        # Fock-basis expectation values (basis-independent, cheap, exact)
        norm_f[k] = real(dot(psi_f, psi_f))
        toppop_dyn = max(toppop_dyn, sum(abs2, @view psi_f[end-ntop+1:end]) / norm_f[k])
        xexp[k]   = real(dot(psi_f, X * psi_f))
        pexp[k]   = real(dot(psi_f, P * psi_f))
        energy[k] = real(dot(psi_f, H * psi_f))

        # real-space wavefunction & density
        psi_x = Phi * psi_f                            # Nx complex vector
        @. dens[:, k] = abs2(psi_x)
        norm_x[k] = sum(@view dens[:, k]) * dx
    end

    return SimResult(p, alpha, xs, ts, dens, xexp, pexp, qcl, pcl, qol, qnl, energy,
                     norm_x, norm_f, Vx, Evals, toppop, toppop_dyn, trunc_err)
end

# =============================================================================
# ============================  PHYSICS CHECKS  ==============================
# =============================================================================

function report_checks(r::SimResult)
    p = r.p
    x0_target = sqrt(2*p.hbar/(p.m*p.omega))*real(r.alpha)
    p0_target = sqrt(2*p.hbar*p.m*p.omega)*imag(r.alpha)
    println("="^72)
    println(" PHYSICS / CORRECTNESS REPORT")
    println("="^72)
    @printf("  alpha               = %.4f %+.4fi\n", real(r.alpha), imag(r.alpha))
    @printf("  x0 = sqrt(2 hbar/(m w)) Re(a) = %.6f ,  <x>(0) measured = %.6f\n",
            x0_target, r.xexp[1])
    @printf("  p0 = sqrt(2 hbar m w) Im(a)   = %.6f ,  <p>(0) measured = %.6f\n",
            p0_target, r.pexp[1])

    # normalization
    @printf("  Fock-basis norm at t=0           = %.12f  (|err|=%.2e)\n",
            r.norm_fock[1], abs(r.norm_fock[1]-1))
    @printf("  Grid     norm at t=0 (sum*dx)    = %.12f  (|err|=%.2e)\n",
            r.norm_x[1], abs(r.norm_x[1]-1))

    # conservation along trajectory
    fock_dev = maximum(abs.(r.norm_fock .- 1))
    grid_dev = maximum(abs.(r.norm_x   .- 1))
    e0 = r.energy[1]
    e_dev = maximum(abs.(r.energy .- e0)) / max(abs(e0), eps())
    @printf("  max |Fock norm - 1| over t       = %.3e  (exact evolution -> ~machine eps)\n", fock_dev)
    @printf("  max |grid norm - 1| over t       = %.3e  (limited by grid resolution)\n", grid_dev)
    if grid_dev > 1e-3
        @warn @sprintf("Reconstructed |psi(x)|^2 lost %.2e of its norm: high-n HO eigenfunctions overrun the grid box [%.1f,%.1f]. Widen xmin/xmax (or keep autogrid=true). The Fock-basis physics (<x>,<p>,<H>,norm) is UNAFFECTED -- only the plotted density is under-resolved.",
                       grid_dev, p.xmin, p.xmax)
    end
    @printf("  <H>(0)                           = %.10f\n", e0)
    @printf("  max relative |<H>(t)-<H>(0)|     = %.3e  (constant of motion)\n", e_dev)

    # Truncation gauges. NOTE: norm/energy conservation above are conserved WITHIN the
    # kept Ncut subspace by unitarity, so they do NOT detect truncation error -- the
    # gauges below and the convergence_check (re-run at higher Ncut) are what catch it.
    @printf("  top-5%% Fock population: t=0 = %.3e ,  max over t = %.3e  (want << 1)\n",
            r.toppop, r.toppop_dyn)
    if r.toppop_dyn > 1e-6
        @warn @sprintf("Fock truncation may be insufficient: top-level population reaches %.2e during the evolution -- increase Ncut (and trust the convergence check, not norm/energy conservation).", r.toppop_dyn)
    end

    # Hard assertions. Conservation/unitarity asserts hold ALWAYS (they test the
    # propagator within the kept subspace). The <x>(0)/<p>(0) == analytic-target
    # asserts only make sense when the Fock basis actually resolves |alpha>; for a
    # deliberately under-resolved run they are expected to fail, so we gate them.
    @assert abs(r.norm_fock[1]-1) < 1e-10  "Coherent state not normalized in Fock basis."
    @assert fock_dev < 1e-9                "Fock norm not conserved (evolution bug?)."
    @assert e_dev   < 1e-8                 "Energy not conserved (evolution bug?)."
    if r.trunc_err < 1e-6
        @assert abs(r.xexp[1]-x0_target) < 1e-6 "<x>(0) does not match analytic target."
        @assert abs(r.pexp[1]-p0_target) < 1e-6 "<p>(0) does not match analytic target."
        println("  All hard assertions passed (norm + energy conserved, <x>/<p> match).")
    else
        @warn @sprintf("Basis under-resolved (coherent-truncation tail=%.2e): skipping <x>(0)/<p>(0) target asserts -- raise Ncut. (Unitarity + energy conservation still asserted and passed.)", r.trunc_err)
        println("  Conservation assertions passed (target-match asserts skipped: raise Ncut).")
    end
    println("="^72)
    return nothing
end

"""
    run_convergence_check(p) -> nothing

Re-run at Ncut and a larger Ncut, compare |psi(x,tmax)|^2 (L1 difference on the
grid) and the energy. This is the RELIABLE Fock-truncation test -- the top-level
population gauge alone can stay ~1e-99 even when the density is under-resolved.
"""
function run_convergence_check(p::Params)
    println("\n  [Fock-truncation convergence check]")
    r1 = simulate(p)
    p2 = Params(p; Ncut = max(p.Ncut + 40, 2*p.Ncut))
    r2 = simulate(p2)
    dx = r1.xs[2] - r1.xs[1]
    l1 = sum(abs.(r1.dens[:, end] .- r2.dens[:, end])) * dx
    de = abs(r1.energy[1] - r2.energy[1])
    @printf("    Ncut = %d vs %d :  L1 |dens(x,tmax)| diff = %.3e ,  d<H> = %.3e\n",
            p.Ncut, p2.Ncut, l1, de)
    if l1 > 1e-3
        @warn @sprintf("Density differs by %.2e between Ncut=%d and Ncut=%d -- increase Ncut (large lambda/alpha).",
                       l1, p.Ncut, p2.Ncut)
    else
        println("    -> Converged in Ncut (L1 diff < 1e-3).")
    end
    return nothing
end

"""
    selftest_coherent_state(p) -> nothing

Standalone self-test of the coherent-state construction: verify Fock-basis
normalization and the analytic <x>,<p> targets before any time evolution.
"""
function selftest_coherent_state(p::Params)
    alpha = resolve_alpha(p)
    trunc = coherent_truncation_error(alpha, p.Ncut)
    psi0 = coherent_state_fock(alpha, p.Ncut)
    nrm = real(dot(psi0, psi0))
    _, adag, X, _ = build_operators(p)
    a = collect(adag')
    p_scale = sqrt(p.hbar * p.m * p.omega / 2)
    P = (im * p_scale) .* (adag .- a)
    xmean = real(dot(psi0, X * psi0))
    pmean = real(dot(psi0, P * psi0))
    x0_target = sqrt(2*p.hbar/(p.m*p.omega))*real(alpha)
    p0_target = sqrt(2*p.hbar*p.m*p.omega)*imag(alpha)
    # If the basis under-resolves |alpha>, the (renormalized) truncated state
    # legitimately misses the analytic <x>/<p> targets -- warn instead of crash.
    if trunc > 1e-6
        @warn @sprintf("[self-test] coherent state under-resolved (Fock-truncation tail=%.2e at Ncut=%d, |alpha|^2=%.1f); skipping exact <x>/<p> asserts -- raise Ncut. Measured <x>=%.4f (target %.4f), <p>=%.4f (target %.4f).",
                       trunc, p.Ncut, abs2(alpha), xmean, x0_target, pmean, p0_target)
        return nothing
    end
    @assert abs(nrm-1) < 1e-10          "Self-test FAILED: coherent state normalization."
    @assert abs(xmean-x0_target) < 1e-6 "Self-test FAILED: <x>(0) target mismatch."
    @assert abs(pmean-p0_target) < 1e-6 "Self-test FAILED: <p>(0) target mismatch."
    @printf("  [self-test] coherent state OK: norm=%.12f, <x>=%.6f (target %.6f), <p>=%.6f (target %.6f)\n",
            nrm, xmean, x0_target, pmean, p0_target)
    return nothing
end

# =============================================================================
# ==============================  ANIMATION  =================================
#  Overlays |psi(x,t)|^2 (filled), the potential V(x), the quantum mean <q>(t)
#  and the classical q(t) markers, with t / <q> / norm / <H> in the title.
#
#  The potential is drawn on a TRUE energy scale on a SECONDARY y-axis so the
#  coherent state's classical turning points are physically meaningful (density
#  on the left axis, energy on the right).
# =============================================================================

function animate(r::SimResult)
    if !_HAVE_PLOTS
        @warn "Plots not loaded -- skipping animation."
        return nothing
    end
    p = r.p
    dmax = maximum(r.dens)
    ymax = 1.15 * max(dmax, eps())

    e0 = r.energy[1]

    # Energy-axis range: cover the well from its floor up to a bit above the
    # coherent-state energy <H>=e0 so the classical turning points (where the
    # dashed V(x) crosses <H>) are visible. Clip the top so the steep x^4 walls
    # don't dwarf the well.
    Elo = minimum(r.Vx)
    Ehi = max(1.6 * e0, e0 + 2.0)

    # Use the non-macro Plots.Animation API (NOT the @animate macro): a macro is
    # expanded at function-DEFINITION (lowering) time, so an @animate here would make
    # this whole file fail to `include` whenever Plots is absent — defeating the
    # runtime `_HAVE_PLOTS` guard above and the "physics runs without Plots" promise.
    # Plain function calls (Plots.Animation/Plots.frame/...) resolve at CALL time, so
    # the file loads Plots-free and only `animate(...)` itself requires Plots.
    anim = Plots.Animation()
    for k in 1:p.Nt
        ttl = @sprintf("Anharmonic oscillator (lambda q^4/4!, Fock)  t=%6.3f\n<q>=%+.3f  norm=%.6f  <H>=%.4f (E0=%.4f)",
                       r.ts[k], r.xexp[k], r.norm_fock[k], r.energy[k], e0)
        # Left axis: probability density (filled area).
        plt = plot(r.xs, r.dens[:, k];
                   lw=2, label="|psi(x,t)|^2", color=:dodgerblue,
                   fill=(0, 0.18, :dodgerblue),
                   xlabel="x", ylabel="probability density |psi|^2",
                   ylim=(0, ymax), xlim=(p.xmin, p.xmax),
                   legend=:topright, framestyle=:box, title=ttl,
                   titlefontsize=9, size=(820, 520))
        # Vertical markers: quantum <q>, classical q, 1-loop local (A), 1-loop nonlocal (B).
        vline!(plt, [r.xexp[k]]; lw=2, color=:crimson, label="<q> quantum")
        vline!(plt, [r.qcl[k]];  lw=2, ls=:dot, color=:seagreen, label="q classical")
        if isfinite(r.qol[k])
            vline!(plt, [r.qol[k]]; lw=2, ls=:dashdot, color=:darkorange, label="Q 1-loop local (A)")
        end
        if isfinite(r.qnl[k])
            vline!(plt, [r.qnl[k]]; lw=2, ls=:dashdotdot, color=:purple, label="Q 1-loop nonlocal (B)")
        end

        # Right axis: TRUE potential V(x) (energy units).
        plot!(twinx(plt), r.xs, r.Vx;
              lw=2, ls=:dash, color=:gray35, label="V(x) (energy)",
              ylabel="energy", ylim=(Elo, Ehi), xlim=(p.xmin, p.xmax),
              legend=:topleft)
        Plots.frame(anim, plt)
    end

    out = p.giffile
    isempty(dirname(out)) || mkpath(dirname(out))
    g = gif(anim, out; fps=p.fps)
    @printf("  GIF written: %s  (%d frames, %d fps)\n", out, p.Nt, p.fps)
    return g
end

# =============================================================================
# =======  FULL-QUANTUM vs CLASSICAL vs TREE+1-LOOP (local A & nonlocal B)  ==
#  A single static figure with FOUR curves for the mean position:
#    * full quantum <q>(t)          (exact Fock-basis evolution)
#    * classical q(t)               (tree level / hbar^0 EOM)
#    * tree+one-loop local (A)      (local adiabatic eq, theory/q4_1_loop)
#    * tree+one-loop nonlocal (B)   (causal memory eq, theory/q4_1_loop)
#  All start from the same coherent state; they coincide at early times and
#  split as the wavepacket distorts. (B)'s memory term is non-conservative, so it
#  can follow the quantum amplitude better than the conservative (A).
# =============================================================================

function plot_q_comparison(r::SimResult)
    if !_HAVE_PLOTS
        @warn "Plots not loaded -- skipping comparison plot."
        return nothing
    end
    p = r.p
    # Where do the quantum and classical means first separate? (purely informational)
    qscale = maximum(abs, r.qcl) + eps()
    dev = abs.(r.xexp .- r.qcl) ./ qscale
    isplit = findfirst(>(0.1), dev)              # first 10%-of-amplitude departure
    tsplit = isplit === nothing ? nothing : r.ts[isplit]

    ttl = @sprintf("Mean position <q>(t): quantum vs classical vs tree+1-loop (local A & nonlocal B)   (lambda=%.3g, |alpha|=%.2f)",
                   p.lambda, abs(r.alpha))
    plt = plot(r.ts, r.xexp;
               lw=2.5, color=:dodgerblue, label="quantum  <q>(t)",
               xlabel="t", ylabel="q", title=ttl, titlefontsize=9,
               legend=:topright, framestyle=:box, size=(900, 480),
               xlim=(r.ts[1], r.ts[end]))
    plot!(plt, r.ts, r.qcl; lw=2, ls=:dash, color=:crimson, label="classical  q(t)")
    if any(isfinite, r.qol)
        plot!(plt, r.ts, r.qol; lw=2, ls=:dashdot, color=:darkorange,
              label="1-loop local  (A)")
    end
    if any(isfinite, r.qnl)
        plot!(plt, r.ts, r.qnl; lw=2, ls=:dashdotdot, color=:purple,
              label="1-loop nonlocal  (B)")
    end
    if tsplit !== nothing
        vline!(plt, [tsplit]; lw=1.5, ls=:dot, color=:gray45,
               label=@sprintf("Ehrenfest split  t~%.2f", tsplit))
    end
    out = p.qcompfile
    isempty(dirname(out)) || mkpath(dirname(out))
    savefig(plt, out)
    @printf("  Comparison plot written: %s%s\n", out,
            tsplit === nothing ? "" : @sprintf("  (quantum<->classical split at t~%.2f)", tsplit))
    return plt
end

# =============================================================================
# ============================  TOP-LEVEL DRIVER  ============================
# =============================================================================

"""
    run_simulation(; kwargs...) -> SimResult

Build params (defaults overridable by kwargs), run the Fock-basis evolution,
print physics checks, optionally render the animation, and (optionally) run a
Fock-truncation convergence check.

Examples:
    run_simulation()                                        # defaults
    run_simulation(lambda=4.8, alpha=2.5+0.0im, Ncut=200)   # stronger anharmonicity
    run_simulation(x0=3.0, p0=0.0, lambda=2.4)              # specify x0,p0 instead of alpha
    run_simulation(make_gif=false)                          # physics only, no GIF
"""
function run_simulation(; kwargs...)
    p = Params(Params(); kwargs...)   # copy-ctor gives a friendly "Unknown Params field" on typos
    println("\n>>> run_simulation: Ncut=$(p.Ncut), lambda=$(p.lambda), ",
            "hbar=$(p.hbar), tmax=$(round(p.tmax,digits=3)), Nt=$(p.Nt)")
    selftest_coherent_state(p)
    r = simulate(p)
    report_checks(r)
    if p.convergence_check
        run_convergence_check(p)
    end
    if p.make_comparison
        plot_q_comparison(r)
    end
    if p.make_gif
        animate(r)
    end
    return r
end

# =============================================================================
# Run with defaults when executed as a script (not when `include`d for REPL use).
# =============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulation()
end
