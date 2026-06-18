# =============================================================================
#  REGRESSION / AUDIT TEST SUITE for anharmonic_oscillator.jl
# =============================================================================
#  Independent verification of the physics, numerics, classical EOM, and
#  robustness/edge-case handling. Does NOT modify the main script.
#
#  Run:   julia audit_checks.jl
#  Exit code is nonzero if any check fails (CI-friendly).
#
#  Highlights:
#   * Harmonic limit (lambda=0): quantum <q>(t) == classical q(t) == analytic
#     x0 cos(wt)+(p0/mw) sin(wt) to ~1e-12, density rigid -> validates BOTH the
#     quantum propagator and the classical RK4 in the only exactly-solvable case.
#   * Ehrenfest d<x>/dt = <p>/m; classical energy conservation; RK4 4th-order.
#   * HO reconstruction basis orthonormal.
#   * tree+one-loop local (A) & causal nonlocal (B): reduce to classical/quantum
#     at lambda=0 and each satisfies its own EOM (the nonlocal memory residual too).
#   * Robustness: graceful Fock-truncation, autogrid, negative-lambda warning,
#     friendly unknown-keyword error.
#   * Truncation honesty: a deliberately under-truncated run conserves energy to
#     ~1e-15 (so conservation is BLIND to cutoff error) while the dynamical
#     top-population gauge and the higher-Ncut convergence re-run both DETECT it.
# =============================================================================
ENV["GKSwstype"] = "nul"
include(joinpath(@__DIR__, "anharmonic_oscillator.jl"))
using Printf, LinearAlgebra

pass(name, ok, msg="") = (@printf("  [%s] %-52s %s\n", ok ? "PASS" : "FAIL", name, msg); ok)
const ALLOK = Ref(true)
check(name, ok, msg="") = (ALLOK[] &= pass(name, ok, msg))

println("="^82); println(" ANHARMONIC OSCILLATOR -- AUDIT / REGRESSION SUITE"); println("="^82)
println("\n#### PART A: PHYSICS & NUMERICS ####")

# 1. Harmonic limit (lambda=0): exact Ehrenfest, rigid Gaussian.
println("\n-- 1. Harmonic limit lambda=0 (exact Ehrenfest, rigid Gaussian) --")
let
    p = Params(lambda=0.0, alpha=2.0+1.0im, hbar=1.0, Ncut=120, tmax=4pi, Nt=200,
               xmin=-12.0, xmax=12.0, make_gif=false, make_comparison=false, convergence_check=false)
    r = simulate(p)
    w, m, hbar = p.omega, p.m, p.hbar
    x0 = sqrt(2hbar/(m*w))*real(r.alpha); p0 = sqrt(2hbar*m*w)*imag(r.alpha)
    qa = x0 .* cos.(w .* r.ts) .+ (p0/(m*w)) .* sin.(w .* r.ts)
    eq = maximum(abs.(r.xexp .- qa)); ec = maximum(abs.(r.qcl .- qa)); eqc = maximum(abs.(r.xexp .- r.qcl))
    check("quantum <q>(t) == analytic x0cos+..", eq < 1e-5, @sprintf("max err=%.2e", eq))
    check("classical q(t) == analytic",          ec < 1e-6, @sprintf("max err=%.2e", ec))
    check("quantum == classical (lambda=0)",     eqc < 1e-5, @sprintf("max err=%.2e", eqc))
    dx = r.xs[2]-r.xs[1]
    varr = [ (sum(r.dens[:,k].*r.xs.^2)*dx) - (sum(r.dens[:,k].*r.xs)*dx)^2 for k in 1:p.Nt ]
    check("density variance constant (rigid)",  maximum(varr)-minimum(varr) < 1e-6,
          @sprintf("var range=%.2e (sigma^2~%.4f)", maximum(varr)-minimum(varr), varr[1]))
end

# 2. Ehrenfest d<x>/dt = <p>/m.
println("\n-- 2. Ehrenfest d<x>/dt = <p>/m (default) --")
let
    p = Params(make_gif=false, make_comparison=false, convergence_check=false, Nt=600, tmax=4pi)
    r = simulate(p); dt = r.ts[2]-r.ts[1]
    dxdt = (r.xexp[3:end] .- r.xexp[1:end-2]) ./ (2dt); pm = r.pexp[2:end-1] ./ p.m
    e = maximum(abs.(dxdt .- pm))
    check("d<x>/dt == <p>/m", e < 5e-3, @sprintf("max err=%.2e (finite-diff O(dt^2))", e))
end

# 3. Classical energy conservation along the RK4 trajectory.
println("\n-- 3. Classical energy E=1/2 m v^2 + V(q) conserved (RK4) --")
let
    p = Params(make_gif=false, make_comparison=false, convergence_check=false, lambda=1.2, Nt=400, tmax=6pi)
    r = simulate(p); v = r.pcl ./ p.m
    E = 0.5*p.m .* v.^2 .+ 0.5*p.m*p.omega^2 .* r.qcl.^2 .+ (p.lambda/24) .* r.qcl.^4
    rel = maximum(abs.(E .- E[1]))/abs(E[1])
    check("classical energy conserved", rel < 1e-6, @sprintf("rel drift=%.2e, E=%.4f", rel, E[1]))
end

# 4. RK4 step-convergence vs a high-resolution reference.
println("\n-- 4. RK4 accuracy vs high-resolution reference --")
let
    p = Params(make_gif=false, make_comparison=false, convergence_check=false, lambda=2.4, Nt=200, tmax=4pi)
    alpha = resolve_alpha(p); ts = collect(range(0,p.tmax,length=p.Nt))
    qcl, _ = classical_trajectory(p, alpha, ts)
    accel(q) = -p.omega^2*q - (p.lambda/(6p.m))*q^3
    q = sqrt(2p.hbar/(p.m*p.omega))*real(alpha); v = sqrt(2p.hbar*p.m*p.omega)*imag(alpha)/p.m
    ref = zeros(p.Nt); ref[1]=q; nsub=4000
    for k in 2:p.Nt
        h=(ts[k]-ts[k-1])/nsub
        for _ in 1:nsub
            k1q=v;k1v=accel(q);k2q=v+0.5h*k1v;k2v=accel(q+0.5h*k1q)
            k3q=v+0.5h*k2v;k3v=accel(q+0.5h*k2q);k4q=v+h*k3v;k4v=accel(q+h*k3q)
            q+=h*(k1q+2k2q+2k3q+k4q)/6; v+=h*(k1v+2k2v+2k3v+k4v)/6
        end
        ref[k]=q
    end
    e = maximum(abs.(qcl .- ref))
    check("RK4 nsub=40 matches nsub=4000 ref", e < 1e-6, @sprintf("max diff=%.2e", e))
end

# 5. HO reconstruction basis orthonormality on the grid.
println("\n-- 5. Reconstruction basis phi_n orthonormality on grid --")
let
    p = Params(xmin=-14.0, xmax=14.0, Nx=2000, Ncut=30)
    xs = collect(range(p.xmin,p.xmax,length=p.Nx)); dx=xs[2]-xs[1]
    Phi = ho_eigenfunctions(xs, p.Ncut, p)
    G = Phi' * Phi .* dx
    Gm = G - Matrix{Float64}(I, p.Ncut, p.Ncut)
    offdiag = maximum(abs.(Gm - Diagonal(Gm))); diagerr = maximum(abs.(diag(G) .- 1))
    check("phi_n normalized (diag~1)", diagerr < 5e-3, @sprintf("max|<n|n>-1|=%.2e", diagerr))
    check("phi_n orthogonal (offdiag~0)", offdiag < 5e-3, @sprintf("max|G-I|=%.2e", offdiag))
end

println("\n#### PART B: ROBUSTNESS & EDGE CASES ####")

# 6. x0/p0 override mapping.
println("\n-- 6. x0/p0 override mapping --")
let
    p = Params(x0=3.0, p0=1.5, make_gif=false, make_comparison=false, convergence_check=false)
    a = resolve_alpha(p)
    x0b = sqrt(2p.hbar/(p.m*p.omega))*real(a); p0b = sqrt(2p.hbar*p.m*p.omega)*imag(a)
    check("x0 override round-trips", abs(x0b-3.0)<1e-12, @sprintf("x0=%.6f", x0b))
    check("p0 override round-trips", abs(p0b-1.5)<1e-12, @sprintf("p0=%.6f", p0b))
    a2 = resolve_alpha(Params(x0=3.0, make_gif=false, make_comparison=false, convergence_check=false))
    check("only x0 finite -> p0=0", abs(imag(a2))<1e-15, @sprintf("Im(alpha)=%.2e", imag(a2)))
end

# 7. Error handling: bad keyword (struct + run_simulation).
println("\n-- 7. Error handling --")
let
    p = Params()
    threw=false;  try; Params(p; bogus=1); catch; threw=true; end
    check("unknown Params field errors", threw)
    msg=""; try; run_simulation(bogus=1, make_gif=false); catch e; msg=sprint(showerror,e); end
    check("run_simulation friendly kwarg error", occursin("Unknown Params field", msg))
end

# 8. Graceful Fock truncation: too-small Ncut WARNS + COMPLETES (not crash).
println("\n-- 8. Graceful Fock truncation (large |alpha|, small Ncut) --")
let
    r = run_simulation(alpha=6.0+0.0im, Ncut=60, make_gif=false, make_comparison=false, convergence_check=false)
    check("alpha=6/Ncut=60 completes (no crash)", r isa SimResult)
    check("Fock norm renormalized to 1", abs(r.norm_fock[1]-1) < 1e-12, @sprintf("|err|=%.2e", abs(r.norm_fock[1]-1)))
    check("truncation error flagged", r.trunc_err > 1e-6, @sprintf("trunc=%.2e", r.trunc_err))
end

# 9. autogrid contains the state for large |alpha|; no-op when the state fits.
println("\n-- 9. autogrid grid sizing --")
let
    roff = simulate(Params(alpha=5.0+0.0im, Ncut=220, autogrid=false, make_gif=false, make_comparison=false, convergence_check=false))
    ron  = simulate(Params(alpha=5.0+0.0im, Ncut=220, autogrid=true,  make_gif=false, make_comparison=false, convergence_check=false))
    devoff = maximum(abs.(roff.norm_x .- 1)); devon = maximum(abs.(ron.norm_x .- 1))
    check("autogrid widened the grid", ron.xs[end] > roff.xs[end], @sprintf("xmax %.2f -> %.2f", roff.xs[end], ron.xs[end]))
    check("autogrid reduced grid-norm loss", devon < devoff, @sprintf("dev %.2e -> %.2e", devoff, devon))
    rd = simulate(Params(alpha=1.0+0.0im, hbar=1.0, xmin=-10.0, xmax=10.0, Nx=500, autogrid=true,
                         make_gif=false, make_comparison=false, convergence_check=false))
    check("autogrid no-op when state fits box", rd.xs[1]≈-10.0 && rd.xs[end]≈10.0 && length(rd.xs)==500,
          @sprintf("[%.1f,%.1f] Nx=%d", rd.xs[1], rd.xs[end], length(rd.xs)))
end

# 10. Negative lambda: quantum completes, classical divergence is flagged.
#     Pinned to ħ=1, α=2 (q0=2.83 sits above the inverted-well barrier -> escapes).
println("\n-- 10. Negative-lambda classical divergence surfaced --")
let
    r = run_simulation(lambda=-1.2, hbar=1.0, alpha=2.0+0.0im, xmin=-12.0, xmax=12.0,
                       make_gif=false, make_comparison=false, convergence_check=false)
    check("lambda<0 quantum run completes", r.norm_fock[1] ≈ 1)
    check("classical qcl flagged non-finite", any(!isfinite, r.qcl))
end

println("\n#### PART C: THEORY -- REAL-TIME TREE + ONE-LOOP ####")

# 11. hbar^0 consistency: at lambda=0 the one-loop EOM has no hbar terms left,
#     so Q(t) must equal BOTH the classical q(t) and the exact quantum <q>(t).
println("\n-- 11. tree+1-loop reduces to classical & quantum at lambda=0 --")
let
    p = Params(lambda=0.0, alpha=2.0+0.5im, Nt=400, tmax=4pi,
               make_gif=false, make_comparison=false, convergence_check=false)
    r = simulate(p)
    check("one-loop == classical (lambda=0)", maximum(abs.(r.qol .- r.qcl)) < 1e-9,
          @sprintf("max diff=%.2e", maximum(abs.(r.qol .- r.qcl))))
    check("one-loop == quantum  (lambda=0)", maximum(abs.(r.qol .- r.xexp)) < 1e-5,
          @sprintf("max diff=%.2e", maximum(abs.(r.qol .- r.xexp))))
    check("one-loop starts at q0", abs(r.qol[1]-r.qcl[1]) < 1e-12)
end

# 12. The RK4 solution actually SOLVES the boxed real-time one-loop EOM:
#     residual  Z Q'' + 1/2 Z' Q'^2 + Veff'  ~ 0  (finite-diff on a fine grid).
println("\n-- 12. tree+1-loop RK4 satisfies its own EOM (residual) --")
let
    p = Params(lambda=1.2, alpha=2.0+0.0im, Nt=4000, tmax=4pi,
               make_gif=false, make_comparison=false, convergence_check=false)
    r = simulate(p); Q = r.qol; ts = r.ts; dt = ts[2]-ts[1]
    m, w, hbar = p.m, p.omega, p.hbar; lam = p.lambda
    maxres = 0.0; maxscale = 0.0
    for k in 2:length(Q)-1
        q  = Q[k]
        dq = (Q[k+1]-Q[k-1])/(2dt)
        ddq= (Q[k+1]-2Q[k]+Q[k-1])/dt^2
        Om = sqrt(w^2 + lam/(2m)*q^2)
        Z      = m + hbar*lam^2*q^2/(32*m^2*Om^5)
        halfZp = hbar*lam^2/(64*m^2)*(2q/Om^5 - 5*lam*q^3/(2m*Om^7))
        Veffp  = m*w^2*q + lam/6*q^3 + hbar*lam*q/(4m*Om)
        maxres   = max(maxres, abs(Z*ddq + halfZp*dq^2 + Veffp))
        maxscale = max(maxscale, abs(Z*ddq)+abs(Veffp))
    end
    check("EOM residual / scale small", maxres/maxscale < 1e-3,
          @sprintf("rel residual=%.2e", maxres/maxscale))
end

# 13. one-loop differs from classical at finite lambda (the correction is real).
println("\n-- 13. one-loop is a genuine correction (lambda>0) --")
let
    r = simulate(Params(lambda=1.2, alpha=2.0+0.0im, hbar=1.0, Ncut=120, Nt=240, tmax=4pi,
                        make_gif=false, make_comparison=false, convergence_check=false))
    sep = maximum(abs.(r.qol .- r.qcl))
    check("one-loop separates from classical (lambda>0)", sep > 1e-2, @sprintf("max sep=%.3f", sep))
end

# 14. CONVENTION: the quartic potential is V_4 = lambda q^4 / 4! = (lambda/24) q^4
#     (uniform with theory/q4_1_loop.tex). Pinned to the standard reference
#     hbar=m=omega=1, lambda=1.2, alpha=2 (independent of the run-time defaults).
println("\n-- 14. uniform convention  V = lambda q^4 / 4! --")
let
    lam = 1.2
    H = build_hamiltonian(Params(lambda=lam, hbar=1.0, m=1.0, omega=1.0))  # reference ħ=m=ω=1
    x4_00 = 3 * (0.5)^2                             # <0|x^4|0> = 3 (hbar/2mw)^2 = 3/4
    expected = 0.5 + (lam/24) * x4_00              # <0|H|0> = hw/2 + (lambda/24)<x^4>
    check("quartic = lambda/24 (= lambda q^4/4!)", abs(real(H[1,1]) - expected) < 1e-12,
          @sprintf("H[1,1]=%.6f vs %.6f", real(H[1,1]), expected))
    # Galerkin-projection check: the FULL x^4 matrix (not just H[1,1]) must equal the
    # analytic three-band P_N x^4 P_N, INCLUDING the cutoff rows where the naive
    # (P_N x P_N)^4 construction is wrong by O(100). s^4=(hbar/2mw)^2=1/4 at hbar=m=w=1.
    let Np = 12, s4 = 0.25
        Hp = Matrix(build_hamiltonian(Params(lambda=24.0, hbar=1.0, m=1.0, omega=1.0, Ncut=Np)))
        x4code = real.(Hp)
        for n in 0:Np-1; x4code[n+1,n+1] -= (n + 0.5); end          # peel off hw(n+1/2)
        x4an = zeros(Np, Np)
        for n in 0:Np-1
            x4an[n+1,n+1] = s4 * (6n^2 + 6n + 3)
            if n+2 <= Np-1; v = s4*2*(2n+3)*sqrt((n+1)*(n+2));            x4an[n+1,n+3]=v; x4an[n+3,n+1]=v; end
            if n+4 <= Np-1; v = s4*sqrt((n+1)*(n+2)*(n+3)*(n+4));         x4an[n+1,n+5]=v; x4an[n+5,n+1]=v; end
        end
        errp = maximum(abs.(x4code .- x4an))
        check("x^4 = analytic P_N x^4 P_N (incl. cutoff rows)", errp < 1e-10,
              @sprintf("max|x^4_code - analytic|=%.2e (naive (PxP)^4 would be O(100))", errp))
    end
    rA = simulate(Params(lambda=1.2, hbar=1.0, alpha=2.0+0.0im, Ncut=120, xmin=-12.0, xmax=12.0,
                         make_gif=false, make_comparison=false, convergence_check=false))
    check("<H>(0)=8.9375 at ħ=m=ω=1, λ=1.2, α=2", abs(rA.energy[1]-8.9375) < 1e-3,
          @sprintf("<H>(0)=%.6f", rA.energy[1]))
end

# 15. Causal NONLOCAL tree+one-loop (theory eq. B): reduces to classical/quantum
#     at lambda=0; distinct from the local (A) curve at lambda>0; and the RK4
#     solution satisfies the boxed integro-differential equation (memory residual).
println("\n-- 15. causal nonlocal tree+one-loop (theory eq. B) --")
let
    # (a) lambda=0: all hbar/lambda corrections vanish -> nonlocal == classical == quantum.
    r0 = simulate(Params(lambda=0.0, alpha=2.0+0.5im, hbar=1.0, Ncut=120, tmax=4pi, Nt=400,
                         xmin=-12.0, xmax=12.0, make_gif=false, make_comparison=false, convergence_check=false))
    check("nonlocal == classical (lambda=0)", maximum(abs.(r0.qnl .- r0.qcl)) < 1e-9,
          @sprintf("max diff=%.2e", maximum(abs.(r0.qnl .- r0.qcl))))
    check("nonlocal == quantum  (lambda=0)", maximum(abs.(r0.qnl .- r0.xexp)) < 1e-4,
          @sprintf("max diff=%.2e", maximum(abs.(r0.qnl .- r0.xexp))))
    check("nonlocal starts at q0", abs(r0.qnl[1]-r0.qcl[1]) < 1e-12)
    # (b) lambda>0: nonlocal (B) is a genuinely distinct curve from local (A).
    r1 = simulate(Params(lambda=1.2, alpha=2.0+0.0im, hbar=1.0, Ncut=120, tmax=4pi, Nt=240,
                         xmin=-12.0, xmax=12.0, make_gif=false, make_comparison=false, convergence_check=false))
    check("nonlocal (B) distinct from local (A)", maximum(abs.(r1.qnl .- r1.qol)) > 1e-3,
          @sprintf("max |B-A|=%.3f", maximum(abs.(r1.qnl .- r1.qol))))
    # (c) residual: recompute the memory integral from qnl (trapezoid, separable kernel)
    #     and verify  m Q'' + m w^2 Q + (lam/6)Q^3 + h*lam/(4mw) Q - h*lam^2/(8m^2w^2) Q I = 0.
    rr = simulate(Params(lambda=1.2, alpha=2.0+0.0im, hbar=1.0, Ncut=80, tmax=4pi, Nt=4000,
                         xmin=-12.0, xmax=12.0, make_gif=false, make_comparison=false, convergence_check=false))
    Q = rr.qnl; ts = rr.ts; dt = ts[2]-ts[1]; m=rr.p.m; w=rr.p.omega; hbar=rr.p.hbar; lam=rr.p.lambda
    n = length(Q); Carr = zeros(n); Sarr = zeros(n)
    for k in 2:n
        Carr[k] = Carr[k-1] + 0.5dt*(cos(2w*ts[k-1])*Q[k-1]^2 + cos(2w*ts[k])*Q[k]^2)
        Sarr[k] = Sarr[k-1] + 0.5dt*(sin(2w*ts[k-1])*Q[k-1]^2 + sin(2w*ts[k])*Q[k]^2)
    end
    maxres = 0.0; maxscale = 0.0
    for k in 3:n-1
        ddq  = (Q[k+1]-2Q[k]+Q[k-1])/dt^2
        Imem = sin(2w*ts[k])*Carr[k] - cos(2w*ts[k])*Sarr[k]
        res  = m*ddq + m*w^2*Q[k] + lam/6*Q[k]^3 + hbar*lam/(4m*w)*Q[k] -
               hbar*lam^2/(8*m^2*w^2)*Q[k]*Imem
        maxres = max(maxres, abs(res)); maxscale = max(maxscale, abs(m*w^2*Q[k]) + abs(m*ddq))
    end
    check("nonlocal RK4 satisfies eq (B) residual", maxres/maxscale < 5e-3,
          @sprintf("rel residual=%.2e", maxres/maxscale))
end

# 16. TRUNCATION ERROR IS DETECTABLE, not hidden by conservation. A deliberately
#     under-truncated run (alpha=4, Ncut=60, lambda=2) conserves energy to machine
#     precision -- proving norm/energy conservation CANNOT see truncation -- yet the
#     dynamical top-population gauge and the higher-Ncut convergence re-run both flag
#     a catastrophic cutoff error. The default run is cleanly resolved.
println("\n-- 16. cutoff error is detectable, not hidden by conservation --")
let
    bad  = simulate(Params(alpha=4.0+0.0im, Ncut=60,  lambda=2.0, hbar=1.0, tmax=4pi, Nt=200,
                          xmin=-16.0, xmax=16.0, make_gif=false, make_comparison=false, convergence_check=false))
    good = simulate(Params(alpha=4.0+0.0im, Ncut=120, lambda=2.0, hbar=1.0, tmax=4pi, Nt=200,
                          xmin=-16.0, xmax=16.0, make_gif=false, make_comparison=false, convergence_check=false))
    e_dev = maximum(abs.(bad.energy .- bad.energy[1])) / max(abs(bad.energy[1]), eps())
    dx = bad.xs[2]-bad.xs[1]; l1 = sum(abs.(bad.dens[:,end] .- good.dens[:,end])) * dx
    rd = simulate(Params(make_gif=false, make_comparison=false, convergence_check=false))
    check("under-truncated run STILL conserves energy (conservation is blind)", e_dev < 1e-8,
          @sprintf("e_dev=%.2e", e_dev))
    check("dynamical top-pop FLAGS the truncation", bad.toppop_dyn > 1e-2,
          @sprintf("max top-5%% pop=%.3f", bad.toppop_dyn))
    check("convergence re-run DETECTS the cutoff error", l1 > 1e-1,
          @sprintf("L1 dens diff Ncut 60 vs 120 = %.3f", l1))
    check("default run is cleanly resolved (dyn top-pop << 1)", rd.toppop_dyn < 1e-6,
          @sprintf("default max top-5%% pop=%.2e", rd.toppop_dyn))
end

println("\n" * "="^82)
println(ALLOK[] ? " ALL AUDIT CHECKS PASSED" : " SOME AUDIT CHECKS FAILED -- SEE ABOVE")
println("="^82)
ALLOK[] || error("audit checks failed")
