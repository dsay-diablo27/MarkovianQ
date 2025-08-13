#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import numpy as np
from scipy.linalg import svd, sqrtm
import matplotlib.pyplot as plt

PAULI = {
    "X": np.array([[0,1],[1,0]], dtype=complex),
    "Y": np.array([[0,-1j],[1j,0]], dtype=complex),
    "Z": np.array([[1,0],[0,-1]], dtype=complex)
}

def ensure_outdir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def amp_damping_kraus(gamma, t):
    p = 1.0 - math.exp(-gamma * t)
    k0 = np.array([[1.0, 0.0],
                   [0.0, math.sqrt(max(0.0, 1.0 - p))]], dtype=complex)
    k1 = np.array([[0.0, math.sqrt(max(0.0, p))],
                   [0.0, 0.0]], dtype=complex)
    return k0, k1

def apply_kraus(rho, K0, K1):
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

def svd_decompose(M):
    U, S, Vh = svd(M)
    return U, S, Vh

def default_initial_rho():
    return np.array([[0.0, 0.0],[0.0, 1.0]], dtype=complex)

def fidelity_uhlmann(rho, sigma):
    rho = 0.5 * (rho + rho.conj().T)
    sigma = 0.5 * (sigma + sigma.conj().T)
    sqrtrho = sqrtm(rho)
    inside = sqrtm(sqrtrho @ sigma @ sqrtrho)
    tr = np.real_if_close(np.trace(inside))
    val = (np.real(tr))**2
    return float(np.clip(val, 0.0, 1.0))

def build_unitary_from_kraus(K0, K1):
    V = np.vstack([K0, K1])
    Q, R = np.linalg.qr(V, mode='complete')
    U = Q
    return U

def simulate_rho_via_unitary(K0, K1, rho_sys):
    U = build_unitary_from_kraus(K0, K1)
    rho_anc = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    rho_comb = np.kron(rho_anc, rho_sys)
    rho_after = U @ rho_comb @ U.conj().T
    rho_sys_sim = np.zeros((2,2), dtype=complex)
    for i in range(2):
        for j in range(2):
            rho_sys_sim[i,j] = rho_after[0*2 + i, 0*2 + j] + rho_after[1*2 + i, 1*2 + j]
    rho_sys_sim = 0.5 * (rho_sys_sim + rho_sys_sim.conj().T)
    return rho_sys_sim

class KrausSVDProcessor:
    def __init__(self, gamma=1.0, omega=1.0):
        self.gamma = gamma
        self.omega = omega

    def kraus_ops(self, t):
        return amp_damping_kraus(self.gamma, t)

    def analytical_rho_t(self, t, rho0=None):
        if rho0 is None: rho0 = default_initial_rho()
        K0, K1 = self.kraus_ops(t)
        return apply_kraus(rho0, K0, K1)

    def svd_components(self, t):
        K0, K1 = self.kraus_ops(t)
        U0,S0,Vh0 = svd_decompose(K0)
        U1,S1,Vh1 = svd_decompose(K1)
        return {"K0":K0,"K1":K1,"U0":U0,"S0":S0,"Vh0":Vh0,"U1":U1,"S1":S1,"Vh1":Vh1}

    def time_series_rhos(self, T, steps=50, rho0=None):
        times = np.linspace(0.0, T, steps)
        rhos = np.zeros((len(times), 2, 2), dtype=complex)
        for i,t in enumerate(times):
            rhos[i] = self.analytical_rho_t(t, rho0=rho0)
        return times, rhos

    def expectation_series(self, axis, T, steps=50, rho0=None):
        axis = axis.upper()
        if axis not in PAULI:
            raise ValueError("Axis must be X, Y, or Z")
        times, rhos = self.time_series_rhos(T, steps=steps, rho0=rho0)
        exps = np.real([np.trace(rhos[i] @ PAULI[axis]) for i in range(len(times))])
        return times, exps

def action_kraus_svd(args):
    proc = KrausSVDProcessor(gamma=args.gamma, omega=args.omega)
    K0, K1 = proc.kraus_ops(args.t)
    U0,S0,Vh0 = svd_decompose(K0)
    U1,S1,Vh1 = svd_decompose(K1)
    rho_t = proc.analytical_rho_t(args.t)
    np.set_printoptions(precision=6, suppress=True)
    def fmt(A):
        return np.array2string(A, formatter={'complex_kind':lambda z: f"{z.real:.6f}{z.imag:+.6f}j"})
    print("\n===== Kraus operators at t = {:.6f}, gamma = {:.6f} =====\n".format(args.t, args.gamma))
    print("K0 =\n", fmt(K0), "\n")
    print("K1 =\n", fmt(K1), "\n")
    print("----- SVD of K0 -----")
    print("U0 =\n", fmt(U0))
    print("S0 =\n", np.array2string(S0, precision=6))
    print("Vh0 =\n", fmt(Vh0))
    print("\n----- SVD of K1 -----")
    print("U1 =\n", fmt(U1))
    print("S1 =\n", np.array2string(S1, precision=6))
    print("Vh1 =\n", fmt(Vh1))
    print("\n----- Analytical rho(t) -----")
    print(fmt(rho_t))
    print("\n(End of kraus_svd output)\n")

def action_fidelity(args):
    out = ensure_outdir(args.outdir)
    proc = KrausSVDProcessor(gamma=args.gamma, omega=args.omega)
    times, rhos_analytical = proc.time_series_rhos(args.t, steps=args.steps)
    fidelities = []
    for idx, tt in enumerate(times):
        K0, K1 = proc.kraus_ops(tt)
        rho_ana = rhos_analytical[idx]
        rho_sim = simulate_rho_via_unitary(K0, K1, default_initial_rho())
        fid = fidelity_uhlmann(rho_ana, rho_sim)
        fidelities.append(fid)
    fidelities = np.array(fidelities)
    plt.figure()
    plt.plot(times, fidelities, marker='o')
    plt.xlabel("Time")
    plt.ylabel("Fidelity (analytical vs simulated-by-unitary)")
    plt.title("Fidelity vs Time (auto-simulated via unitary dilation)")
    plt.grid(True)
    figpath = out / "fidelity_vs_time.png"
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()
    print(f"Saved fidelity plot to: {figpath}")
    print(f"Mean fidelity: {np.nanmean(fidelities):.6f}")
    print(f"Min fidelity:  {np.nanmin(fidelities):.6f}")
    print(f"Max fidelity:  {np.nanmax(fidelities):.6f}")

def action_expectation(args):
    out = ensure_outdir(args.outdir)
    axis = (args.axis or "Z").upper()
    proc = KrausSVDProcessor(gamma=args.gamma, omega=args.omega)
    times, exps = proc.expectation_series(axis, args.t, steps=args.steps)
    plt.figure()
    plt.plot(times, exps, marker='o')
    plt.xlabel("Time")
    plt.ylabel(f"<{axis}>")
    plt.title(f"Expectation value <{axis}> vs Time")
    plt.grid(True)
    figpath = out / f"expectation_{axis}.png"
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()
    print(f"Saved expectation plot to: {figpath}")
    print(f"Min <{axis}>: {np.min(exps):.6f}, Max <{axis}>: {np.max(exps):.6f}, Final <{axis}>: {exps[-1]:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Simple Lindblad->Kraus CLI (print & plots).")
    parser.add_argument("action", choices=["kraus_svd", "fidelity", "expectation"], help="Action to run")
    parser.add_argument("axis", nargs="?", help="Axis for expectation (X/Y/Z). For 'expectation' action only.", default="Z")
    parser.add_argument("--outdir", default="./outputs", help="Directory to save plot images")
    parser.add_argument("--t", type=float, default=5.0, help="Final time T")
    parser.add_argument("--gamma", type=float, default=1.0, help="Decay rate gamma")
    parser.add_argument("--omega", type=float, default=1.0, help="omega (unused for amplitude-damping)")
    parser.add_argument("--steps", type=int, default=50, help="Number of time points for time-series actions")
    parser.add_argument("--simulated", default=None, help="Optional .npy file with simulated rhos (shape N x 2 x 2) for fidelity")
    args = parser.parse_args()
    if args.action == "kraus_svd":
        action_kraus_svd(args)
    elif args.action == "fidelity":
        action_fidelity(args)
    elif args.action == "expectation":
        args.axis = args.axis
        action_expectation(args)
    else:
        print("Unknown action")

if __name__ == "__main__":
    main()
