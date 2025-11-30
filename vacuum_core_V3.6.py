"""
================================================================================
                        COMPUTATIONAL VACUUM ENGINE
                       (Core Kernel v3.6 - SELF-AWARE)
================================================================================

DESCRIPTION:
This module implements the fundamental tensor network algorithms for simulating
1D quantum lattice systems under the "Computational Vacuum" hypothesis.

INTEGRITY ASSURANCE:
This kernel includes a built-in 'VacuumDiagnostics' suite. 
Run this file directly to verify mathematical correctness before experiments.

CAPABILITIES:
1. MPS (Matrix Product State): Exact Tensor Network Contraction.
2. TEBD (Time-Evolving Block Decimation): Trotterized Evolution.
3. ARBITER (The Scissors): Information Capacity Enforcement.

AUTHOR: Project Computational Vacuum
STATUS: PRODUCTION
================================================================================
"""

import numpy as np
import scipy.linalg as la
from dataclasses import dataclass
from typing import List, Tuple
import time

# --- GLOBAL CONSTANTS ---
PRECISION = 1e-12

class Pauli:
    """Standard Pauli matrices."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

@dataclass
class VacuumConfig:
    """Simulation Hyperparameters."""
    N: int = 20              # System Size
    chi_max: int = 32        # RAM Limit (Holographic Bound)
    d: int = 2               # Physical Dimension
    dt: float = 0.05         # Planck Time
    J: float = 1.0           # Interaction
    h: float = 1.0           # Field

class MPS:
    """State of the Universe (Wavefunction)."""
    def __init__(self, N: int, chi_max: int, d: int = 2):
        self.N = N
        self.chi_max = chi_max
        self.d = d
        self.Gammas: List[np.ndarray] = []
        self.Lambdas: List[np.ndarray] = []
        self._init_vacuum()

    def _init_vacuum(self):
        self.Gammas = []
        self.Lambdas = [np.array([1.0])]
        for _ in range(self.N):
            G = np.zeros((1, self.d, 1), dtype=complex)
            G[0, 0, 0] = 1.0
            self.Gammas.append(G)
            self.Lambdas.append(np.array([1.0]))

    def normalize(self):
        """Local numerical stabilization."""
        for i in range(len(self.Lambdas)):
            n = la.norm(self.Lambdas[i])
            if n > PRECISION: self.Lambdas[i] /= n

    def get_norm_full(self) -> float:
        """
        EXACT GLOBAL NORM CALCULATION.
        Contracts the full Tensor Network to check probability conservation.
        """
        env = np.ones((1, 1), dtype=complex)
        for i in range(self.N):
            G = self.Gammas[i]
            L = self.Lambdas[i]
            A = G * L[:, None, None]
            step1 = np.tensordot(env, A, axes=(0, 0))
            env = np.tensordot(step1, A.conj(), axes=([0, 1], [0, 1]))
        L_last = self.Lambdas[-1]
        res = np.einsum('ij,i,j->', env, L_last, L_last.conj())
        return np.sqrt(np.real(res))

    def global_restore_norm(self):
        """Restores global unitarity (Gauge Fixing)."""
        current_norm = self.get_norm_full()
        if current_norm > PRECISION:
            self.Lambdas[self.N // 2] /= current_norm

    def get_entanglement_entropy(self, bond: int) -> float:
        if bond <= 0 or bond >= self.N: return 0.0
        S = self.Lambdas[bond]**2
        S = S[S > PRECISION]
        S /= np.sum(S)
        return float(-np.sum(S * np.log(S)))
    
    def expectation_two_site(self, i: int, op1: np.ndarray, op2: np.ndarray) -> float:
        G1, G2 = self.Gammas[i], self.Gammas[i+1]
        L1, L2, L3 = self.Lambdas[i], self.Lambdas[i+1], self.Lambdas[i+2]
        T1 = G1 * L1[:, None, None] * L2[None, None, :]
        T2 = G2 * L3[None, None, :]
        Theta = np.tensordot(T1, T2, axes=(2,0))
        res = np.einsum('astc,ks,lt,aklc->', Theta, op1, op2, Theta.conj())
        return np.real(res)

    def expectation_one_site(self, i: int, op: np.ndarray) -> float:
        G = self.Gammas[i]
        L_L = self.Lambdas[i]
        L_R = self.Lambdas[i+1]
        A = G * L_L[:, None, None] * L_R[None, None, :]
        res = np.einsum('asb,ks,akb->', A, op, A.conj())
        return np.real(res)

class TEBD:
    """Dynamics Engine (Hamiltonian Evolution)."""
    def __init__(self, mps: MPS, config: VacuumConfig):
        self.mps = mps
        self.cfg = config
        self._build_gates()

    def _build_gates(self):
        XX = np.kron(Pauli.X, Pauli.X)
        ZI = np.kron(Pauli.Z, Pauli.I)
        IZ = np.kron(Pauli.I, Pauli.Z)
        XI = np.kron(Pauli.X, Pauli.I)
        IX = np.kron(Pauli.I, Pauli.X)
        epsilon = 1e-3
        H_bond = -self.cfg.J * XX - 0.5 * self.cfg.h * (ZI + IZ) - 0.5 * epsilon * (XI + IX)
        self.gate_real = la.expm(-1j * H_bond * self.cfg.dt).reshape(2,2,2,2)
        self.gate_imag = la.expm(-H_bond * self.cfg.dt).reshape(2,2,2,2)

    def apply_gate(self, i: int, gate: np.ndarray):
        G_L, G_R = self.mps.Gammas[i], self.mps.Gammas[i+1]
        L_L, L_M, L_R = self.mps.Lambdas[i], self.mps.Lambdas[i+1], self.mps.Lambdas[i+2]
        
        T1 = G_L * L_L[:, None, None] * L_M[None, None, :]
        T2 = G_R * L_R[None, None, :]
        Theta = np.tensordot(T1, T2, axes=(2,0))
        Theta_new = np.tensordot(Theta, gate, axes=([1,2], [2,3]))
        Theta_new = np.transpose(Theta_new, (0, 2, 3, 1))
        
        d = self.mps.d
        sh = Theta_new.shape
        Theta_mat = Theta_new.reshape(sh[0]*d, d*sh[3])
        
        try:
            U, S, Vh = la.svd(Theta_mat, full_matrices=False)
        except la.LinAlgError: return
            
        # --- THE SCISSORS (Hypothesis Enforcement) ---
        chi_new = min(self.mps.chi_max, len(S))
        S = S[:chi_new]
        U = U[:, :chi_new]
        Vh = Vh[:chi_new, :]
        S /= la.norm(S)
        
        self.mps.Lambdas[i+1] = S
        inv_L_L = np.where(L_L > PRECISION, 1.0/L_L, 0.0)
        inv_L_R = np.where(L_R > PRECISION, 1.0/L_R, 0.0)
        self.mps.Gammas[i] = U.reshape(sh[0], d, chi_new) * inv_L_L[:, None, None]
        self.mps.Gammas[i+1] = Vh.reshape(chi_new, d, sh[3]) * inv_L_R[None, None, :]

    def sweep(self, imaginary: bool = False):
        gate = self.gate_imag if imaginary else self.gate_real
        for i in range(0, self.mps.N-1, 2): self.apply_gate(i, gate)
        for i in range(1, self.mps.N-1, 2): self.apply_gate(i, gate)
        if imaginary: self.mps.normalize()

    def get_energy(self) -> float:
        E = 0.0
        for i in range(self.mps.N - 1):
            E -= self.cfg.J * self.mps.expectation_two_site(i, Pauli.X, Pauli.X)
        for i in range(self.mps.N):
            E -= self.cfg.h * self.mps.expectation_one_site(i, Pauli.Z)
        return E

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM DIAGNOSTICS (SELF-CHECK)
# ═══════════════════════════════════════════════════════════════════════════════

def run_system_diagnostics():
    print("\n" + "█" * 70)
    print("█   VACUUM ENGINE v3.6: INTEGRITY PROTOCOL INITIATED...".center(70) + "█")
    print("█" * 70 + "\n")
    
    # 1. Initialize
    try:
        cfg = VacuumConfig(N=14, chi_max=32, dt=0.02)
        mps = MPS(cfg.N, cfg.chi_max)
        tebd = TEBD(mps, cfg)
        print("[OK] Kernel Initialization")
    except Exception as e:
        print(f"[FAIL] Kernel Crash: {e}")
        return

    # 2. Test Cooling (Imaginary Time)
    E_start = tebd.get_energy()
    for _ in range(50): tebd.sweep(imaginary=True)
    E_cool = tebd.get_energy()
    
    print(f"[TEST 1] Thermodynamic Cooling: {E_start:.2f} -> {E_cool:.2f}")
    if E_cool > E_start:
        print("   ❌ CRITICAL: System is heating up in imaginary time!")
        return
    
    # 3. Test Norm Restoration
    n_raw = mps.get_norm_full()
    mps.global_restore_norm()
    n_fixed = mps.get_norm_full()
    print(f"[TEST 2] Gauge Fixing (Norm): {n_raw:.4f} -> {n_fixed:.8f}")
    
    if abs(n_fixed - 1.0) > 1e-9:
        print("   ❌ CRITICAL: Unitarity violation (Norm != 1)")
        return

    # 4. Test Conservation Laws (Real Time)
    print("[TEST 3] Conservation Laws (Real Time Evolution)...")
    E0 = tebd.get_energy()
    norms = []
    energies = []
    
    for _ in range(50):
        tebd.sweep(imaginary=False)
        norms.append(mps.get_norm_full())
        energies.append(tebd.get_energy())
        
    max_norm_err = np.max(np.abs(np.array(norms) - 1.0))
    max_energy_err = np.max(np.abs(np.array(energies) - E0) / abs(E0))
    
    print(f"   Max Probability Leak: {max_norm_err:.2e} (Tol: 1e-6)")
    print(f"   Max Energy Drift:     {max_energy_err:.2e} (Tol: 5e-3)")
    
    if max_norm_err < 1e-6 and max_energy_err < 5e-3:
        print("\n" + "="*70)
        print("✅ SYSTEM STATUS: GREEN. MATHEMATICS IS SOUND.")
        print("   You may proceed with Hypotheses 11-14.")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ SYSTEM STATUS: RED. DO NOT PROCEED.")
        print("="*70)

if __name__ == "__main__":
    run_system_diagnostics()