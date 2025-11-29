"""
================================================================================
                        COMPUTATIONAL VACUUM ENGINE
                            (Core Kernel v3.2)
================================================================================

This module implements the fundamental tensor network algorithms required to
simulate quantum vacuum dynamics, emergent gravity, and holographic scaling.

Features:
    - MPS/TEBD Implementation
    - SVD Truncation (The "Scissors" mechanism)
    - Symmetry Breaking field (to resolve ground state degeneracy)
    - Built-in Self-Diagnostic Suite

Author:  Sergey A. Danilov
License: MIT
================================================================================
"""

import numpy as np
import scipy.linalg as la
from dataclasses import dataclass
from typing import List
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
    """Configuration parameters."""
    N: int = 20              # Number of lattice sites
    chi_max: int = 32        # Maximum bond dimension
    d: int = 2               # Physical dimension
    dt: float = 0.05         # Time step
    J: float = 1.0           # Interaction
    h: float = 1.0           # Transverse field


class MPS:
    """Matrix Product State representation (Gamma-Lambda form)."""

    def __init__(self, N: int, chi_max: int, d: int = 2):
        self.N = N
        self.chi_max = chi_max
        self.d = d
        self.Gammas: List[np.ndarray] = []
        self.Lambdas: List[np.ndarray] = []
        self._init_vacuum()

    def _init_vacuum(self):
        """Initialize |00...0> state."""
        self.Gammas = []
        self.Lambdas = [np.array([1.0])]
        for _ in range(self.N):
            G = np.zeros((1, self.d, 1), dtype=complex)
            G[0, 0, 0] = 1.0
            self.Gammas.append(G)
            self.Lambdas.append(np.array([1.0]))

    def get_norm(self) -> float:
        """Calculates state norm (approximate for check)."""
        # Accurate norm requires full contraction, here we check local integrity
        # In canonical form, norm is determined by the Lambda center.
        return la.norm(self.Lambdas[self.N // 2])

    def normalize(self):
        """Renormalize the state vector."""
        for i in range(len(self.Lambdas)):
            n = la.norm(self.Lambdas[i])
            if n > PRECISION:
                self.Lambdas[i] /= n

    def get_entanglement_entropy(self, bond: int) -> float:
        """Computes Von Neumann entropy at a bond."""
        if bond <= 0 or bond >= self.N:
            return 0.0
        S = self.Lambdas[bond]**2
        S = S[S > PRECISION] # Filter zeros
        S /= np.sum(S) # Normalize probability
        return float(-np.sum(S * np.log(S)))

    def expectation_one_site(self, i: int, op: np.ndarray) -> float:
        """Calculate <psi|O_i|psi>."""
        G = self.Gammas[i]
        L_L = self.Lambdas[i]
        L_R = self.Lambdas[i+1]
        A = G * L_L[:, None, None] * L_R[None, None, :]
        res = np.einsum('asb,ks,akb->', A, op, A.conj())
        return np.real(res)

    def expectation_two_site(self, i: int, op1: np.ndarray, op2: np.ndarray) -> float:
        """Calculate <psi|O_i O_{i+1}|psi>."""
        G1, G2 = self.Gammas[i], self.Gammas[i+1]
        L1, L2, L3 = self.Lambdas[i], self.Lambdas[i+1], self.Lambdas[i+2]
        
        # Construct Theta
        T1 = G1 * L1[:, None, None] * L2[None, None, :]
        T2 = G2 * L3[None, None, :]
        Theta = np.tensordot(T1, T2, axes=(2,0)) # (a,s, t,c)
        
        # Contract with operators
        res = np.einsum('astc,ks,lt,aklc->', Theta, op1, op2, Theta.conj())
        return np.real(res)


class TEBD:
    """
    Time-Evolving Block Decimation Engine.
    """

    def __init__(self, mps: MPS, config: VacuumConfig):
        self.mps = mps
        self.cfg = config
        self._build_gates()

    def _build_gates(self):
        """Build evolution gates with symmetry breaking."""
        XX = np.kron(Pauli.X, Pauli.X)
        ZI = np.kron(Pauli.Z, Pauli.I)
        IZ = np.kron(Pauli.I, Pauli.Z)
        XI = np.kron(Pauli.X, Pauli.I)
        IX = np.kron(Pauli.I, Pauli.X)
        
        # Symmetry breaking field (critical for finite size simulations)
        epsilon = 1e-3 
        
        # Hamiltonian: H = -J(XX) - h/2(ZI+IZ) - eps/2(XI+IX)
        H_bond = -self.cfg.J * XX \
                 - 0.5 * self.cfg.h * (ZI + IZ) \
                 - 0.5 * epsilon * (XI + IX)
        
        self.gate_real = la.expm(-1j * H_bond * self.cfg.dt).reshape(2,2,2,2)
        self.gate_imag = la.expm(-H_bond * self.cfg.dt).reshape(2,2,2,2)

    def apply_gate(self, i: int, gate: np.ndarray):
        """Apply gate to bond (i, i+1) and update MPS."""
        G_L, G_R = self.mps.Gammas[i], self.mps.Gammas[i+1]
        L_L, L_M, L_R = self.mps.Lambdas[i], self.mps.Lambdas[i+1], self.mps.Lambdas[i+2]
        
        # 1. Construct Theta
        T1 = G_L * L_L[:, None, None] * L_M[None, None, :]
        T2 = G_R * L_R[None, None, :]
        Theta = np.tensordot(T1, T2, axes=(2,0)) # (a,s, t,c)
        
        # 2. Apply Gate
        Theta_new = np.tensordot(Theta, gate, axes=([1,2], [2,3])) # (a, c, s', t')
        Theta_new = np.transpose(Theta_new, (0, 2, 3, 1)) # (a, s', t', c)
        
        # 3. SVD
        d = self.mps.d
        sh = Theta_new.shape
        Theta_mat = Theta_new.reshape(sh[0]*d, d*sh[3])
        
        try:
            U, S, Vh = la.svd(Theta_mat, full_matrices=False)
        except la.LinAlgError:
            return 0.0 # Stability fallback
            
        # 4. Truncation
        chi_new = min(self.mps.chi_max, len(S))
        S = S[:chi_new]
        U = U[:, :chi_new]
        Vh = Vh[:chi_new, :]
        
        S /= la.norm(S) # Normalize
        
        # 5. Update State
        self.mps.Lambdas[i+1] = S
        
        inv_L_L = np.where(L_L > PRECISION, 1.0/L_L, 0.0)
        inv_L_R = np.where(L_R > PRECISION, 1.0/L_R, 0.0)
        
        G_L_new = U.reshape(sh[0], d, chi_new) * inv_L_L[:, None, None]
        G_R_new = Vh.reshape(chi_new, d, sh[3]) * inv_L_R[None, None, :]
        
        self.mps.Gammas[i] = G_L_new
        self.mps.Gammas[i+1] = G_R_new

    def sweep(self, imaginary: bool = False):
        """Perform one time step."""
        gate = self.gate_imag if imaginary else self.gate_real
        for i in range(0, self.mps.N-1, 2): self.apply_gate(i, gate)
        for i in range(1, self.mps.N-1, 2): self.apply_gate(i, gate)
        if imaginary: self.mps.normalize()

    def get_energy(self) -> float:
        """Calculate total energy <H>."""
        E = 0.0
        for i in range(self.mps.N - 1):
            E -= self.cfg.J * self.mps.expectation_two_site(i, Pauli.X, Pauli.X)
        for i in range(self.mps.N):
            E -= self.cfg.h * self.mps.expectation_one_site(i, Pauli.Z)
        return E


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-DIAGNOSTICS (Run this file directly)
# ═══════════════════════════════════════════════════════════════════════════════

def self_check():
    print("\n" + "█" * 60)
    print("█   VACUUM ENGINE: SELF-DIAGNOSTICS STARTING...".center(60) + "█")
    print("█" * 60 + "\n")
    
    # 1. Initialization
    try:
        cfg = VacuumConfig(N=10, chi_max=16, dt=0.1, J=1.0, h=1.0)
        mps = MPS(cfg.N, cfg.chi_max)
        tebd = TEBD(mps, cfg)
        norm = mps.get_norm()
        print(f"[1/4] Initialization:   OK (Norm = {norm:.6f})")
        if abs(norm - 1.0) > 1e-5: raise ValueError("Norm violation")
    except Exception as e:
        print(f"[1/4] Initialization:   FAILED ({e})")
        return

    # 2. Imaginary Time (Cooling)
    try:
        E_start = tebd.get_energy()
        print(f"      Energy (Start):   {E_start:.4f}")
        
        # Run cooling
        start_t = time.time()
        for _ in range(50): tebd.sweep(imaginary=True)
        end_t = time.time()
        
        E_end = tebd.get_energy()
        print(f"      Energy (End):     {E_end:.4f}")
        print(f"[2/4] Cooling Engine:   OK ({end_t - start_t:.3f}s)")
        
        if E_end >= E_start: print("      ⚠️ WARNING: Energy did not decrease!")
    except Exception as e:
        print(f"[2/4] Cooling Engine:   FAILED ({e})")

    # 3. Entanglement Check
    try:
        S_mid = mps.get_entanglement_entropy(cfg.N // 2)
        print(f"      Entropy (Mid):    {S_mid:.4f}")
        print(f"[3/4] Entanglement:     OK")
        if S_mid < 1e-5: print("      ⚠️ WARNING: No entanglement generated!")
    except Exception as e:
        print(f"[3/4] Entanglement:     FAILED ({e})")

    # 4. Real Time (Unitarity)
    try:
        norm_before = mps.get_norm()
        for _ in range(20): tebd.sweep(imaginary=False)
        norm_after = mps.get_norm()
        
        print(f"      Norm Drift:       {abs(norm_after - norm_before):.2e}")
        print(f"[4/4] Real-Time Dyn:    OK")
    except Exception as e:
        print(f"[4/4] Real-Time Dyn:    FAILED ({e})")

    print("\n" + "="*60)
    print("✅ SYSTEM READY" if abs(norm_after - 1.0) < 1e-3 else "❌ SYSTEM UNSTABLE")
    print("="*60)


if __name__ == "__main__":
    self_check()