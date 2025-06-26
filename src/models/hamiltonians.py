"""
Common Hamiltonians for AMO Physics
Library of standard Hamiltonians used in atomic, molecular, and optical physics.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.sparse import csr_matrix, diags
from scipy.linalg import expm
import logging

from ..resources.constants import FundamentalConstants, AtomicUnits

logger = logging.getLogger(__name__)


class PauliMatrices:
    """Pauli matrices and common two-level system operators."""
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    # Raising and lowering operators
    sigma_plus = 0.5 * (sigma_x + 1j * sigma_y)
    sigma_minus = 0.5 * (sigma_x - 1j * sigma_y)
    
    # Projection operators
    ground_projector = np.array([[1, 0], [0, 0]], dtype=complex)
    excited_projector = np.array([[0, 0], [0, 1]], dtype=complex)


class HarmonicOscillator:
    """Harmonic oscillator Hamiltonians and operators."""
    
    @staticmethod
    def creation_operator(n_max: int) -> np.ndarray:
        """
        Creation operator for harmonic oscillator.
        
        Args:
            n_max: Maximum number state
            
        Returns:
            Creation operator matrix
        """
        a_dag = np.zeros((n_max + 1, n_max + 1), dtype=complex)
        for n in range(n_max):
            a_dag[n + 1, n] = np.sqrt(n + 1)
        return a_dag
    
    @staticmethod
    def annihilation_operator(n_max: int) -> np.ndarray:
        """
        Annihilation operator for harmonic oscillator.
        
        Args:
            n_max: Maximum number state
            
        Returns:
            Annihilation operator matrix
        """
        a = np.zeros((n_max + 1, n_max + 1), dtype=complex)
        for n in range(1, n_max + 1):
            a[n - 1, n] = np.sqrt(n)
        return a
    
    @staticmethod
    def number_operator(n_max: int) -> np.ndarray:
        """
        Number operator for harmonic oscillator.
        
        Args:
            n_max: Maximum number state
            
        Returns:
            Number operator matrix
        """
        return np.diag(range(n_max + 1), dtype=complex)
    
    @staticmethod
    def position_operator(n_max: int, x_zpf: float = 1.0) -> np.ndarray:
        """
        Position operator for harmonic oscillator.
        
        Args:
            n_max: Maximum number state
            x_zpf: Zero-point fluctuation scale
            
        Returns:
            Position operator matrix
        """
        a = HarmonicOscillator.annihilation_operator(n_max)
        a_dag = HarmonicOscillator.creation_operator(n_max)
        return x_zpf / np.sqrt(2) * (a + a_dag)
    
    @staticmethod
    def momentum_operator(n_max: int, p_zpf: float = 1.0) -> np.ndarray:
        """
        Momentum operator for harmonic oscillator.
        
        Args:
            n_max: Maximum number state
            p_zpf: Zero-point momentum scale
            
        Returns:
            Momentum operator matrix
        """
        a = HarmonicOscillator.annihilation_operator(n_max)
        a_dag = HarmonicOscillator.creation_operator(n_max)
        return 1j * p_zpf / np.sqrt(2) * (a_dag - a)
    
    @staticmethod
    def hamiltonian(n_max: int, omega: float = 1.0) -> np.ndarray:
        """
        Harmonic oscillator Hamiltonian.
        
        Args:
            n_max: Maximum number state
            omega: Angular frequency
            
        Returns:
            Harmonic oscillator Hamiltonian
        """
        n_op = HarmonicOscillator.number_operator(n_max)
        return FundamentalConstants.hbar * omega * (n_op + 0.5 * np.eye(n_max + 1))


class AtomicHamiltonians:
    """Common atomic Hamiltonians."""
    
    @staticmethod
    def two_level_system(omega_0: float, omega_l: float, rabi_frequency: float) -> np.ndarray:
        """
        Two-level atom in electromagnetic field (rotating wave approximation).
        
        Args:
            omega_0: Atomic transition frequency
            omega_l: Laser frequency
            rabi_frequency: Rabi frequency
            
        Returns:
            Two-level system Hamiltonian
        """
        detuning = omega_l - omega_0
        H = np.array([
            [0, 0.5 * rabi_frequency],
            [0.5 * rabi_frequency, detuning]
        ], dtype=complex)
        return FundamentalConstants.hbar * H
    
    @staticmethod
    def three_level_lambda(omega_12: float, omega_23: float, 
                          omega_l1: float, omega_l2: float,
                          rabi_1: float, rabi_2: float) -> np.ndarray:
        """
        Three-level Lambda system.
        
        Args:
            omega_12: Transition frequency |1⟩ ↔ |2⟩
            omega_23: Transition frequency |2⟩ ↔ |3⟩
            omega_l1: Laser 1 frequency
            omega_l2: Laser 2 frequency
            rabi_1: Rabi frequency for transition |1⟩ ↔ |2⟩
            rabi_2: Rabi frequency for transition |2⟩ ↔ |3⟩
            
        Returns:
            Three-level Lambda Hamiltonian
        """
        delta_1 = omega_l1 - omega_12
        delta_2 = omega_l2 - omega_23
        
        H = np.array([
            [0, 0.5 * rabi_1, 0],
            [0.5 * rabi_1, delta_1, 0.5 * rabi_2],
            [0, 0.5 * rabi_2, delta_1 + delta_2]
        ], dtype=complex)
        
        return FundamentalConstants.hbar * H
    
    @staticmethod
    def three_level_ladder(omega_12: float, omega_23: float,
                          omega_l1: float, omega_l2: float,
                          rabi_1: float, rabi_2: float) -> np.ndarray:
        """
        Three-level ladder system.
        
        Args:
            omega_12: Transition frequency |1⟩ ↔ |2⟩
            omega_23: Transition frequency |2⟩ ↔ |3⟩
            omega_l1: Laser 1 frequency
            omega_l2: Laser 2 frequency
            rabi_1: Rabi frequency for transition |1⟩ ↔ |2⟩
            rabi_2: Rabi frequency for transition |2⟩ ↔ |3⟩
            
        Returns:
            Three-level ladder Hamiltonian
        """
        delta_1 = omega_l1 - omega_12
        delta_2 = omega_l2 - omega_23
        
        H = np.array([
            [0, 0.5 * rabi_1, 0],
            [0.5 * rabi_1, delta_1, 0.5 * rabi_2],
            [0, 0.5 * rabi_2, delta_1 + delta_2]
        ], dtype=complex)
        
        return FundamentalConstants.hbar * H
    
    @staticmethod
    def zeeman_effect(B_field: float, g_factor: float = 2.0, 
                     j_quantum: float = 0.5) -> np.ndarray:
        """
        Zeeman effect Hamiltonian.
        
        Args:
            B_field: Magnetic field strength (Tesla)
            g_factor: Landé g-factor
            j_quantum: Total angular momentum quantum number
            
        Returns:
            Zeeman Hamiltonian
        """
        # Number of Zeeman sublevels
        n_levels = int(2 * j_quantum + 1)
        
        # Magnetic quantum numbers
        m_j = np.arange(-j_quantum, j_quantum + 1)
        
        # Zeeman energies
        zeeman_energies = g_factor * FundamentalConstants.mu_B * B_field * m_j
        
        return np.diag(zeeman_energies)
    
    @staticmethod
    def stark_effect(E_field: float, polarizability: float) -> np.ndarray:
        """
        AC Stark effect Hamiltonian (scalar polarizability).
        
        Args:
            E_field: Electric field amplitude (V/m)
            polarizability: Scalar polarizability (C⋅m²/V)
            
        Returns:
            Stark shift Hamiltonian
        """
        stark_shift = -0.5 * polarizability * E_field**2
        return stark_shift * np.eye(2)  # For two-level system


class CavityQEDHamiltonians:
    """Hamiltonians for cavity quantum electrodynamics."""
    
    @staticmethod
    def jaynes_cummings(omega_a: float, omega_c: float, g: float, n_max: int) -> np.ndarray:
        """
        Jaynes-Cummings Hamiltonian.
        
        Args:
            omega_a: Atomic transition frequency
            omega_c: Cavity mode frequency
            g: Atom-cavity coupling strength
            n_max: Maximum photon number
            
        Returns:
            Jaynes-Cummings Hamiltonian
        """
        dim_total = 2 * (n_max + 1)
        H = np.zeros((dim_total, dim_total), dtype=complex)
        
        for n in range(n_max + 1):
            # Ground state |g,n⟩
            idx_g = 2 * n
            H[idx_g, idx_g] = FundamentalConstants.hbar * omega_c * n
            
            # Excited state |e,n⟩
            idx_e = 2 * n + 1
            H[idx_e, idx_e] = FundamentalConstants.hbar * (omega_c * n + omega_a)
            
            # Interaction terms
            if n > 0:
                # |g,n⟩ ↔ |e,n-1⟩
                idx_g_n = 2 * n
                idx_e_n_minus_1 = 2 * (n - 1) + 1
                coupling = FundamentalConstants.hbar * g * np.sqrt(n)
                H[idx_g_n, idx_e_n_minus_1] = coupling
                H[idx_e_n_minus_1, idx_g_n] = coupling
            
            if n < n_max:
                # |e,n⟩ ↔ |g,n+1⟩
                idx_e_n = 2 * n + 1
                idx_g_n_plus_1 = 2 * (n + 1)
                coupling = FundamentalConstants.hbar * g * np.sqrt(n + 1)
                H[idx_e_n, idx_g_n_plus_1] = coupling
                H[idx_g_n_plus_1, idx_e_n] = coupling
        
        return H
    
    @staticmethod
    def tavis_cummings(omega_a: float, omega_c: float, g: float, 
                      n_atoms: int, n_max: int) -> np.ndarray:
        """
        Tavis-Cummings Hamiltonian (multiple atoms in cavity).
        
        Args:
            omega_a: Atomic transition frequency
            omega_c: Cavity mode frequency
            g: Single-atom cavity coupling strength
            n_atoms: Number of atoms
            n_max: Maximum photon number
            
        Returns:
            Tavis-Cummings Hamiltonian (simplified version)
        """
        # This is a simplified version for identical atoms
        # Full implementation would require careful treatment of atomic states
        
        # Collective atomic operators
        j_max = n_atoms / 2
        n_atomic_states = int(2 * j_max + 1)
        
        # Simplified Hamiltonian (placeholder)
        dim_total = n_atomic_states * (n_max + 1)
        H = np.zeros((dim_total, dim_total), dtype=complex)
        
        # Add atomic and cavity energies (simplified)
        for i in range(dim_total):
            H[i, i] = FundamentalConstants.hbar * omega_a  # Placeholder
        
        return H


class MolecularHamiltonians:
    """Hamiltonians for molecular systems."""
    
    @staticmethod
    def rigid_rotor(B_rot: float, j_max: int) -> np.ndarray:
        """
        Rigid rotor Hamiltonian.
        
        Args:
            B_rot: Rotational constant (rad/s)
            j_max: Maximum rotational quantum number
            
        Returns:
            Rigid rotor Hamiltonian
        """
        dim = sum(2 * j + 1 for j in range(j_max + 1))
        H = np.zeros((dim, dim), dtype=complex)
        
        idx = 0
        for j in range(j_max + 1):
            for m in range(-j, j + 1):
                H[idx, idx] = FundamentalConstants.hbar * B_rot * j * (j + 1)
                idx += 1
        
        return H
    
    @staticmethod
    def harmonic_vibrational(omega_vib: float, v_max: int) -> np.ndarray:
        """
        Harmonic vibrational Hamiltonian.
        
        Args:
            omega_vib: Vibrational frequency (rad/s)
            v_max: Maximum vibrational quantum number
            
        Returns:
            Harmonic vibrational Hamiltonian
        """
        v_levels = np.arange(v_max + 1)
        energies = FundamentalConstants.hbar * omega_vib * (v_levels + 0.5)
        return np.diag(energies)
    
    @staticmethod
    def morse_potential(D_e: float, a: float, v_max: int) -> np.ndarray:
        """
        Morse potential Hamiltonian (approximate).
        
        Args:
            D_e: Dissociation energy
            a: Morse parameter
            v_max: Maximum vibrational quantum number
            
        Returns:
            Morse potential Hamiltonian
        """
        v_levels = np.arange(v_max + 1)
        
        # Morse energy levels (approximate)
        omega_e = a * np.sqrt(2 * D_e / FundamentalConstants.m_e)  # Simplified
        omega_e_x_e = omega_e**2 / (4 * D_e)  # Anharmonicity constant
        
        energies = (FundamentalConstants.hbar * omega_e * (v_levels + 0.5) - 
                   FundamentalConstants.hbar * omega_e_x_e * (v_levels + 0.5)**2)
        
        return np.diag(energies)


class TimeEvolutionOperators:
    """Time evolution operators for quantum systems."""
    
    @staticmethod
    def unitary_evolution(hamiltonian: np.ndarray, time: float) -> np.ndarray:
        """
        Calculate unitary time evolution operator U(t) = exp(-iHt/ℏ).
        
        Args:
            hamiltonian: System Hamiltonian
            time: Evolution time
            
        Returns:
            Unitary evolution operator
        """
        return expm(-1j * hamiltonian * time / FundamentalConstants.hbar)
    
    @staticmethod
    def time_ordered_evolution(hamiltonians: List[np.ndarray], 
                             times: List[float]) -> np.ndarray:
        """
        Calculate time-ordered evolution for time-dependent Hamiltonian.
        
        Args:
            hamiltonians: List of Hamiltonians
            times: List of time intervals
            
        Returns:
            Time-ordered evolution operator
        """
        U_total = np.eye(hamiltonians[0].shape[0], dtype=complex)
        
        for H, dt in zip(hamiltonians, times):
            U_step = TimeEvolutionOperators.unitary_evolution(H, dt)
            U_total = U_step @ U_total
        
        return U_total
    
    @staticmethod
    def suzuki_trotter_evolution(H_kinetic: np.ndarray, H_potential: np.ndarray,
                                time: float, n_steps: int) -> np.ndarray:
        """
        Suzuki-Trotter decomposition for split-operator evolution.
        
        Args:
            H_kinetic: Kinetic energy Hamiltonian
            H_potential: Potential energy Hamiltonian
            time: Total evolution time
            n_steps: Number of Trotter steps
            
        Returns:
            Approximate evolution operator
        """
        dt = time / n_steps
        
        # First-order Trotter decomposition
        U_kinetic = expm(-1j * H_kinetic * dt / (2 * FundamentalConstants.hbar))
        U_potential = expm(-1j * H_potential * dt / FundamentalConstants.hbar)
        
        # Single Trotter step
        U_step = U_kinetic @ U_potential @ U_kinetic
        
        # Apply n_steps times
        U_total = np.eye(H_kinetic.shape[0], dtype=complex)
        for _ in range(n_steps):
            U_total = U_step @ U_total
        
        return U_total


def build_tensor_product_hamiltonian(hamiltonians: List[np.ndarray]) -> np.ndarray:
    """
    Build tensor product Hamiltonian for composite systems.
    
    Args:
        hamiltonians: List of Hamiltonians for each subsystem
        
    Returns:
        Total Hamiltonian for composite system
    """
    if len(hamiltonians) == 1:
        return hamiltonians[0]
    
    # Start with first Hamiltonian
    H_total = hamiltonians[0]
    
    # Add subsequent Hamiltonians
    for i, H in enumerate(hamiltonians[1:], 1):
        # Identity matrices for other subsystems
        left_identity = np.eye(H_total.shape[0])
        right_identity = np.eye(np.prod([h.shape[0] for h in hamiltonians[i+1:]]))
        
        # Current Hamiltonian contribution
        H_current = np.kron(np.kron(left_identity, H), right_identity)
        
        # Add to total
        if i == 1:
            H_total = np.kron(H_total, np.eye(H.shape[0])) + H_current
        else:
            H_total = H_total + H_current
    
    return H_total