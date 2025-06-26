"""
Quantum Optics Tools for AMO Physics
Advanced tools for cavity QED, quantum entanglement, and photon statistics.
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.linalg import expm
from scipy.integrate import solve_ivp

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    logging.warning("QuTiP not available, using basic NumPy implementation")

from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class QuantumState:
    """Represents a quantum state in the cavity QED system."""
    density_matrix: np.ndarray
    time: float
    atom_populations: np.ndarray
    photon_statistics: Dict[str, float]
    entanglement_measures: Dict[str, float]


@dataclass
class CavityQEDResult:
    """Container for cavity QED simulation results."""
    times: np.ndarray
    states: List[QuantumState]
    atom_populations: np.ndarray
    photon_numbers: np.ndarray
    g2_correlations: np.ndarray
    metadata: Dict[str, Any]


class JaynesCummingsModel:
    """Jaynes-Cummings model for cavity quantum electrodynamics."""
    
    def __init__(self, coupling_strength: float, cavity_frequency: float,
                 atomic_frequency: float, max_photons: int = 10):
        """
        Initialize Jaynes-Cummings model.
        
        Args:
            coupling_strength: Atom-cavity coupling strength (rad/s)
            cavity_frequency: Cavity mode frequency (rad/s)
            atomic_frequency: Atomic transition frequency (rad/s)
            max_photons: Maximum photon number to consider
        """
        self.g = coupling_strength
        self.omega_c = cavity_frequency
        self.omega_a = atomic_frequency
        self.max_photons = max_photons
        self.dim_total = 2 * (max_photons + 1)  # 2 atom levels × (max_photons + 1) cavity levels
        
        # Build Hamiltonian
        self.hamiltonian = self._build_hamiltonian()
        
        # Detuning
        self.delta = atomic_frequency - cavity_frequency
    
    def _build_hamiltonian(self) -> np.ndarray:
        """Build the Jaynes-Cummings Hamiltonian."""
        if QUTIP_AVAILABLE:
            return self._build_qutip_hamiltonian()
        else:
            return self._build_numpy_hamiltonian()
    
    def _build_qutip_hamiltonian(self):
        """Build Hamiltonian using QuTiP."""
        # Operators
        a = qt.tensor(qt.destroy(self.max_photons + 1), qt.qeye(2))  # Cavity annihilation
        sigma_z = qt.tensor(qt.qeye(self.max_photons + 1), qt.sigmaz())  # Atomic σz
        sigma_plus = qt.tensor(qt.qeye(self.max_photons + 1), qt.sigmap())  # Atomic σ+
        sigma_minus = qt.tensor(qt.qeye(self.max_photons + 1), qt.sigmam())  # Atomic σ-
        
        # Hamiltonian components
        H_cavity = self.omega_c * a.dag() * a
        H_atom = 0.5 * self.omega_a * sigma_z
        H_interaction = self.g * (a.dag() * sigma_minus + a * sigma_plus)
        
        return H_cavity + H_atom + H_interaction
    
    def _build_numpy_hamiltonian(self) -> np.ndarray:
        """Build Hamiltonian using NumPy (simplified version)."""
        H = np.zeros((self.dim_total, self.dim_total), dtype=complex)
        
        for n in range(self.max_photons + 1):
            # Cavity energy: ω_c * n
            # Atom in ground state |g,n⟩
            idx_g = 2 * n
            H[idx_g, idx_g] = self.omega_c * n
            
            # Atom in excited state |e,n⟩
            idx_e = 2 * n + 1
            H[idx_e, idx_e] = self.omega_c * n + self.omega_a
            
            # Interaction terms
            if n > 0:
                # |g,n⟩ ↔ |e,n-1⟩
                idx_g_n = 2 * n
                idx_e_n_minus_1 = 2 * (n - 1) + 1
                H[idx_g_n, idx_e_n_minus_1] = self.g * np.sqrt(n)
                H[idx_e_n_minus_1, idx_g_n] = self.g * np.sqrt(n)
            
            if n < self.max_photons:
                # |e,n⟩ ↔ |g,n+1⟩
                idx_e_n = 2 * n + 1
                idx_g_n_plus_1 = 2 * (n + 1)
                H[idx_e_n, idx_g_n_plus_1] = self.g * np.sqrt(n + 1)
                H[idx_g_n_plus_1, idx_e_n] = self.g * np.sqrt(n + 1)
        
        return H
    
    def evolve(self, initial_state: str, evolution_time: float,
               time_points: int = 1000) -> CavityQEDResult:
        """
        Evolve the Jaynes-Cummings system.
        
        Args:
            initial_state: Initial state ('vacuum', 'coherent', 'fock')
            evolution_time: Evolution time (μs)
            time_points: Number of time points
            
        Returns:
            CavityQEDResult containing evolution data
        """
        if QUTIP_AVAILABLE:
            return self._evolve_qutip(initial_state, evolution_time, time_points)
        else:
            return self._evolve_numpy(initial_state, evolution_time, time_points)
    
    def _evolve_qutip(self, initial_state: str, evolution_time: float,
                     time_points: int) -> CavityQEDResult:
        """Evolve using QuTiP."""
        # Convert time to seconds
        evolution_time_s = evolution_time * 1e-6
        times = np.linspace(0, evolution_time_s, time_points)
        
        # Initial state
        if initial_state == "vacuum":
            psi0 = qt.tensor(qt.basis(self.max_photons + 1, 0), qt.basis(2, 0))  # |0,g⟩
        elif initial_state == "coherent":
            alpha = 2.0  # Coherent state parameter
            psi0 = qt.tensor(qt.coherent(self.max_photons + 1, alpha), qt.basis(2, 0))
        elif initial_state == "fock":
            n_photons = min(5, self.max_photons)
            psi0 = qt.tensor(qt.basis(self.max_photons + 1, n_photons), qt.basis(2, 0))
        else:
            raise ValueError(f"Unknown initial state: {initial_state}")
        
        # Time evolution
        result = qt.mesolve(self.hamiltonian, psi0, times, [], [])
        
        # Extract data
        states = []
        atom_populations = np.zeros((len(times), 2))
        photon_numbers = np.zeros(len(times))
        g2_correlations = np.zeros(len(times))
        
        for i, (t, state) in enumerate(zip(times, result.states)):
            rho = state * state.dag() if state.type == 'ket' else state
            
            # Atom populations
            P_ground = (qt.tensor(qt.qeye(self.max_photons + 1), qt.basis(2, 0) * qt.basis(2, 0).dag()) * rho).tr()
            P_excited = (qt.tensor(qt.qeye(self.max_photons + 1), qt.basis(2, 1) * qt.basis(2, 1).dag()) * rho).tr()
            atom_populations[i] = [np.real(P_ground), np.real(P_excited)]
            
            # Photon number
            n_op = qt.tensor(qt.num(self.max_photons + 1), qt.qeye(2))
            photon_numbers[i] = np.real((n_op * rho).tr())
            
            # g^(2)(0) correlation
            a_op = qt.tensor(qt.destroy(self.max_photons + 1), qt.qeye(2))
            n_avg = photon_numbers[i]
            if n_avg > 0:
                g2_correlations[i] = np.real((a_op.dag() * a_op.dag() * a_op * a_op * rho).tr()) / (n_avg**2)
            else:
                g2_correlations[i] = 0
            
            # Entanglement measures
            entanglement = self._calculate_entanglement_qutip(rho)
            
            states.append(QuantumState(
                density_matrix=rho.full(),
                time=t,
                atom_populations=atom_populations[i],
                photon_statistics={"mean_photons": photon_numbers[i], "g2": g2_correlations[i]},
                entanglement_measures=entanglement
            ))
        
        metadata = {
            "coupling_strength": self.g,
            "cavity_frequency": self.omega_c,
            "atomic_frequency": self.omega_a,
            "detuning": self.delta,
            "max_photons": self.max_photons,
            "initial_state": initial_state,
            "solver": "qutip"
        }
        
        return CavityQEDResult(
            times=times * 1e6,  # Convert back to μs
            states=states,
            atom_populations=atom_populations,
            photon_numbers=photon_numbers,
            g2_correlations=g2_correlations,
            metadata=metadata
        )
    
    def _evolve_numpy(self, initial_state: str, evolution_time: float,
                     time_points: int) -> CavityQEDResult:
        """Evolve using NumPy (simplified version)."""
        evolution_time_s = evolution_time * 1e-6
        times = np.linspace(0, evolution_time_s, time_points)
        
        # Initial state vector
        if initial_state == "vacuum":
            psi0 = np.zeros(self.dim_total, dtype=complex)
            psi0[0] = 1.0  # |g,0⟩
        elif initial_state == "fock":
            psi0 = np.zeros(self.dim_total, dtype=complex)
            n_photons = min(5, self.max_photons)
            psi0[2 * n_photons] = 1.0  # |g,n⟩
        else:
            # Default to vacuum
            psi0 = np.zeros(self.dim_total, dtype=complex)
            psi0[0] = 1.0
        
        states = []
        atom_populations = np.zeros((len(times), 2))
        photon_numbers = np.zeros(len(times))
        
        for i, t in enumerate(times):
            # Time evolution
            U = expm(-1j * self.hamiltonian * t)
            psi_t = U @ psi0
            
            # Calculate populations
            P_ground = 0
            P_excited = 0
            mean_photons = 0
            
            for n in range(self.max_photons + 1):
                idx_g = 2 * n
                idx_e = 2 * n + 1
                
                P_ground += np.abs(psi_t[idx_g])**2
                P_excited += np.abs(psi_t[idx_e])**2
                mean_photons += n * (np.abs(psi_t[idx_g])**2 + np.abs(psi_t[idx_e])**2)
            
            atom_populations[i] = [P_ground, P_excited]
            photon_numbers[i] = mean_photons
            
            # Density matrix
            rho = np.outer(psi_t, np.conj(psi_t))
            
            states.append(QuantumState(
                density_matrix=rho,
                time=t,
                atom_populations=atom_populations[i],
                photon_statistics={"mean_photons": mean_photons, "g2": 1.0},
                entanglement_measures={"concurrence": 0.0}
            ))
        
        metadata = {
            "coupling_strength": self.g,
            "cavity_frequency": self.omega_c,
            "atomic_frequency": self.omega_a,
            "detuning": self.delta,
            "max_photons": self.max_photons,
            "initial_state": initial_state,
            "solver": "numpy"
        }
        
        return CavityQEDResult(
            times=times * 1e6,
            states=states,
            atom_populations=atom_populations,
            photon_numbers=photon_numbers,
            g2_correlations=np.ones(len(times)),
            metadata=metadata
        )
    
    def _calculate_entanglement_qutip(self, rho) -> Dict[str, float]:
        """Calculate entanglement measures using QuTiP."""
        try:
            # Partial trace over cavity
            rho_atom = rho.ptrace(1)
            
            # Von Neumann entropy
            eigenvals = rho_atom.eigenenergies()
            eigenvals = eigenvals[eigenvals > 1e-12]  # Remove zeros
            entropy = -np.sum(eigenvals * np.log2(eigenvals)) if len(eigenvals) > 0 else 0
            
            # Linear entropy (measure of mixedness)
            linear_entropy = 1 - (rho_atom * rho_atom).tr()
            
            return {
                "von_neumann_entropy": float(np.real(entropy)),
                "linear_entropy": float(np.real(linear_entropy)),
            }
        except:
            return {"von_neumann_entropy": 0.0, "linear_entropy": 0.0}


class PhotonStatistics:
    """Tools for analyzing photon statistics and correlations."""
    
    @staticmethod
    def g2_correlation(photon_counts: np.ndarray, tau: float = 0) -> float:
        """
        Calculate second-order correlation function g^(2)(τ).
        
        Args:
            photon_counts: Array of photon count measurements
            tau: Time delay (for τ=0, gives instantaneous correlation)
            
        Returns:
            g^(2)(τ) value
        """
        if tau == 0:
            # Instantaneous g^(2)(0)
            n_mean = np.mean(photon_counts)
            n2_mean = np.mean(photon_counts**2)
            
            if n_mean > 0:
                return n2_mean / (n_mean**2)
            else:
                return 0.0
        else:
            # For non-zero τ, would need time-resolved data
            return 1.0  # Placeholder
    
    @staticmethod
    def mandel_q_parameter(photon_counts: np.ndarray) -> float:
        """
        Calculate Mandel Q parameter for photon statistics classification.
        
        Args:
            photon_counts: Array of photon count measurements
            
        Returns:
            Mandel Q parameter
        """
        n_mean = np.mean(photon_counts)
        n_var = np.var(photon_counts)
        
        if n_mean > 0:
            return (n_var - n_mean) / n_mean
        else:
            return 0.0
    
    @staticmethod
    def classify_light(photon_counts: np.ndarray) -> Dict[str, Any]:
        """
        Classify the type of light based on photon statistics.
        
        Args:
            photon_counts: Array of photon count measurements
            
        Returns:
            Dictionary with classification results
        """
        g2 = PhotonStatistics.g2_correlation(photon_counts)
        Q = PhotonStatistics.mandel_q_parameter(photon_counts)
        
        # Classification
        if g2 < 1 and Q < 0:
            light_type = "sub-Poissonian (antibunched)"
        elif abs(g2 - 1) < 0.1 and abs(Q) < 0.1:
            light_type = "Poissonian (coherent)"
        elif g2 > 1 and Q > 0:
            light_type = "super-Poissonian (bunched)"
        else:
            light_type = "mixed or thermal"
        
        return {
            "g2_0": g2,
            "mandel_q": Q,
            "classification": light_type,
            "mean_photons": np.mean(photon_counts),
            "variance": np.var(photon_counts),
            "fano_factor": np.var(photon_counts) / np.mean(photon_counts) if np.mean(photon_counts) > 0 else 0
        }


# Tool handler functions
async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle quantum optics tool calls."""
    try:
        if name == "cavity_qed":
            return await cavity_qed(**arguments)
        elif name == "photon_statistics":
            return await photon_statistics(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in quantum optics tool {name}: {str(e)}")
        raise


async def cavity_qed(coupling_strength: float, cavity_frequency: float,
                   atomic_frequency: float, max_photons: int = 10,
                   evolution_time: float = 10.0, initial_state: str = "vacuum") -> Dict[str, Any]:
    """Simulate cavity quantum electrodynamics (Jaynes-Cummings model)."""
    logger.info(f"Cavity QED simulation: g={coupling_strength}, Δ={atomic_frequency-cavity_frequency}")
    
    # Initialize model
    jc_model = JaynesCummingsModel(
        coupling_strength=coupling_strength,
        cavity_frequency=cavity_frequency,
        atomic_frequency=atomic_frequency,
        max_photons=max_photons
    )
    
    # Run simulation
    result = jc_model.evolve(initial_state, evolution_time)
    
    # Calculate additional properties
    rabi_frequency_vacuum = 2 * coupling_strength  # Vacuum Rabi frequency
    cooperativity = coupling_strength**2 / (abs(jc_model.delta) + 1e-10)
    strong_coupling = coupling_strength > abs(jc_model.delta)
    
    # Analyze dynamics
    max_excited_pop = np.max(result.atom_populations[:, 1])
    oscillation_period = None
    if strong_coupling and abs(jc_model.delta) < coupling_strength:
        oscillation_period = 2 * np.pi / rabi_frequency_vacuum
    
    return {
        "success": True,
        "result_type": "cavity_qed_simulation",
        "time_evolution": {
            "times_us": result.times.tolist(),
            "ground_population": result.atom_populations[:, 0].tolist(),
            "excited_population": result.atom_populations[:, 1].tolist(),
            "mean_photon_number": result.photon_numbers.tolist(),
            "g2_correlations": result.g2_correlations.tolist(),
        },
        "system_parameters": {
            "coupling_strength": coupling_strength,
            "cavity_frequency": cavity_frequency,
            "atomic_frequency": atomic_frequency,
            "detuning": jc_model.delta,
            "max_photons": max_photons,
            "initial_state": initial_state,
        },
        "characteristic_frequencies": {
            "vacuum_rabi_frequency": rabi_frequency_vacuum,
            "detuning": jc_model.delta,
            "cooperativity": cooperativity,
        },
        "analysis": {
            "strong_coupling_regime": strong_coupling,
            "max_excited_population": float(max_excited_pop),
            "oscillation_period_us": oscillation_period,
            "average_photon_number": float(np.mean(result.photon_numbers)),
            "photon_antibunching": np.any(result.g2_correlations < 1),
        },
        "entanglement": {
            "final_entanglement": result.states[-1].entanglement_measures,
            "max_entanglement": max([s.entanglement_measures.get("von_neumann_entropy", 0) for s in result.states]),
        },
        "metadata": result.metadata,
    }


async def photon_statistics(photon_counts: List[float], analysis_type: str = "full") -> Dict[str, Any]:
    """Analyze photon statistics and correlations."""
    logger.info(f"Analyzing photon statistics for {len(photon_counts)} measurements")
    
    counts_array = np.array(photon_counts)
    
    # Basic statistics
    basic_stats = {
        "mean": float(np.mean(counts_array)),
        "variance": float(np.var(counts_array)),
        "std_dev": float(np.std(counts_array)),
        "min": float(np.min(counts_array)),
        "max": float(np.max(counts_array)),
    }
    
    # Photon statistics analysis
    photon_analysis = PhotonStatistics.classify_light(counts_array)
    
    # Higher-order moments
    if len(counts_array) > 0:
        skewness = float(np.mean((counts_array - basic_stats["mean"])**3) / basic_stats["std_dev"]**3)
        kurtosis = float(np.mean((counts_array - basic_stats["mean"])**4) / basic_stats["std_dev"]**4 - 3)
    else:
        skewness = 0.0
        kurtosis = 0.0
    
    # Probability distribution
    unique_counts, count_frequencies = np.unique(counts_array, return_counts=True)
    probabilities = count_frequencies / len(counts_array)
    
    return {
        "success": True,
        "result_type": "photon_statistics_analysis",
        "basic_statistics": basic_stats,
        "photon_classification": photon_analysis,
        "higher_moments": {
            "skewness": skewness,
            "kurtosis": kurtosis,
        },
        "probability_distribution": {
            "values": unique_counts.tolist(),
            "probabilities": probabilities.tolist(),
        },
        "quantum_properties": {
            "sub_poissonian": photon_analysis["mandel_q"] < -0.1,
            "super_poissonian": photon_analysis["mandel_q"] > 0.1,
            "antibunched": photon_analysis["g2_0"] < 0.9,
            "bunched": photon_analysis["g2_0"] > 1.1,
        },
        "summary": {
            "light_type": photon_analysis["classification"],
            "quantum_nature": "quantum" if photon_analysis["g2_0"] < 1 else "classical",
            "measurement_quality": "good" if len(counts_array) > 100 else "limited",
        }
    }