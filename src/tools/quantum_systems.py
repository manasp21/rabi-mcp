"""
Quantum Systems Tools for AMO Physics
Advanced simulation tools for atomic, molecular, and optical quantum systems.
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

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
    """Represents a quantum state with metadata."""
    state_vector: np.ndarray
    time: float
    energy: Optional[float] = None
    populations: Optional[np.ndarray] = None
    coherences: Optional[np.ndarray] = None


@dataclass
class SimulationResult:
    """Container for simulation results."""
    times: np.ndarray
    states: List[QuantumState]
    populations: np.ndarray
    expectation_values: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class TwoLevelSystem:
    """Two-level atomic system simulator."""
    
    def __init__(self, rabi_frequency: float, detuning: float, decay_rate: float = 0.0):
        """
        Initialize two-level system.
        
        Args:
            rabi_frequency: Rabi frequency (rad/s)
            detuning: Laser detuning from resonance (rad/s)
            decay_rate: Spontaneous emission rate (rad/s)
        """
        self.rabi_frequency = rabi_frequency
        self.detuning = detuning
        self.decay_rate = decay_rate
        
        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.eye(2, dtype=complex)
        
        # Hamiltonian
        self.hamiltonian = self._build_hamiltonian()
    
    def _build_hamiltonian(self) -> np.ndarray:
        """Build the system Hamiltonian."""
        return 0.5 * (self.detuning * self.sigma_z + self.rabi_frequency * self.sigma_x)
    
    def evolve(self, initial_state: str, evolution_time: float, 
               time_points: int = 1000) -> SimulationResult:
        """
        Evolve the two-level system.
        
        Args:
            initial_state: Initial state ('ground', 'excited', 'superposition')
            evolution_time: Total evolution time (s)
            time_points: Number of time points
            
        Returns:
            SimulationResult containing evolution data
        """
        # Set initial state
        if initial_state == "ground":
            psi0 = np.array([1, 0], dtype=complex)
        elif initial_state == "excited":
            psi0 = np.array([0, 1], dtype=complex)
        elif initial_state == "superposition":
            psi0 = np.array([1, 1], dtype=complex) / np.sqrt(2)
        else:
            raise ValueError(f"Unknown initial state: {initial_state}")
        
        times = np.linspace(0, evolution_time, time_points)
        
        if self.decay_rate == 0:
            # Closed system evolution
            return self._evolve_closed_system(psi0, times)
        else:
            # Open system evolution with decay
            return self._evolve_open_system(psi0, times)
    
    def _evolve_closed_system(self, psi0: np.ndarray, times: np.ndarray) -> SimulationResult:
        """Evolve closed quantum system."""
        states = []
        populations = np.zeros((len(times), 2))
        
        for i, t in enumerate(times):
            # Time evolution operator
            U = self._time_evolution_operator(t)
            psi_t = U @ psi0
            
            # Calculate populations
            pops = np.abs(psi_t)**2
            populations[i] = pops
            
            states.append(QuantumState(
                state_vector=psi_t,
                time=t,
                populations=pops
            ))
        
        # Calculate expectation values
        expectation_values = {
            "sigma_x": np.array([np.real(np.conj(s.state_vector) @ self.sigma_x @ s.state_vector) 
                               for s in states]),
            "sigma_y": np.array([np.real(np.conj(s.state_vector) @ self.sigma_y @ s.state_vector) 
                               for s in states]),
            "sigma_z": np.array([np.real(np.conj(s.state_vector) @ self.sigma_z @ s.state_vector) 
                               for s in states]),
        }
        
        metadata = {
            "system_type": "two_level",
            "rabi_frequency": self.rabi_frequency,
            "detuning": self.detuning,
            "decay_rate": self.decay_rate,
            "evolution_type": "closed"
        }
        
        return SimulationResult(
            times=times,
            states=states,
            populations=populations,
            expectation_values=expectation_values,
            metadata=metadata
        )
    
    def _evolve_open_system(self, psi0: np.ndarray, times: np.ndarray) -> SimulationResult:
        """Evolve open quantum system with decay."""
        if not QUTIP_AVAILABLE:
            logger.warning("QuTiP not available, using simplified decay model")
            return self._evolve_with_simple_decay(psi0, times)
        
        # Convert to QuTiP format
        H = qt.Qobj(self.hamiltonian)
        c_ops = [np.sqrt(self.decay_rate) * qt.sigmam()]  # Lowering operator
        psi0_qt = qt.Qobj(psi0)
        
        # Solve master equation
        result = qt.mesolve(H, psi0_qt, times, c_ops, [qt.sigmax(), qt.sigmay(), qt.sigmaz()])
        
        states = []
        populations = np.zeros((len(times), 2))
        
        for i, (t, state) in enumerate(zip(times, result.states)):
            psi = state.full().flatten()
            pops = np.abs(psi)**2
            populations[i] = pops
            
            states.append(QuantumState(
                state_vector=psi,
                time=t,
                populations=pops
            ))
        
        expectation_values = {
            "sigma_x": np.real(result.expect[0]),
            "sigma_y": np.real(result.expect[1]),
            "sigma_z": np.real(result.expect[2]),
        }
        
        metadata = {
            "system_type": "two_level",
            "rabi_frequency": self.rabi_frequency,
            "detuning": self.detuning,
            "decay_rate": self.decay_rate,
            "evolution_type": "open"
        }
        
        return SimulationResult(
            times=times,
            states=states,
            populations=populations,
            expectation_values=expectation_values,
            metadata=metadata
        )
    
    def _evolve_with_simple_decay(self, psi0: np.ndarray, times: np.ndarray) -> SimulationResult:
        """Simple exponential decay model without QuTiP."""
        states = []
        populations = np.zeros((len(times), 2))
        
        for i, t in enumerate(times):
            # Simplified evolution with exponential decay
            U = self._time_evolution_operator(t)
            psi_t = U @ psi0
            
            # Apply exponential decay to excited state
            decay_factor = np.exp(-self.decay_rate * t / 2)
            psi_t[1] *= decay_factor
            
            # Renormalize
            norm = np.linalg.norm(psi_t)
            if norm > 0:
                psi_t /= norm
            
            pops = np.abs(psi_t)**2
            populations[i] = pops
            
            states.append(QuantumState(
                state_vector=psi_t,
                time=t,
                populations=pops
            ))
        
        expectation_values = {
            "sigma_x": np.array([np.real(np.conj(s.state_vector) @ self.sigma_x @ s.state_vector) 
                               for s in states]),
            "sigma_y": np.array([np.real(np.conj(s.state_vector) @ self.sigma_y @ s.state_vector) 
                               for s in states]),
            "sigma_z": np.array([np.real(np.conj(s.state_vector) @ self.sigma_z @ s.state_vector) 
                               for s in states]),
        }
        
        metadata = {
            "system_type": "two_level",
            "rabi_frequency": self.rabi_frequency,
            "detuning": self.detuning,
            "decay_rate": self.decay_rate,
            "evolution_type": "simple_decay"
        }
        
        return SimulationResult(
            times=times,
            states=states,
            populations=populations,
            expectation_values=expectation_values,
            metadata=metadata
        )
    
    def _time_evolution_operator(self, t: float) -> np.ndarray:
        """Calculate time evolution operator U(t) = exp(-iHt/ℏ)."""
        return np.linalg.matrix_power(
            np.eye(2, dtype=complex) - 1j * self.hamiltonian * t / 1000, 1000
        ) if t * np.max(np.abs(self.hamiltonian)) > 0.1 else np.eye(2, dtype=complex) - 1j * self.hamiltonian * t


class MultiLevelSystem:
    """Multi-level atomic system simulator."""
    
    def __init__(self, energy_levels: List[float], transition_dipoles: List[List[float]], 
                 laser_frequencies: List[float], laser_intensities: List[float]):
        """
        Initialize multi-level system.
        
        Args:
            energy_levels: Energy levels (rad/s)
            transition_dipoles: Transition dipole moment matrix
            laser_frequencies: Laser frequencies (rad/s)
            laser_intensities: Laser intensities (W/m²)
        """
        self.energy_levels = np.array(energy_levels)
        self.transition_dipoles = np.array(transition_dipoles)
        self.laser_frequencies = np.array(laser_frequencies)
        self.laser_intensities = np.array(laser_intensities)
        self.num_levels = len(energy_levels)
        
        # Build Hamiltonian
        self.hamiltonian = self._build_hamiltonian()
    
    def _build_hamiltonian(self) -> np.ndarray:
        """Build the multi-level Hamiltonian."""
        H = np.diag(self.energy_levels)
        
        # Add laser interactions
        for freq, intensity in zip(self.laser_frequencies, self.laser_intensities):
            rabi_matrix = self._calculate_rabi_matrix(intensity)
            H += rabi_matrix * np.cos(freq * 0)  # Rotating wave approximation
        
        return H
    
    def _calculate_rabi_matrix(self, intensity: float) -> np.ndarray:
        """Calculate Rabi coupling matrix from laser intensity."""
        # Convert intensity to electric field
        c = 3e8  # speed of light
        epsilon_0 = 8.854e-12
        E_field = np.sqrt(2 * intensity / (c * epsilon_0))
        
        # Rabi frequency matrix
        return E_field * self.transition_dipoles
    
    def evolve(self, initial_populations: List[float], evolution_time: float, 
               time_points: int = 1000) -> SimulationResult:
        """
        Evolve the multi-level system.
        
        Args:
            initial_populations: Initial level populations
            evolution_time: Total evolution time (s)
            time_points: Number of time points
            
        Returns:
            SimulationResult containing evolution data
        """
        # Initialize state vector
        psi0 = np.zeros(self.num_levels, dtype=complex)
        for i, pop in enumerate(initial_populations):
            psi0[i] = np.sqrt(pop)
        
        times = np.linspace(0, evolution_time, time_points)
        states = []
        populations = np.zeros((len(times), self.num_levels))
        
        for i, t in enumerate(times):
            # Time evolution
            U = np.linalg.matrix_power(
                np.eye(self.num_levels, dtype=complex) - 1j * self.hamiltonian * t / 1000, 
                1000
            )
            psi_t = U @ psi0
            
            pops = np.abs(psi_t)**2
            populations[i] = pops
            
            states.append(QuantumState(
                state_vector=psi_t,
                time=t,
                populations=pops,
                energy=np.real(np.conj(psi_t) @ self.hamiltonian @ psi_t)
            ))
        
        # Calculate expectation values
        expectation_values = {
            "energy": np.array([s.energy for s in states]),
            "total_population": np.sum(populations, axis=1),
        }
        
        metadata = {
            "system_type": "multi_level",
            "num_levels": self.num_levels,
            "energy_levels": self.energy_levels.tolist(),
            "laser_frequencies": self.laser_frequencies.tolist(),
            "laser_intensities": self.laser_intensities.tolist(),
        }
        
        return SimulationResult(
            times=times,
            states=states,
            populations=populations,
            expectation_values=expectation_values,
            metadata=metadata
        )


# Tool handler functions
async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle quantum systems tool calls."""
    try:
        if name == "simulate_two_level_atom":
            return await simulate_two_level_atom(**arguments)
        elif name == "rabi_oscillations":
            return await rabi_oscillations(**arguments)
        elif name == "multi_level_atom":
            return await multi_level_atom(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in quantum systems tool {name}: {str(e)}")
        raise


async def simulate_two_level_atom(rabi_frequency: float, detuning: float, 
                                evolution_time: float, initial_state: str = "ground",
                                decay_rate: float = 0.0) -> Dict[str, Any]:
    """Simulate two-level atom dynamics."""
    logger.info(f"Simulating two-level atom: Ω={rabi_frequency}, Δ={detuning}, T={evolution_time}")
    
    system = TwoLevelSystem(rabi_frequency, detuning, decay_rate)
    result = system.evolve(initial_state, evolution_time)
    
    return {
        "success": True,
        "result_type": "two_level_simulation",
        "times": result.times.tolist(),
        "ground_population": result.populations[:, 0].tolist(),
        "excited_population": result.populations[:, 1].tolist(),
        "sigma_x": result.expectation_values["sigma_x"].tolist(),
        "sigma_y": result.expectation_values["sigma_y"].tolist(),
        "sigma_z": result.expectation_values["sigma_z"].tolist(),
        "metadata": result.metadata,
        "summary": {
            "max_excited_population": float(np.max(result.populations[:, 1])),
            "final_excited_population": float(result.populations[-1, 1]),
            "rabi_period": 2 * np.pi / rabi_frequency if rabi_frequency > 0 else None,
        }
    }


async def rabi_oscillations(rabi_frequency: float, max_time: float, 
                          time_points: int = 1000, include_decay: bool = False,
                          decay_rate: float = 0.0) -> Dict[str, Any]:
    """Calculate Rabi oscillations."""
    logger.info(f"Calculating Rabi oscillations: Ω={rabi_frequency}, T_max={max_time}")
    
    decay = decay_rate if include_decay else 0.0
    system = TwoLevelSystem(rabi_frequency, 0.0, decay)  # On resonance
    result = system.evolve("ground", max_time, time_points)
    
    # Calculate Rabi frequency from data
    excited_pop = result.populations[:, 1]
    fft_freq = np.fft.fftfreq(len(excited_pop), max_time / len(excited_pop))
    fft_vals = np.abs(np.fft.fft(excited_pop))
    dominant_freq = fft_freq[np.argmax(fft_vals[1:])] * 2 * np.pi  # Convert to rad/s
    
    return {
        "success": True,
        "result_type": "rabi_oscillations",
        "times": result.times.tolist(),
        "excited_population": excited_pop.tolist(),
        "ground_population": result.populations[:, 0].tolist(),
        "theoretical_rabi_freq": rabi_frequency,
        "measured_rabi_freq": float(abs(dominant_freq)),
        "rabi_period": 2 * np.pi / rabi_frequency,
        "include_decay": include_decay,
        "decay_rate": decay_rate,
        "metadata": result.metadata,
    }


async def multi_level_atom(energy_levels: List[float], transition_dipoles: List[List[float]],
                         laser_frequencies: List[float], laser_intensities: List[float],
                         evolution_time: float, initial_populations: List[float]) -> Dict[str, Any]:
    """Simulate multi-level atomic system."""
    logger.info(f"Simulating multi-level atom with {len(energy_levels)} levels")
    
    system = MultiLevelSystem(energy_levels, transition_dipoles, laser_frequencies, laser_intensities)
    result = system.evolve(initial_populations, evolution_time)
    
    return {
        "success": True,
        "result_type": "multi_level_simulation",
        "times": result.times.tolist(),
        "populations": result.populations.tolist(),
        "energy_expectation": result.expectation_values["energy"].tolist(),
        "level_labels": [f"Level {i}" for i in range(len(energy_levels))],
        "final_populations": result.populations[-1].tolist(),
        "population_transfer_efficiency": float(np.sum(result.populations[-1][1:])),  # Population in excited states
        "metadata": result.metadata,
    }