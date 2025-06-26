"""
Cold Atoms Tools for AMO Physics
Advanced simulation tools for cold atoms and Bose-Einstein condensates.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.fft import fft2, ifft2, fftfreq
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class BECState:
    """Represents a BEC wavefunction state."""
    wavefunction: np.ndarray
    time: float
    energy: float
    particle_number: float
    chemical_potential: float
    grid_x: np.ndarray
    grid_y: np.ndarray


@dataclass
class BECSimulationResult:
    """Container for BEC simulation results."""
    times: np.ndarray
    states: List[BECState]
    energies: np.ndarray
    particle_numbers: np.ndarray
    chemical_potentials: np.ndarray
    metadata: Dict[str, Any]


class GrossPitaevskiiSolver:
    """Solver for the Gross-Pitaevskii equation."""
    
    def __init__(self, grid_size: int, box_length: float, particle_number: int,
                 scattering_length: float, trap_frequency: float):
        """
        Initialize GP equation solver.
        
        Args:
            grid_size: Number of grid points per dimension
            box_length: Box length (μm)
            particle_number: Number of particles
            scattering_length: s-wave scattering length (nm)
            trap_frequency: Harmonic trap frequency (Hz)
        """
        self.grid_size = grid_size
        self.box_length = box_length * 1e-6  # Convert to meters
        self.particle_number = particle_number
        self.scattering_length = scattering_length * 1e-9  # Convert to meters
        self.trap_frequency = trap_frequency
        
        # Physical constants
        self.hbar = 1.054571817e-34  # J⋅s
        self.m_rb87 = 1.443160648e-25  # kg (Rb-87 mass)
        self.a0 = 5.291772109e-11  # Bohr radius
        
        # Grid setup
        self.dx = self.box_length / grid_size
        self.dy = self.dx
        
        # Coordinate grids
        x = np.linspace(-self.box_length/2, self.box_length/2, grid_size)
        y = np.linspace(-self.box_length/2, self.box_length/2, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # k-space grids for kinetic energy
        kx = 2 * np.pi * fftfreq(grid_size, self.dx)
        ky = 2 * np.pi * fftfreq(grid_size, self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        
        # Harmonic trap potential
        omega = 2 * np.pi * trap_frequency
        self.V_trap = 0.5 * self.m_rb87 * omega**2 * (self.X**2 + self.Y**2)
        
        # Interaction strength
        self.g = 4 * np.pi * self.hbar**2 * self.scattering_length / self.m_rb87
        
        # Initial wavefunction (Gaussian)
        sigma = np.sqrt(self.hbar / (self.m_rb87 * omega))
        self.psi_initial = np.exp(-(self.X**2 + self.Y**2) / (2 * sigma**2))
        self.psi_initial = self.psi_initial / np.sqrt(np.sum(np.abs(self.psi_initial)**2) * self.dx * self.dy)
        self.psi_initial = np.sqrt(particle_number) * self.psi_initial
    
    def imaginary_time_evolution(self, max_iterations: int = 1000, 
                                tolerance: float = 1e-8) -> BECState:
        """
        Find ground state using imaginary time evolution.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Ground state BEC state
        """
        psi = self.psi_initial.copy()
        dt_imag = 1e-6  # Imaginary time step
        
        for i in range(max_iterations):
            psi_old = psi.copy()
            
            # Split-step method for imaginary time evolution
            # Kinetic energy step (momentum space)
            psi_k = fft2(psi)
            psi_k *= np.exp(-dt_imag * self.hbar * self.K2 / (2 * self.m_rb87))
            psi = ifft2(psi_k)
            
            # Potential energy step (position space)
            V_total = self.V_trap + self.g * np.abs(psi)**2
            psi *= np.exp(-dt_imag * V_total / self.hbar)
            
            # Normalize
            norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx * self.dy)
            psi = np.sqrt(self.particle_number) * psi / norm
            
            # Check convergence
            change = np.sum(np.abs(psi - psi_old)**2) * self.dx * self.dy
            if change < tolerance:
                logger.info(f"Ground state converged after {i+1} iterations")
                break
        
        # Calculate properties
        energy = self._calculate_energy(psi)
        mu = self._calculate_chemical_potential(psi)
        N = np.sum(np.abs(psi)**2) * self.dx * self.dy
        
        return BECState(
            wavefunction=psi,
            time=0.0,
            energy=energy,
            particle_number=N,
            chemical_potential=mu,
            grid_x=self.X,
            grid_y=self.Y
        )
    
    def real_time_evolution(self, initial_state: BECState, evolution_time: float,
                          time_points: int = 100) -> BECSimulationResult:
        """
        Evolve BEC in real time.
        
        Args:
            initial_state: Initial BEC state
            evolution_time: Evolution time (ms)
            time_points: Number of time points
            
        Returns:
            BEC simulation result
        """
        evolution_time_s = evolution_time * 1e-3  # Convert to seconds
        times = np.linspace(0, evolution_time_s, time_points)
        dt = times[1] - times[0]
        
        states = [initial_state]
        energies = [initial_state.energy]
        particle_numbers = [initial_state.particle_number]
        chemical_potentials = [initial_state.chemical_potential]
        
        psi = initial_state.wavefunction.copy()
        
        for i, t in enumerate(times[1:]):
            # Split-step method for real time evolution
            # Half kinetic energy step
            psi_k = fft2(psi)
            psi_k *= np.exp(-1j * dt * self.hbar * self.K2 / (4 * self.m_rb87))
            psi = ifft2(psi_k)
            
            # Full potential energy step
            V_total = self.V_trap + self.g * np.abs(psi)**2
            psi *= np.exp(-1j * dt * V_total / self.hbar)
            
            # Half kinetic energy step
            psi_k = fft2(psi)
            psi_k *= np.exp(-1j * dt * self.hbar * self.K2 / (4 * self.m_rb87))
            psi = ifft2(psi_k)
            
            # Calculate properties
            energy = self._calculate_energy(psi)
            mu = self._calculate_chemical_potential(psi)
            N = np.sum(np.abs(psi)**2) * self.dx * self.dy
            
            state = BECState(
                wavefunction=psi.copy(),
                time=t,
                energy=energy,
                particle_number=N,
                chemical_potential=mu,
                grid_x=self.X,
                grid_y=self.Y
            )
            
            states.append(state)
            energies.append(energy)
            particle_numbers.append(N)
            chemical_potentials.append(mu)
        
        metadata = {
            "grid_size": self.grid_size,
            "box_length_um": self.box_length * 1e6,
            "particle_number": self.particle_number,
            "scattering_length_nm": self.scattering_length * 1e9,
            "trap_frequency_hz": self.trap_frequency,
            "evolution_time_ms": evolution_time,
        }
        
        return BECSimulationResult(
            times=times,
            states=states,
            energies=np.array(energies),
            particle_numbers=np.array(particle_numbers),
            chemical_potentials=np.array(chemical_potentials),
            metadata=metadata
        )
    
    def _calculate_energy(self, psi: np.ndarray) -> float:
        """Calculate total energy of the wavefunction."""
        # Kinetic energy
        psi_k = fft2(psi)
        kinetic = np.sum(self.hbar**2 * self.K2 * np.abs(psi_k)**2 / (2 * self.m_rb87))
        kinetic *= self.dx * self.dy / (2 * np.pi)**2
        
        # Potential energy
        potential = np.sum(self.V_trap * np.abs(psi)**2) * self.dx * self.dy
        
        # Interaction energy
        interaction = 0.5 * self.g * np.sum(np.abs(psi)**4) * self.dx * self.dy
        
        return float(kinetic + potential + interaction)
    
    def _calculate_chemical_potential(self, psi: np.ndarray) -> float:
        """Calculate chemical potential."""
        # Apply GP operator
        psi_k = fft2(psi)
        kinetic_term = ifft2(-self.hbar**2 * self.K2 * psi_k / (2 * self.m_rb87))
        potential_term = (self.V_trap + self.g * np.abs(psi)**2) * psi
        
        H_psi = kinetic_term + potential_term
        
        # Chemical potential = <ψ|H|ψ> / <ψ|ψ>
        numerator = np.sum(np.conj(psi) * H_psi) * self.dx * self.dy
        denominator = np.sum(np.abs(psi)**2) * self.dx * self.dy
        
        return float(np.real(numerator / denominator))


class OpticalLattice:
    """Optical lattice potential and dynamics."""
    
    def __init__(self, lattice_depth: float, lattice_spacing: float):
        """
        Initialize optical lattice.
        
        Args:
            lattice_depth: Lattice depth in units of recoil energy
            lattice_spacing: Lattice spacing (μm)
        """
        self.lattice_depth = lattice_depth
        self.lattice_spacing = lattice_spacing * 1e-6  # Convert to meters
        
        # Lattice wave vector
        self.k_lattice = 2 * np.pi / self.lattice_spacing
    
    def potential(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate optical lattice potential.
        
        Args:
            x: x coordinates
            y: y coordinates
            
        Returns:
            Lattice potential
        """
        V_x = -self.lattice_depth * np.cos(self.k_lattice * x)**2
        V_y = -self.lattice_depth * np.cos(self.k_lattice * y)**2
        
        return V_x + V_y
    
    def band_structure(self, num_bands: int = 5) -> Dict[str, np.ndarray]:
        """
        Calculate band structure of the optical lattice.
        
        Args:
            num_bands: Number of bands to calculate
            
        Returns:
            Dictionary with band structure data
        """
        # Quasi-momentum grid
        num_k = 100
        k_values = np.linspace(-np.pi/self.lattice_spacing, np.pi/self.lattice_spacing, num_k)
        
        # Band energies (simplified 1D calculation)
        bands = np.zeros((num_bands, num_k))
        
        for i, k in enumerate(k_values):
            # Construct Hamiltonian matrix (truncated)
            H_size = 2 * num_bands + 1
            H = np.zeros((H_size, H_size))
            
            # Diagonal elements (kinetic energy)
            for n in range(-num_bands, num_bands + 1):
                idx = n + num_bands
                H[idx, idx] = (k + n * self.k_lattice)**2 / 2
            
            # Off-diagonal elements (lattice coupling)
            for n in range(-num_bands, num_bands):
                idx = n + num_bands
                H[idx, idx + 1] = -self.lattice_depth / 4
                H[idx + 1, idx] = -self.lattice_depth / 4
            
            # Diagonalize
            eigenvalues = np.linalg.eigvals(H)
            eigenvalues.sort()
            
            bands[:, i] = eigenvalues[:num_bands]
        
        return {
            "k_values": k_values,
            "bands": bands,
            "lattice_depth": self.lattice_depth,
            "lattice_spacing_um": self.lattice_spacing * 1e6,
        }


# Tool handler functions
async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle cold atoms tool calls."""
    try:
        if name == "bec_simulation":
            return await bec_simulation(**arguments)
        elif name == "optical_lattice_bands":
            return await optical_lattice_bands(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in cold atoms tool {name}: {str(e)}")
        raise


async def bec_simulation(grid_size: int, box_length: float, particle_number: int,
                       scattering_length: float, trap_frequency: float,
                       evolution_time: float) -> Dict[str, Any]:
    """Simulate Bose-Einstein condensate using Gross-Pitaevskii equation."""
    logger.info(f"BEC simulation: N={particle_number}, a={scattering_length}nm, ω={trap_frequency}Hz")
    
    # Initialize solver
    solver = GrossPitaevskiiSolver(
        grid_size=grid_size,
        box_length=box_length,
        particle_number=particle_number,
        scattering_length=scattering_length,
        trap_frequency=trap_frequency
    )
    
    # Find ground state
    ground_state = solver.imaginary_time_evolution()
    
    # Evolve in real time
    result = solver.real_time_evolution(ground_state, evolution_time)
    
    # Extract key properties
    final_state = result.states[-1]
    density_profile = np.abs(final_state.wavefunction)**2
    
    # Calculate characteristic lengths
    healing_length = 1 / np.sqrt(8 * np.pi * solver.scattering_length * np.max(density_profile))
    thomas_fermi_radius = np.sqrt(2 * ground_state.chemical_potential / 
                                 (solver.m_rb87 * (2 * np.pi * trap_frequency)**2))
    
    return {
        "success": True,
        "result_type": "bec_simulation",
        "ground_state": {
            "energy": ground_state.energy,
            "chemical_potential": ground_state.chemical_potential,
            "particle_number": ground_state.particle_number,
            "density_profile": density_profile.tolist(),
        },
        "time_evolution": {
            "times_ms": (result.times * 1000).tolist(),
            "energies": result.energies.tolist(),
            "particle_numbers": result.particle_numbers.tolist(),
            "chemical_potentials": result.chemical_potentials.tolist(),
        },
        "characteristic_scales": {
            "healing_length_um": healing_length * 1e6,
            "thomas_fermi_radius_um": thomas_fermi_radius * 1e6,
            "scattering_length_nm": scattering_length,
            "oscillator_length_um": np.sqrt(solver.hbar / (solver.m_rb87 * 2 * np.pi * trap_frequency)) * 1e6,
        },
        "grid_info": {
            "grid_size": grid_size,
            "box_length_um": box_length,
            "grid_spacing_um": box_length / grid_size,
        },
        "metadata": result.metadata,
        "analysis": {
            "interaction_parameter": 4 * np.pi * scattering_length * 1e-9 * particle_number / box_length,
            "thomas_fermi_regime": scattering_length > 0 and particle_number > 100,
            "strongly_interacting": abs(scattering_length) > 100,  # nm
        }
    }


async def optical_lattice_bands(lattice_depth: float, lattice_spacing: float,
                              num_bands: int = 5) -> Dict[str, Any]:
    """Calculate optical lattice band structure."""
    logger.info(f"Optical lattice bands: V₀={lattice_depth}Er, a={lattice_spacing}μm")
    
    lattice = OpticalLattice(lattice_depth, lattice_spacing)
    band_data = lattice.band_structure(num_bands)
    
    # Calculate band gaps
    band_gaps = []
    for i in range(num_bands - 1):
        gap = np.min(band_data["bands"][i+1]) - np.max(band_data["bands"][i])
        band_gaps.append(float(gap))
    
    return {
        "success": True,
        "result_type": "optical_lattice_bands",
        "k_values": band_data["k_values"].tolist(),
        "bands": band_data["bands"].tolist(),
        "band_gaps": band_gaps,
        "lattice_parameters": {
            "depth": lattice_depth,
            "spacing_um": lattice_spacing,
            "recoil_energy": 1.0,  # Reference energy
        },
        "analysis": {
            "num_bands": num_bands,
            "bandwidth_lowest": float(np.max(band_data["bands"][0]) - np.min(band_data["bands"][0])),
            "first_gap": band_gaps[0] if band_gaps else 0,
            "deep_lattice": lattice_depth > 10,
        }
    }