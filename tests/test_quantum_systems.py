"""
Tests for quantum systems tools.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools.quantum_systems import (
    TwoLevelSystem, 
    MultiLevelSystem,
    simulate_two_level_atom,
    rabi_oscillations,
    multi_level_atom
)


class TestTwoLevelSystem:
    """Test two-level atomic system simulations."""
    
    def test_initialization(self):
        """Test proper initialization of two-level system."""
        system = TwoLevelSystem(
            rabi_frequency=1e6,  # 1 MHz
            detuning=0,          # On resonance
            decay_rate=1e3       # 1 kHz decay
        )
        
        assert system.rabi_frequency == 1e6
        assert system.detuning == 0
        assert system.decay_rate == 1e3
        assert system.hamiltonian.shape == (2, 2)
    
    def test_ground_state_evolution(self):
        """Test evolution from ground state."""
        system = TwoLevelSystem(
            rabi_frequency=2*np.pi*1e6,  # 2π MHz
            detuning=0,
            decay_rate=0
        )
        
        result = system.evolve("ground", 1e-6, 100)  # 1 μs evolution
        
        # Check result structure
        assert len(result.times) == 100
        assert len(result.states) == 100
        assert result.populations.shape == (100, 2)
        
        # Initial population should be in ground state
        assert result.populations[0, 0] == pytest.approx(1.0, abs=1e-10)
        assert result.populations[0, 1] == pytest.approx(0.0, abs=1e-10)
        
        # Check conservation of probability
        total_populations = np.sum(result.populations, axis=1)
        assert np.all(np.abs(total_populations - 1.0) < 1e-10)
    
    def test_rabi_frequency_oscillations(self):
        """Test that oscillations occur at correct Rabi frequency."""
        omega_R = 2*np.pi*1e6  # 1 MHz Rabi frequency
        system = TwoLevelSystem(
            rabi_frequency=omega_R,
            detuning=0,
            decay_rate=0
        )
        
        # Evolve for exactly one Rabi period
        period = 2*np.pi / omega_R
        result = system.evolve("ground", period, 1000)
        
        # Should return to ground state after one period
        final_ground_pop = result.populations[-1, 0]
        assert final_ground_pop == pytest.approx(1.0, abs=1e-2)
    
    def test_detuning_effects(self):
        """Test effects of laser detuning."""
        # Large detuning should suppress oscillations
        system = TwoLevelSystem(
            rabi_frequency=2*np.pi*1e6,  # 1 MHz
            detuning=2*np.pi*10e6,       # 10 MHz detuning
            decay_rate=0
        )
        
        result = system.evolve("ground", 1e-6, 100)
        
        # Maximum excited population should be much less than 1
        max_excited = np.max(result.populations[:, 1])
        assert max_excited < 0.1


class TestMultiLevelSystem:
    """Test multi-level atomic system simulations."""
    
    def test_three_level_system(self):
        """Test three-level system initialization and evolution."""
        energy_levels = [0, 1e15, 2e15]  # 3 levels
        transition_dipoles = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]  # Simple coupling
        laser_frequencies = [1e15]
        laser_intensities = [1e8]  # W/m²
        
        system = MultiLevelSystem(
            energy_levels, transition_dipoles, 
            laser_frequencies, laser_intensities
        )
        
        assert system.num_levels == 3
        assert system.hamiltonian.shape == (3, 3)
        
        # Test evolution
        initial_populations = [1.0, 0.0, 0.0]  # Start in ground state
        result = system.evolve(initial_populations, 1e-12, 100)  # 1 ps
        
        # Check result structure
        assert len(result.states) == 100
        assert result.populations.shape == (100, 3)
        
        # Probability conservation
        total_populations = np.sum(result.populations, axis=1)
        assert np.all(np.abs(total_populations - 1.0) < 1e-10)


class TestQuantumSystemsAPI:
    """Test the async API functions."""
    
    @pytest.mark.asyncio
    async def test_simulate_two_level_atom(self):
        """Test the async two-level atom simulation API."""
        result = await simulate_two_level_atom(
            rabi_frequency=2*np.pi*1e6,
            detuning=0,
            evolution_time=1e-6,
            initial_state="ground",
            decay_rate=0
        )
        
        assert result["success"] is True
        assert result["result_type"] == "two_level_simulation"
        assert "times" in result
        assert "ground_population" in result
        assert "excited_population" in result
        assert "sigma_x" in result
        assert "sigma_y" in result
        assert "sigma_z" in result
        assert "metadata" in result
        assert "summary" in result
        
        # Check data lengths match
        times = result["times"]
        ground_pop = result["ground_population"]
        excited_pop = result["excited_population"]
        
        assert len(times) == len(ground_pop) == len(excited_pop)
        
        # Check initial conditions
        assert ground_pop[0] == pytest.approx(1.0, abs=1e-10)
        assert excited_pop[0] == pytest.approx(0.0, abs=1e-10)
    
    @pytest.mark.asyncio
    async def test_rabi_oscillations(self):
        """Test the Rabi oscillations API."""
        result = await rabi_oscillations(
            rabi_frequency=2*np.pi*1e6,
            max_time=2e-6,  # 2 μs
            time_points=1000,
            include_decay=False
        )
        
        assert result["success"] is True
        assert result["result_type"] == "rabi_oscillations"
        assert "times" in result
        assert "excited_population" in result
        assert "theoretical_rabi_freq" in result
        assert "measured_rabi_freq" in result
        assert "rabi_period" in result
        
        # Check that measured frequency is close to theoretical
        theoretical = result["theoretical_rabi_freq"]
        measured = result["measured_rabi_freq"]
        assert measured == pytest.approx(theoretical, rel=0.1)  # 10% tolerance
    
    @pytest.mark.asyncio
    async def test_multi_level_atom(self):
        """Test the multi-level atom API."""
        result = await multi_level_atom(
            energy_levels=[0, 1e15, 2e15],
            transition_dipoles=[[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            laser_frequencies=[1e15],
            laser_intensities=[1e8],
            evolution_time=1e-12,
            initial_populations=[1.0, 0.0, 0.0]
        )
        
        assert result["success"] is True
        assert result["result_type"] == "multi_level_simulation"
        assert "times" in result
        assert "populations" in result
        assert "level_labels" in result
        assert "final_populations" in result
        assert "population_transfer_efficiency" in result
        
        # Check data structure
        populations = result["populations"]
        assert len(populations) > 0
        assert len(populations[0]) == 3  # 3 levels


class TestPhysicalCorrectness:
    """Test physical correctness of simulations."""
    
    def test_energy_conservation_closed_system(self):
        """Test energy conservation in closed quantum system."""
        system = TwoLevelSystem(
            rabi_frequency=2*np.pi*1e6,
            detuning=2*np.pi*5e5,  # 0.5 MHz detuning
            decay_rate=0  # Closed system
        )
        
        result = system.evolve("ground", 1e-6, 1000)
        
        # Calculate total energy at each time
        energies = []
        for state in result.states:
            energy = np.real(np.conj(state.state_vector) @ system.hamiltonian @ state.state_vector)
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Energy should be conserved (constant within numerical precision)
        energy_variation = np.std(energies) / np.mean(np.abs(energies))
        assert energy_variation < 1e-10
    
    def test_unitarity_closed_system(self):
        """Test unitarity of evolution in closed system."""
        system = TwoLevelSystem(
            rabi_frequency=2*np.pi*1e6,
            detuning=0,
            decay_rate=0
        )
        
        result = system.evolve("ground", 1e-6, 100)
        
        # Check that state norms are preserved
        norms = [np.linalg.norm(state.state_vector) for state in result.states]
        
        for norm in norms:
            assert norm == pytest.approx(1.0, abs=1e-10)
    
    def test_population_decrease_with_decay(self):
        """Test that total population decreases with spontaneous emission."""
        system = TwoLevelSystem(
            rabi_frequency=2*np.pi*1e6,
            detuning=0,
            decay_rate=1e6  # Strong decay
        )
        
        # Start in excited state
        result = system.evolve("excited", 1e-5, 1000)
        
        # Total population should decrease over time
        total_populations = np.sum(result.populations, axis=1)
        
        # Should start near 1 and decrease
        assert total_populations[0] == pytest.approx(1.0, abs=1e-3)
        assert total_populations[-1] < total_populations[0]
    
    def test_bloch_vector_magnitude(self):
        """Test that Bloch vector magnitude is ≤ 1."""
        system = TwoLevelSystem(
            rabi_frequency=2*np.pi*1e6,
            detuning=2*np.pi*2e5,
            decay_rate=1e5  # Some decay
        )
        
        result = system.evolve("superposition", 1e-6, 100)
        
        # Calculate Bloch vector for each state
        sigma_x = system.sigma_x
        sigma_y = system.sigma_y
        sigma_z = system.sigma_z
        
        for state in result.states:
            psi = state.state_vector
            x = np.real(np.conj(psi) @ sigma_x @ psi)
            y = np.real(np.conj(psi) @ sigma_y @ psi)
            z = np.real(np.conj(psi) @ sigma_z @ psi)
            
            bloch_magnitude = np.sqrt(x**2 + y**2 + z**2)
            assert bloch_magnitude <= 1.0 + 1e-10  # Allow for numerical precision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])