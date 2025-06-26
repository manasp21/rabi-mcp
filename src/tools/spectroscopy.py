"""
Spectroscopy Tools for AMO Physics
Advanced tools for spectroscopic analysis and laser-atom interactions.
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import constants as const
import scipy.integrate as integrate

from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SpectralLine:
    """Represents a spectral line with its properties."""
    frequency: float  # Center frequency (rad/s)
    intensity: float  # Line intensity
    width: float     # Linewidth (FWHM)
    shape: str       # Line shape ('lorentzian', 'gaussian', 'voigt')


@dataclass
class SpectroscopyResult:
    """Container for spectroscopy calculation results."""
    frequencies: np.ndarray
    spectrum: np.ndarray
    lines: List[SpectralLine]
    metadata: Dict[str, Any]


class SpectralLineShapes:
    """Collection of spectral line shape functions."""
    
    @staticmethod
    def lorentzian(freq: np.ndarray, center: float, width: float, intensity: float = 1.0) -> np.ndarray:
        """
        Lorentzian line shape (natural broadening).
        
        Args:
            freq: Frequency array (rad/s)
            center: Center frequency (rad/s)  
            width: FWHM width (rad/s)
            intensity: Peak intensity
            
        Returns:
            Lorentzian line shape
        """
        gamma = width / 2  # Half-width at half maximum
        return intensity * gamma**2 / ((freq - center)**2 + gamma**2)
    
    @staticmethod
    def gaussian(freq: np.ndarray, center: float, width: float, intensity: float = 1.0) -> np.ndarray:
        """
        Gaussian line shape (Doppler broadening).
        
        Args:
            freq: Frequency array (rad/s)
            center: Center frequency (rad/s)
            width: FWHM width (rad/s)
            intensity: Peak intensity
            
        Returns:
            Gaussian line shape
        """
        sigma = width / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
        return intensity * np.exp(-(freq - center)**2 / (2 * sigma**2))
    
    @staticmethod
    def voigt(freq: np.ndarray, center: float, gamma_l: float, gamma_g: float, 
              intensity: float = 1.0) -> np.ndarray:
        """
        Voigt profile (convolution of Lorentzian and Gaussian).
        
        Args:
            freq: Frequency array (rad/s)
            center: Center frequency (rad/s)
            gamma_l: Lorentzian HWHM (rad/s)
            gamma_g: Gaussian HWHM (rad/s)
            intensity: Peak intensity
            
        Returns:
            Voigt line shape
        """
        # Simplified Voigt approximation
        gamma_v = 0.5346 * gamma_l + np.sqrt(0.2166 * gamma_l**2 + gamma_g**2)
        return SpectralLineShapes.lorentzian(freq, center, 2 * gamma_v, intensity)


class AbsorptionSpectroscopy:
    """Tools for absorption spectroscopy calculations."""
    
    def __init__(self):
        """Initialize absorption spectroscopy calculator."""
        self.line_shapes = SpectralLineShapes()
    
    def calculate_spectrum(self, transition_frequency: float, linewidth: float,
                          frequency_range: Tuple[float, float],
                          broadening_type: str = "natural",
                          temperature: float = 300.0,
                          atomic_mass: float = 23.0,
                          num_points: int = 1000) -> SpectroscopyResult:
        """
        Calculate absorption spectrum.
        
        Args:
            transition_frequency: Transition frequency (rad/s)
            linewidth: Natural linewidth (rad/s)
            frequency_range: (min_freq, max_freq) in rad/s
            broadening_type: Type of broadening ('natural', 'doppler', 'collisional')
            temperature: Temperature for Doppler broadening (K)
            atomic_mass: Atomic mass for Doppler broadening (amu)
            num_points: Number of frequency points
            
        Returns:
            SpectroscopyResult with spectrum data
        """
        frequencies = np.linspace(frequency_range[0], frequency_range[1], num_points)
        
        if broadening_type == "natural":
            spectrum = self._natural_broadening(frequencies, transition_frequency, linewidth)
            effective_width = linewidth
            
        elif broadening_type == "doppler":
            doppler_width = self._calculate_doppler_width(transition_frequency, temperature, atomic_mass)
            spectrum = self._doppler_broadening(frequencies, transition_frequency, doppler_width)
            effective_width = doppler_width
            
        elif broadening_type == "collisional":
            # Assume pressure broadening similar to natural
            collisional_width = linewidth * 2  # Simplified assumption
            spectrum = self._natural_broadening(frequencies, transition_frequency, collisional_width)
            effective_width = collisional_width
            
        else:
            raise ValueError(f"Unknown broadening type: {broadening_type}")
        
        lines = [SpectralLine(
            frequency=transition_frequency,
            intensity=np.max(spectrum),
            width=effective_width,
            shape=broadening_type
        )]
        
        metadata = {
            "transition_frequency": transition_frequency,
            "natural_linewidth": linewidth,
            "broadening_type": broadening_type,
            "temperature": temperature,
            "atomic_mass": atomic_mass,
            "frequency_range": frequency_range,
        }
        
        return SpectroscopyResult(
            frequencies=frequencies,
            spectrum=spectrum,
            lines=lines,
            metadata=metadata
        )
    
    def _natural_broadening(self, freq: np.ndarray, center: float, width: float) -> np.ndarray:
        """Calculate natural broadening (Lorentzian)."""
        return self.line_shapes.lorentzian(freq, center, width)
    
    def _doppler_broadening(self, freq: np.ndarray, center: float, width: float) -> np.ndarray:
        """Calculate Doppler broadening (Gaussian)."""
        return self.line_shapes.gaussian(freq, center, width)
    
    def _calculate_doppler_width(self, freq: float, temp: float, mass: float) -> float:
        """
        Calculate Doppler broadening width.
        
        Args:
            freq: Transition frequency (rad/s)
            temp: Temperature (K)
            mass: Atomic mass (amu)
            
        Returns:
            Doppler width (FWHM) in rad/s
        """
        # Convert frequency to Hz
        freq_hz = freq / (2 * np.pi)
        
        # Convert mass to kg
        mass_kg = mass * const.atomic_mass
        
        # Doppler width (FWHM)
        doppler_width_hz = 2 * freq_hz * np.sqrt(2 * np.log(2) * const.k * temp / (mass_kg * const.c**2))
        
        # Convert back to rad/s
        return doppler_width_hz * 2 * np.pi


class LaserAtomInteraction:
    """Tools for strong-field laser-atom interactions."""
    
    def __init__(self):
        """Initialize laser-atom interaction calculator."""
        pass
    
    def calculate_interaction(self, laser_intensity: float, laser_wavelength: float,
                            pulse_duration: float, ionization_potential: float,
                            interaction_type: str = "tunneling") -> Dict[str, Any]:
        """
        Calculate strong-field laser-atom interactions.
        
        Args:
            laser_intensity: Laser intensity (W/cm²)
            laser_wavelength: Laser wavelength (nm)
            pulse_duration: Pulse duration (fs)
            ionization_potential: Ionization potential (eV)
            interaction_type: Type of interaction
            
        Returns:
            Dictionary with interaction parameters
        """
        # Convert units
        intensity_si = laser_intensity * 1e4  # W/m²
        wavelength_m = laser_wavelength * 1e-9  # m
        duration_s = pulse_duration * 1e-15  # s
        ip_j = ionization_potential * const.eV  # J
        
        # Calculate laser parameters
        frequency = const.c / wavelength_m  # Hz
        angular_freq = 2 * np.pi * frequency  # rad/s
        
        # Electric field amplitude
        E0 = np.sqrt(2 * intensity_si / (const.c * const.epsilon_0))  # V/m
        
        # Ponderomotive energy
        Up = const.e**2 * E0**2 / (4 * const.m_e * angular_freq**2)  # J
        Up_ev = Up / const.eV  # eV
        
        # Keldysh parameter
        gamma_k = np.sqrt(ip_j / (2 * Up))
        
        # Cycle-averaged ionization rate (ADK theory)
        if interaction_type == "tunneling":
            ionization_rate = self._adk_ionization_rate(E0, ip_j, angular_freq)
        else:
            ionization_rate = 0.0  # Placeholder for other methods
        
        # Critical intensity for over-the-barrier ionization
        I_critical = ip_j**4 / (16 * const.e**4) * (const.c * const.epsilon_0)  # W/m²
        I_critical_wcm2 = I_critical / 1e4  # W/cm²
        
        return {
            "laser_parameters": {
                "intensity_wcm2": laser_intensity,
                "wavelength_nm": laser_wavelength,
                "frequency_hz": frequency,
                "pulse_duration_fs": pulse_duration,
                "electric_field_vm": E0,
            },
            "atom_parameters": {
                "ionization_potential_ev": ionization_potential,
            },
            "interaction_parameters": {
                "ponderomotive_energy_ev": Up_ev,
                "keldysh_parameter": gamma_k,
                "interaction_regime": self._classify_regime(gamma_k),
                "ionization_rate_hz": ionization_rate,
                "critical_intensity_wcm2": I_critical_wcm2,
            },
            "analysis": {
                "is_strong_field": laser_intensity > I_critical_wcm2 * 0.01,
                "is_tunneling_regime": gamma_k < 1,
                "is_multiphoton_regime": gamma_k > 1,
                "is_over_barrier": laser_intensity > I_critical_wcm2,
                "photons_for_ionization": int(np.ceil(ionization_potential / (const.h * frequency / const.eV))),
            }
        }
    
    def _adk_ionization_rate(self, E0: float, ip: float, omega: float) -> float:
        """
        Calculate ADK (Ammosov-Delone-Krainov) ionization rate.
        
        Args:
            E0: Electric field amplitude (V/m)
            ip: Ionization potential (J)
            omega: Angular frequency (rad/s)
            
        Returns:
            Ionization rate (Hz)
        """
        # Atomic units conversion
        E_au = E0 / (const.e / (4 * np.pi * const.epsilon_0 * const.a_0**2))  # a.u.
        Ip_au = ip / (const.e**2 / (4 * np.pi * const.epsilon_0 * const.a_0))  # a.u.
        
        # ADK rate (simplified for hydrogen-like atoms)
        n_eff = 1 / np.sqrt(2 * Ip_au)  # Effective principal quantum number
        l = 0  # Assuming s-state
        
        C_nl2 = 2**(2*n_eff) / (n_eff * np.math.gamma(n_eff + l + 1) * np.math.gamma(n_eff - l))
        
        rate_au = C_nl2 * Ip_au * (2 * Ip_au / E_au)**(2*n_eff - 1) * np.exp(-2 * Ip_au / (3 * E_au))
        
        # Convert to Hz
        rate_hz = rate_au * (const.e**2 / (4 * np.pi * const.epsilon_0 * const.hbar * const.a_0))
        
        return rate_hz
    
    def _classify_regime(self, gamma_k: float) -> str:
        """Classify interaction regime based on Keldysh parameter."""
        if gamma_k < 0.5:
            return "tunneling"
        elif gamma_k < 1.5:
            return "intermediate"
        else:
            return "multiphoton"


# Tool handler functions
async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle spectroscopy tool calls."""
    try:
        if name == "absorption_spectrum":
            return await absorption_spectrum(**arguments)
        elif name == "laser_atom_interaction":
            return await laser_atom_interaction(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in spectroscopy tool {name}: {str(e)}")
        raise


async def absorption_spectrum(transition_frequency: float, linewidth: float,
                            frequency_range: List[float], broadening_type: str = "natural",
                            temperature: float = 300.0, atomic_mass: float = 23.0) -> Dict[str, Any]:
    """Calculate absorption spectrum."""
    logger.info(f"Calculating absorption spectrum: ν₀={transition_frequency}, Γ={linewidth}")
    
    spectroscopy = AbsorptionSpectroscopy()
    result = spectroscopy.calculate_spectrum(
        transition_frequency=transition_frequency,
        linewidth=linewidth,
        frequency_range=(frequency_range[0], frequency_range[1]),
        broadening_type=broadening_type,
        temperature=temperature,
        atomic_mass=atomic_mass
    )
    
    # Calculate additional properties
    peak_freq = result.frequencies[np.argmax(result.spectrum)]
    fwhm = result.lines[0].width
    integrated_intensity = np.trapz(result.spectrum, result.frequencies)
    
    return {
        "success": True,
        "result_type": "absorption_spectrum",
        "frequencies": result.frequencies.tolist(),
        "spectrum": result.spectrum.tolist(),
        "peak_frequency": float(peak_freq),
        "peak_intensity": float(np.max(result.spectrum)),
        "fwhm_width": float(fwhm),
        "integrated_intensity": float(integrated_intensity),
        "broadening_type": broadening_type,
        "metadata": result.metadata,
        "analysis": {
            "frequency_range_hz": [f / (2 * np.pi) for f in frequency_range],
            "peak_wavelength_nm": const.c / (peak_freq / (2 * np.pi)) * 1e9,
            "quality_factor": peak_freq / fwhm if fwhm > 0 else float('inf'),
        }
    }


async def laser_atom_interaction(laser_intensity: float, laser_wavelength: float,
                               pulse_duration: float, ionization_potential: float,
                               interaction_type: str = "tunneling") -> Dict[str, Any]:
    """Calculate laser-atom interaction parameters."""
    logger.info(f"Calculating laser-atom interaction: I={laser_intensity} W/cm², λ={laser_wavelength} nm")
    
    interaction = LaserAtomInteraction()
    result = interaction.calculate_interaction(
        laser_intensity=laser_intensity,
        laser_wavelength=laser_wavelength,
        pulse_duration=pulse_duration,
        ionization_potential=ionization_potential,
        interaction_type=interaction_type
    )
    
    return {
        "success": True,
        "result_type": "laser_atom_interaction",
        **result,
        "summary": {
            "regime": result["interaction_parameters"]["interaction_regime"],
            "keldysh_parameter": result["interaction_parameters"]["keldysh_parameter"],
            "ponderomotive_energy_ev": result["interaction_parameters"]["ponderomotive_energy_ev"],
            "ionization_rate_hz": result["interaction_parameters"]["ionization_rate_hz"],
            "strong_field_regime": result["analysis"]["is_strong_field"],
        }
    }