"""
Utility Tools for AMO Physics
General utility functions and helper tools for the Rabi MCP server.
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import constants as const
from scipy.optimize import minimize_scalar, curve_fit
import json

from ..config.settings import settings

logger = logging.getLogger(__name__)


class PhysicalConstants:
    """Collection of physical constants relevant to AMO physics."""
    
    # Fundamental constants
    c = const.c  # Speed of light (m/s)
    h = const.h  # Planck constant (J⋅s)
    hbar = const.hbar  # Reduced Planck constant (J⋅s)
    e = const.e  # Elementary charge (C)
    me = const.m_e  # Electron mass (kg)
    mp = const.m_p  # Proton mass (kg)
    kB = const.k  # Boltzmann constant (J/K)
    epsilon0 = const.epsilon_0  # Vacuum permittivity (F/m)
    mu0 = const.mu_0  # Vacuum permeability (H/m)
    
    # Atomic units
    a0 = const.physical_constants['Bohr radius'][0]  # Bohr radius (m)
    Eh = const.physical_constants['Hartree energy'][0]  # Hartree energy (J)
    
    # Common atomic masses (in atomic mass units)
    masses = {
        'H': 1.007825,
        'He': 4.002603,
        'Li': 6.015122,
        'C': 12.000000,
        'N': 14.003074,
        'O': 15.994915,
        'Na': 22.989769,
        'Mg': 23.985042,
        'K': 38.963707,
        'Ca': 39.962591,
        'Rb': 86.909183,
        'Cs': 132.905447,
    }
    
    # Common transition wavelengths (nm)
    transitions = {
        'H_Lyman_alpha': 121.6,
        'He_D3': 587.6,
        'Li_D1': 670.8,
        'Li_D2': 670.8,
        'Na_D1': 589.6,
        'Na_D2': 589.0,
        'K_D1': 770.1,
        'K_D2': 766.5,
        'Rb_D1': 795.0,
        'Rb_D2': 780.2,
        'Cs_D1': 894.6,
        'Cs_D2': 852.3,
    }


class UnitConverter:
    """Unit conversion utilities for AMO physics."""
    
    @staticmethod
    def wavelength_to_frequency(wavelength_nm: float) -> float:
        """Convert wavelength to frequency."""
        return PhysicalConstants.c / (wavelength_nm * 1e-9)
    
    @staticmethod
    def frequency_to_wavelength(frequency_hz: float) -> float:
        """Convert frequency to wavelength (nm)."""
        return PhysicalConstants.c / frequency_hz * 1e9
    
    @staticmethod
    def energy_to_frequency(energy_j: float) -> float:
        """Convert energy to frequency."""
        return energy_j / PhysicalConstants.h
    
    @staticmethod
    def frequency_to_energy(frequency_hz: float) -> float:
        """Convert frequency to energy (J)."""
        return PhysicalConstants.h * frequency_hz
    
    @staticmethod
    def ev_to_joules(energy_ev: float) -> float:
        """Convert eV to Joules."""
        return energy_ev * PhysicalConstants.e
    
    @staticmethod
    def joules_to_ev(energy_j: float) -> float:
        """Convert Joules to eV."""
        return energy_j / PhysicalConstants.e
    
    @staticmethod
    def angular_to_linear_frequency(omega: float) -> float:
        """Convert angular frequency to linear frequency."""
        return omega / (2 * np.pi)
    
    @staticmethod
    def linear_to_angular_frequency(f: float) -> float:
        """Convert linear frequency to angular frequency."""
        return 2 * np.pi * f
    
    @staticmethod
    def intensity_to_electric_field(intensity_w_cm2: float) -> float:
        """Convert laser intensity to electric field amplitude (V/m)."""
        intensity_si = intensity_w_cm2 * 1e4  # Convert to W/m²
        return np.sqrt(2 * intensity_si / (PhysicalConstants.c * PhysicalConstants.epsilon0))
    
    @staticmethod
    def electric_field_to_intensity(E_field_vm: float) -> float:
        """Convert electric field amplitude to intensity (W/cm²)."""
        intensity_si = PhysicalConstants.c * PhysicalConstants.epsilon0 * E_field_vm**2 / 2
        return intensity_si / 1e4  # Convert to W/cm²
    
    @staticmethod
    def rabi_frequency_from_intensity(intensity_w_cm2: float, dipole_moment_debye: float) -> float:
        """Calculate Rabi frequency from laser intensity and dipole moment."""
        E_field = UnitConverter.intensity_to_electric_field(intensity_w_cm2)
        dipole_si = dipole_moment_debye * 3.336e-30  # Convert Debye to C⋅m
        return dipole_si * E_field / PhysicalConstants.hbar


class DataAnalysis:
    """Data analysis utilities for experimental data."""
    
    @staticmethod
    def fit_exponential_decay(times: np.ndarray, data: np.ndarray, 
                            initial_guess: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Fit exponential decay function to data.
        
        Args:
            times: Time points
            data: Data points
            initial_guess: Initial parameter guess [A, gamma, offset]
            
        Returns:
            Fit parameters and statistics
        """
        def exp_decay(t, A, gamma, offset):
            return A * np.exp(-gamma * t) + offset
        
        if initial_guess is None:
            A_guess = np.max(data) - np.min(data)
            gamma_guess = 1 / (times[-1] / 3)  # Rough estimate
            offset_guess = np.min(data)
            initial_guess = [A_guess, gamma_guess, offset_guess]
        
        try:
            popt, pcov = curve_fit(exp_decay, times, data, p0=initial_guess)
            
            # Calculate R²
            residuals = data - exp_decay(times, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((data - np.mean(data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            return {
                "amplitude": float(popt[0]),
                "decay_rate": float(popt[1]),
                "offset": float(popt[2]),
                "amplitude_error": float(param_errors[0]),
                "decay_rate_error": float(param_errors[1]),
                "offset_error": float(param_errors[2]),
                "r_squared": float(r_squared),
                "lifetime": float(1 / popt[1]) if popt[1] > 0 else float('inf'),
            }
        except Exception as e:
            logger.error(f"Exponential fit failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def fit_rabi_oscillations(times: np.ndarray, populations: np.ndarray,
                            initial_guess: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Fit Rabi oscillations to population data.
        
        Args:
            times: Time points
            populations: Population data
            initial_guess: Initial parameter guess [A, Omega, phi, offset, gamma]
            
        Returns:
            Fit parameters and statistics
        """
        def rabi_oscillation(t, A, Omega, phi, offset, gamma):
            return A * np.exp(-gamma * t) * np.cos(Omega * t + phi) + offset
        
        if initial_guess is None:
            A_guess = (np.max(populations) - np.min(populations)) / 2
            # Estimate frequency from FFT
            fft_vals = np.abs(np.fft.fft(populations - np.mean(populations)))
            freqs = np.fft.fftfreq(len(populations), times[1] - times[0])
            Omega_guess = 2 * np.pi * freqs[np.argmax(fft_vals[1:len(fft_vals)//2]) + 1]
            phi_guess = 0.0
            offset_guess = np.mean(populations)
            gamma_guess = 0.1 / times[-1]  # Small decay
            initial_guess = [A_guess, Omega_guess, phi_guess, offset_guess, gamma_guess]
        
        try:
            popt, pcov = curve_fit(rabi_oscillation, times, populations, p0=initial_guess)
            
            # Calculate R²
            residuals = populations - rabi_oscillation(times, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((populations - np.mean(populations))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            return {
                "amplitude": float(popt[0]),
                "rabi_frequency": float(popt[1]),
                "phase": float(popt[2]),
                "offset": float(popt[3]),
                "decay_rate": float(popt[4]),
                "amplitude_error": float(param_errors[0]),
                "rabi_frequency_error": float(param_errors[1]),
                "phase_error": float(param_errors[2]),
                "offset_error": float(param_errors[3]),
                "decay_rate_error": float(param_errors[4]),
                "r_squared": float(r_squared),
                "period": float(2 * np.pi / abs(popt[1])) if popt[1] != 0 else float('inf'),
            }
        except Exception as e:
            logger.error(f"Rabi oscillation fit failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def calculate_fft_spectrum(data: np.ndarray, sampling_rate: float) -> Dict[str, np.ndarray]:
        """
        Calculate FFT spectrum of data.
        
        Args:
            data: Input data
            sampling_rate: Sampling rate (Hz)
            
        Returns:
            Dictionary with frequencies and spectrum
        """
        N = len(data)
        
        # Calculate FFT
        fft_vals = np.fft.fft(data)
        frequencies = np.fft.fftfreq(N, 1/sampling_rate)
        
        # Take only positive frequencies
        pos_freqs = frequencies[:N//2]
        pos_spectrum = np.abs(fft_vals[:N//2])
        
        # Normalize
        pos_spectrum = pos_spectrum / N
        pos_spectrum[1:] *= 2  # Account for negative frequencies
        
        return {
            "frequencies": pos_freqs,
            "spectrum": pos_spectrum,
            "power_spectrum": pos_spectrum**2,
            "peak_frequency": float(pos_freqs[np.argmax(pos_spectrum[1:]) + 1]),
            "peak_amplitude": float(np.max(pos_spectrum[1:])),
        }


class ExperimentalTools:
    """Tools for experimental parameter calculations."""
    
    @staticmethod
    def doppler_cooling_limit(wavelength_nm: float, natural_linewidth_hz: float) -> Dict[str, float]:
        """
        Calculate Doppler cooling limit.
        
        Args:
            wavelength_nm: Transition wavelength (nm)
            natural_linewidth_hz: Natural linewidth (Hz)
            
        Returns:
            Cooling limit parameters
        """
        # Convert to SI units
        wavelength_m = wavelength_nm * 1e-9
        gamma = 2 * np.pi * natural_linewidth_hz
        
        # Recoil velocity
        v_recoil = PhysicalConstants.h / (PhysicalConstants.masses['Rb'] * const.u * wavelength_m)
        
        # Doppler temperature
        T_doppler = PhysicalConstants.hbar * gamma / (2 * PhysicalConstants.kB)
        
        # Doppler velocity
        v_doppler = np.sqrt(PhysicalConstants.kB * T_doppler / (PhysicalConstants.masses['Rb'] * const.u))
        
        return {
            "doppler_temperature_mk": T_doppler * 1e6,  # μK
            "doppler_velocity_ms": v_doppler,
            "recoil_velocity_ms": v_recoil,
            "recoil_temperature_mk": (PhysicalConstants.masses['Rb'] * const.u * v_recoil**2 / 
                                    (2 * PhysicalConstants.kB)) * 1e6,
        }
    
    @staticmethod
    def magnetic_trap_frequency(gradient_gauss_cm: float, mass_amu: float) -> float:
        """
        Calculate magnetic trap frequency.
        
        Args:
            gradient_gauss_cm: Magnetic field gradient (G/cm)
            mass_amu: Atomic mass (amu)
            
        Returns:
            Trap frequency (Hz)
        """
        # Convert to SI units
        gradient_si = gradient_gauss_cm * 1e-2  # G/cm to G/m
        gradient_si *= 1e-4  # G to T
        mass_kg = mass_amu * const.u
        
        # Magnetic moment (assume μ = μ_B)
        mu_B = const.physical_constants['Bohr magneton'][0]
        
        # Trap frequency
        omega = np.sqrt(mu_B * gradient_si / mass_kg)
        
        return omega / (2 * np.pi)  # Convert to Hz


# Tool handler functions
async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle utility tool calls."""
    try:
        if name == "unit_conversion":
            return await unit_conversion(**arguments)
        elif name == "data_analysis":
            return await data_analysis(**arguments)
        elif name == "physical_constants":
            return await physical_constants(**arguments)
        elif name == "experimental_parameters":
            return await experimental_parameters(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in utility tool {name}: {str(e)}")
        raise


async def unit_conversion(conversion_type: str, value: float, 
                        from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Perform unit conversions."""
    logger.info(f"Unit conversion: {value} {from_unit} -> {to_unit}")
    
    converter = UnitConverter()
    
    try:
        if conversion_type == "frequency_wavelength":
            if from_unit == "nm" and to_unit == "hz":
                result = converter.wavelength_to_frequency(value)
            elif from_unit == "hz" and to_unit == "nm":
                result = converter.frequency_to_wavelength(value)
            else:
                raise ValueError(f"Unsupported conversion: {from_unit} -> {to_unit}")
        
        elif conversion_type == "energy":
            if from_unit == "ev" and to_unit == "j":
                result = converter.ev_to_joules(value)
            elif from_unit == "j" and to_unit == "ev":
                result = converter.joules_to_ev(value)
            elif from_unit == "j" and to_unit == "hz":
                result = converter.energy_to_frequency(value)
            elif from_unit == "hz" and to_unit == "j":
                result = converter.frequency_to_energy(value)
            else:
                raise ValueError(f"Unsupported conversion: {from_unit} -> {to_unit}")
        
        elif conversion_type == "intensity_field":
            if from_unit == "w_cm2" and to_unit == "v_m":
                result = converter.intensity_to_electric_field(value)
            elif from_unit == "v_m" and to_unit == "w_cm2":
                result = converter.electric_field_to_intensity(value)
            else:
                raise ValueError(f"Unsupported conversion: {from_unit} -> {to_unit}")
        
        else:
            raise ValueError(f"Unknown conversion type: {conversion_type}")
        
        return {
            "success": True,
            "result_type": "unit_conversion",
            "input_value": value,
            "input_unit": from_unit,
            "output_value": float(result),
            "output_unit": to_unit,
            "conversion_type": conversion_type,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result_type": "unit_conversion_error",
        }


async def data_analysis(analysis_type: str, data: Dict[str, List[float]], 
                      **kwargs) -> Dict[str, Any]:
    """Perform data analysis."""
    logger.info(f"Data analysis: {analysis_type}")
    
    analyzer = DataAnalysis()
    
    try:
        if analysis_type == "exponential_decay":
            times = np.array(data["times"])
            values = np.array(data["values"])
            result = analyzer.fit_exponential_decay(times, values)
        
        elif analysis_type == "rabi_oscillations":
            times = np.array(data["times"])
            populations = np.array(data["populations"])
            result = analyzer.fit_rabi_oscillations(times, populations)
        
        elif analysis_type == "fft_spectrum":
            signal = np.array(data["signal"])
            sampling_rate = kwargs.get("sampling_rate", 1.0)
            result = analyzer.calculate_fft_spectrum(signal, sampling_rate)
            # Convert numpy arrays to lists for JSON serialization
            result["frequencies"] = result["frequencies"].tolist()
            result["spectrum"] = result["spectrum"].tolist()
            result["power_spectrum"] = result["power_spectrum"].tolist()
        
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        return {
            "success": True,
            "result_type": f"data_analysis_{analysis_type}",
            "analysis_results": result,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result_type": f"data_analysis_error",
        }


async def physical_constants(**kwargs) -> Dict[str, Any]:
    """Get physical constants."""
    constants = PhysicalConstants()
    
    return {
        "success": True,
        "result_type": "physical_constants",
        "fundamental_constants": {
            "speed_of_light": constants.c,
            "planck_constant": constants.h,
            "reduced_planck": constants.hbar,
            "elementary_charge": constants.e,
            "electron_mass": constants.me,
            "boltzmann_constant": constants.kB,
        },
        "atomic_units": {
            "bohr_radius": constants.a0,
            "hartree_energy": constants.Eh,
        },
        "atomic_masses": constants.masses,
        "common_transitions": constants.transitions,
    }


async def experimental_parameters(parameter_type: str, **kwargs) -> Dict[str, Any]:
    """Calculate experimental parameters."""
    logger.info(f"Calculating experimental parameters: {parameter_type}")
    
    tools = ExperimentalTools()
    
    try:
        if parameter_type == "doppler_cooling":
            wavelength = kwargs["wavelength_nm"]
            linewidth = kwargs["natural_linewidth_hz"]
            result = tools.doppler_cooling_limit(wavelength, linewidth)
        
        elif parameter_type == "magnetic_trap":
            gradient = kwargs["gradient_gauss_cm"]
            mass = kwargs["mass_amu"]
            result = {"trap_frequency_hz": tools.magnetic_trap_frequency(gradient, mass)}
        
        else:
            raise ValueError(f"Unknown parameter type: {parameter_type}")
        
        return {
            "success": True,
            "result_type": f"experimental_parameters_{parameter_type}",
            "parameters": result,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result_type": "experimental_parameters_error",
        }