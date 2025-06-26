"""
Physical Constants for AMO Physics
Comprehensive collection of physical constants, atomic data, and unit conversions.
"""

import numpy as np
from scipy import constants as const
from typing import Dict, Any, Optional


class FundamentalConstants:
    """Fundamental physical constants in SI units."""
    
    # Speed of light
    c = const.c  # m/s
    
    # Planck constants
    h = const.h  # J⋅s
    hbar = const.hbar  # J⋅s
    
    # Electromagnetic constants
    e = const.e  # C (elementary charge)
    epsilon_0 = const.epsilon_0  # F/m (vacuum permittivity)
    mu_0 = const.mu_0  # H/m (vacuum permeability)
    
    # Particle masses
    m_e = const.m_e  # kg (electron mass)
    m_p = const.m_p  # kg (proton mass)
    m_n = const.m_n  # kg (neutron mass)
    u = const.u  # kg (atomic mass unit)
    
    # Boltzmann constant
    k_B = const.k  # J/K
    
    # Fine structure constant
    alpha = const.alpha  # dimensionless
    
    # Avogadro's number
    N_A = const.N_A  # mol⁻¹


class AtomicUnits:
    """Atomic units (Hartree system)."""
    
    # Length
    a0 = const.physical_constants['Bohr radius'][0]  # m
    
    # Energy
    Eh = const.physical_constants['Hartree energy'][0]  # J
    Eh_eV = const.physical_constants['Hartree energy in eV'][0]  # eV
    
    # Electric field
    E_h = const.e / (4 * np.pi * const.epsilon_0 * const.a_0**2)  # V/m
    
    # Time
    t_au = const.hbar / AtomicUnits.Eh  # s
    
    # Velocity
    v_au = const.alpha * const.c  # m/s


class AtomicData:
    """Atomic data for common elements used in AMO physics."""
    
    # Atomic masses (in atomic mass units)
    masses = {
        'H': 1.007825032,
        'D': 2.014101778,
        'T': 3.016049268,
        'He': 4.002603254,
        'Li': 6.015122887,
        'Be': 9.012182201,
        'B': 10.012936862,
        'C': 12.000000000,
        'N': 14.003074004,
        'O': 15.994914620,
        'F': 18.998403163,
        'Ne': 19.992440175,
        'Na': 22.989769282,
        'Mg': 23.985041697,
        'Al': 26.981538627,
        'Si': 27.976926533,
        'P': 30.973761998,
        'S': 31.972071174,
        'Cl': 34.968852682,
        'Ar': 39.962383123,
        'K': 38.963706679,
        'Ca': 39.962590863,
        'Rb': 86.909180527,
        'Sr': 87.905614306,
        'Cs': 132.905451931,
        'Ba': 137.905247237,
        'Yb': 173.938862089,
    }
    
    # Ionization potentials (in eV)
    ionization_potentials = {
        'H': 13.59844,
        'He': 24.58741,
        'Li': 5.39172,
        'Be': 9.32270,
        'B': 8.29803,
        'C': 11.26030,
        'N': 14.53414,
        'O': 13.61806,
        'F': 17.42282,
        'Ne': 21.56454,
        'Na': 5.13908,
        'Mg': 7.64624,
        'Al': 5.98577,
        'Si': 8.15169,
        'P': 10.48669,
        'S': 10.36001,
        'Cl': 12.96764,
        'Ar': 15.75962,
        'K': 4.34066,
        'Ca': 6.11316,
        'Rb': 4.17713,
        'Sr': 5.69484,
        'Cs': 3.89390,
        'Ba': 5.21170,
        'Yb': 6.25416,
    }
    
    # Common D-line transitions (wavelength in nm)
    d_lines = {
        'Li': {'D1': 670.776, 'D2': 670.791},
        'Na': {'D1': 589.756, 'D2': 588.995},
        'K': {'D1': 770.108, 'D2': 766.490},
        'Rb': {'D1': 794.979, 'D2': 780.241},
        'Cs': {'D1': 894.593, 'D2': 852.347},
    }
    
    # Natural linewidths (in MHz)
    natural_linewidths = {
        'Li_D1': 5.872,
        'Li_D2': 5.872,
        'Na_D1': 9.795,
        'Na_D2': 9.795,
        'K_D1': 6.035,
        'K_D2': 6.035,
        'Rb_D1': 5.746,
        'Rb_D2': 6.066,
        'Cs_D1': 4.560,
        'Cs_D2': 5.234,
    }
    
    # Nuclear spins
    nuclear_spins = {
        'H': 1/2,
        'Li': 3/2,
        'Na': 3/2,
        'K': 3/2,
        'Rb': 3/2,
        'Cs': 7/2,
        'Sr': 0,
        'Yb': 0,
    }


class LaserParameters:
    """Common laser parameters and wavelengths."""
    
    # Common laser wavelengths (in nm)
    wavelengths = {
        'HeNe_red': 632.8,
        'HeNe_green': 543.5,
        'Ar_blue': 488.0,
        'Ar_green': 514.5,
        'Ti_sapphire_center': 800.0,
        'Nd_YAG_fundamental': 1064.0,
        'Nd_YAG_second_harmonic': 532.0,
        'Nd_YAG_third_harmonic': 355.0,
        'Nd_YAG_fourth_harmonic': 266.0,
        'CO2': 10600.0,
        'diode_780': 780.0,
        'diode_852': 852.0,
        'diode_895': 895.0,
    }
    
    # Laser linewidths (typical values in Hz)
    linewidths = {
        'HeNe': 1e6,      # ~1 MHz
        'diode_free': 1e6,  # ~1 MHz
        'diode_stabilized': 1e3,  # ~1 kHz
        'Ti_sapphire': 1e5,  # ~100 kHz
        'external_cavity': 1e3,  # ~1 kHz
        'stabilized_cavity': 1,  # ~1 Hz
    }


class MagneticFields:
    """Magnetic field related constants."""
    
    # Bohr magneton
    mu_B = const.physical_constants['Bohr magneton'][0]  # J/T
    mu_B_eV_T = const.physical_constants['Bohr magneton in eV/T'][0]  # eV/T
    
    # Nuclear magneton
    mu_N = const.physical_constants['nuclear magneton'][0]  # J/T
    mu_N_eV_T = const.physical_constants['nuclear magneton in eV/T'][0]  # eV/T
    
    # Landé g-factors (typical values)
    g_factors = {
        'electron': 2.002319304,
        'proton': 5.585694702,
        'neutron': -3.826085455,
    }
    
    # Earth's magnetic field (typical values in Tesla)
    earth_field = {
        'horizontal': 2e-5,  # ~20 μT
        'vertical': 4e-5,    # ~40 μT
        'total': 5e-5,       # ~50 μT
    }


class TrapParameters:
    """Parameters for common atomic traps."""
    
    # Typical trap depths (in μK)
    trap_depths = {
        'magneto_optical': 1000,    # 1 mK
        'magnetic': 100,            # 100 μK
        'optical_dipole': 10,       # 10 μK
        'optical_lattice': 100,     # 100 μK
    }
    
    # Typical trap frequencies (in Hz)
    trap_frequencies = {
        'magnetic_quadrupole': 100,   # 100 Hz
        'optical_dipole_tight': 1000, # 1 kHz
        'optical_dipole_loose': 10,   # 10 Hz
        'optical_lattice': 10000,     # 10 kHz
    }


class ConversionFactors:
    """Unit conversion factors."""
    
    # Energy conversions
    J_to_eV = 1 / const.e
    eV_to_J = const.e
    K_to_eV = const.k / const.e  # Boltzmann constant in eV/K
    eV_to_K = const.e / const.k
    Hz_to_eV = const.h / const.e
    eV_to_Hz = const.e / const.h
    
    # Length conversions
    m_to_nm = 1e9
    nm_to_m = 1e-9
    m_to_a0 = 1 / AtomicUnits.a0
    a0_to_m = AtomicUnits.a0
    
    # Time conversions
    s_to_ns = 1e9
    ns_to_s = 1e-9
    s_to_fs = 1e15
    fs_to_s = 1e-15
    
    # Frequency conversions
    Hz_to_rad_s = 2 * np.pi
    rad_s_to_Hz = 1 / (2 * np.pi)
    
    # Temperature conversions
    K_to_mK = 1e3
    mK_to_K = 1e-3
    K_to_uK = 1e6
    uK_to_K = 1e-6
    K_to_nK = 1e9
    nK_to_K = 1e-9


def get_atomic_property(element: str, property_name: str) -> Optional[float]:
    """
    Get atomic property for a given element.
    
    Args:
        element: Element symbol (e.g., 'Rb', 'Cs')
        property_name: Property name ('mass', 'ionization_potential', etc.)
        
    Returns:
        Property value or None if not found
    """
    if property_name == 'mass':
        return AtomicData.masses.get(element)
    elif property_name == 'ionization_potential':
        return AtomicData.ionization_potentials.get(element)
    elif property_name == 'nuclear_spin':
        return AtomicData.nuclear_spins.get(element)
    else:
        return None


def calculate_recoil_energy(wavelength_nm: float, mass_amu: float) -> float:
    """
    Calculate photon recoil energy.
    
    Args:
        wavelength_nm: Photon wavelength (nm)
        mass_amu: Atomic mass (amu)
        
    Returns:
        Recoil energy (J)
    """
    wavelength_m = wavelength_nm * 1e-9
    mass_kg = mass_amu * const.u
    
    momentum = const.h / wavelength_m
    recoil_energy = momentum**2 / (2 * mass_kg)
    
    return recoil_energy


def calculate_doppler_width(frequency_hz: float, temperature_k: float, mass_amu: float) -> float:
    """
    Calculate Doppler broadening width.
    
    Args:
        frequency_hz: Transition frequency (Hz)
        temperature_k: Temperature (K)
        mass_amu: Atomic mass (amu)
        
    Returns:
        Doppler width (FWHM) in Hz
    """
    mass_kg = mass_amu * const.u
    
    # Most probable velocity
    v_mp = np.sqrt(2 * const.k * temperature_k / mass_kg)
    
    # Doppler width (FWHM)
    doppler_width = 2 * frequency_hz * v_mp / const.c * np.sqrt(np.log(2))
    
    return doppler_width


def wavelength_to_energy(wavelength_nm: float) -> float:
    """Convert wavelength to photon energy (eV)."""
    wavelength_m = wavelength_nm * 1e-9
    energy_j = const.h * const.c / wavelength_m
    return energy_j / const.e  # Convert to eV


def energy_to_wavelength(energy_ev: float) -> float:
    """Convert photon energy to wavelength (nm)."""
    energy_j = energy_ev * const.e
    wavelength_m = const.h * const.c / energy_j
    return wavelength_m * 1e9  # Convert to nm