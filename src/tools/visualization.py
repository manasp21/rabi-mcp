"""
Visualization Tools for AMO Physics
Advanced visualization tools for quantum states and dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from ..config.settings import settings

logger = logging.getLogger(__name__)


class BlochSphere:
    """3D Bloch sphere visualization for two-level quantum systems."""
    
    def __init__(self, style: str = "plotly"):
        """
        Initialize Bloch sphere visualization.
        
        Args:
            style: Plotting style ('plotly' or 'matplotlib')
        """
        self.style = style
        
    def create_sphere(self, state_vector: np.ndarray, show_trajectory: bool = False,
                     trajectory_data: Optional[List[np.ndarray]] = None,
                     title: str = "Quantum State") -> Dict[str, Any]:
        """
        Create Bloch sphere visualization.
        
        Args:
            state_vector: Complex state vector [c0, c1]
            show_trajectory: Whether to show evolution trajectory
            trajectory_data: List of state vectors for trajectory
            title: Plot title
            
        Returns:
            Dictionary with plot data and metadata
        """
        # Calculate Bloch vector
        bloch_vector = self._state_to_bloch(state_vector)
        
        if self.style == "plotly":
            return self._create_plotly_sphere(bloch_vector, show_trajectory, 
                                            trajectory_data, title)
        else:
            return self._create_matplotlib_sphere(bloch_vector, show_trajectory,
                                                trajectory_data, title)
    
    def _state_to_bloch(self, state: np.ndarray) -> np.ndarray:
        """Convert quantum state to Bloch vector."""
        # Normalize state
        state = state / np.linalg.norm(state)
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        
        # Calculate expectation values
        x = np.real(np.conj(state) @ sigma_x @ state)
        y = np.real(np.conj(state) @ sigma_y @ state)
        z = np.real(np.conj(state) @ sigma_z @ state)
        
        return np.array([x, y, z])
    
    def _create_plotly_sphere(self, bloch_vector: np.ndarray, show_trajectory: bool,
                            trajectory_data: Optional[List[np.ndarray]], title: str) -> Dict[str, Any]:
        """Create Plotly 3D Bloch sphere."""
        fig = go.Figure()
        
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name='Bloch Sphere'
        ))
        
        # Add coordinate axes
        axis_length = 1.2
        
        # X-axis (red)
        fig.add_trace(go.Scatter3d(
            x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
            mode='lines',
            line=dict(color='red', width=6),
            name='X-axis'
        ))
        
        # Y-axis (green)
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
            mode='lines',
            line=dict(color='green', width=6),
            name='Y-axis'
        ))
        
        # Z-axis (blue)
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
            mode='lines',
            line=dict(color='blue', width=6),
            name='Z-axis'
        ))
        
        # Add state vector
        fig.add_trace(go.Scatter3d(
            x=[0, bloch_vector[0]], y=[0, bloch_vector[1]], z=[0, bloch_vector[2]],
            mode='lines+markers',
            line=dict(color='black', width=8),
            marker=dict(size=10, color='red'),
            name='State Vector'
        ))
        
        # Add trajectory if requested
        if show_trajectory and trajectory_data:
            trajectory_bloch = [self._state_to_bloch(state) for state in trajectory_data]
            traj_x = [vec[0] for vec in trajectory_bloch]
            traj_y = [vec[1] for vec in trajectory_bloch]
            traj_z = [vec[2] for vec in trajectory_bloch]
            
            fig.add_trace(go.Scatter3d(
                x=traj_x, y=traj_y, z=traj_z,
                mode='lines+markers',
                line=dict(color='orange', width=4),
                marker=dict(size=3),
                name='Trajectory'
            ))
        
        # Add pole labels
        fig.add_annotation(x=0, y=0, z=1.3, text="|0⟩", showarrow=False, font=dict(size=16))
        fig.add_annotation(x=0, y=0, z=-1.3, text="|1⟩", showarrow=False, font=dict(size=16))
        fig.add_annotation(x=1.3, y=0, z=0, text="|+⟩", showarrow=False, font=dict(size=16))
        fig.add_annotation(x=-1.3, y=0, z=0, text="|-⟩", showarrow=False, font=dict(size=16))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        # Convert to HTML
        html_str = fig.to_html(include_plotlyjs='cdn')
        
        return {
            "plot_type": "bloch_sphere_3d",
            "html": html_str,
            "bloch_vector": bloch_vector.tolist(),
            "state_properties": {
                "theta": float(np.arccos(bloch_vector[2])),
                "phi": float(np.arctan2(bloch_vector[1], bloch_vector[0])),
                "purity": float(np.linalg.norm(bloch_vector)),
            }
        }
    
    def _create_matplotlib_sphere(self, bloch_vector: np.ndarray, show_trajectory: bool,
                                trajectory_data: Optional[List[np.ndarray]], title: str) -> Dict[str, Any]:
        """Create matplotlib 3D Bloch sphere."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='lightblue')
        
        # Add coordinate axes
        axis_length = 1.2
        ax.plot([-axis_length, axis_length], [0, 0], [0, 0], 'r-', linewidth=3, label='X')
        ax.plot([0, 0], [-axis_length, axis_length], [0, 0], 'g-', linewidth=3, label='Y')
        ax.plot([0, 0], [0, 0], [-axis_length, axis_length], 'b-', linewidth=3, label='Z')
        
        # Add state vector
        ax.quiver(0, 0, 0, bloch_vector[0], bloch_vector[1], bloch_vector[2], 
                 color='red', linewidth=4, arrow_length_ratio=0.1, label='State')
        
        # Add trajectory if requested
        if show_trajectory and trajectory_data:
            trajectory_bloch = [self._state_to_bloch(state) for state in trajectory_data]
            traj_x = [vec[0] for vec in trajectory_bloch]
            traj_y = [vec[1] for vec in trajectory_bloch]
            traj_z = [vec[2] for vec in trajectory_bloch]
            
            ax.plot(traj_x, traj_y, traj_z, 'orange', linewidth=2, label='Trajectory')
        
        # Labels
        ax.text(0, 0, 1.3, '|0⟩', fontsize=14, ha='center')
        ax.text(0, 0, -1.3, '|1⟩', fontsize=14, ha='center')
        ax.text(1.3, 0, 0, '|+⟩', fontsize=14, ha='center')
        ax.text(-1.3, 0, 0, '|-⟩', fontsize=14, ha='center')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        # Convert to base64 image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "plot_type": "bloch_sphere_3d",
            "image_base64": image_base64,
            "bloch_vector": bloch_vector.tolist(),
            "state_properties": {
                "theta": float(np.arccos(bloch_vector[2])),
                "phi": float(np.arctan2(bloch_vector[1], bloch_vector[0])),
                "purity": float(np.linalg.norm(bloch_vector)),
            }
        }


class PopulationDynamics:
    """Visualization tools for population dynamics."""
    
    def __init__(self, style: str = "plotly"):
        """Initialize population dynamics visualizer."""
        self.style = style
    
    def plot_dynamics(self, time_data: np.ndarray, population_data: np.ndarray,
                     level_labels: Optional[List[str]] = None,
                     title: str = "Population Dynamics") -> Dict[str, Any]:
        """
        Plot population dynamics.
        
        Args:
            time_data: Time points
            population_data: Population data [time, level]
            level_labels: Labels for energy levels
            title: Plot title
            
        Returns:
            Dictionary with plot data
        """
        if level_labels is None:
            level_labels = [f"Level {i}" for i in range(population_data.shape[1])]
        
        if self.style == "plotly":
            return self._create_plotly_dynamics(time_data, population_data, level_labels, title)
        else:
            return self._create_matplotlib_dynamics(time_data, population_data, level_labels, title)
    
    def _create_plotly_dynamics(self, time_data: np.ndarray, population_data: np.ndarray,
                              level_labels: List[str], title: str) -> Dict[str, Any]:
        """Create Plotly population dynamics plot."""
        fig = go.Figure()
        
        # Color palette
        colors = px.colors.qualitative.Set1
        
        for i, label in enumerate(level_labels):
            fig.add_trace(go.Scatter(
                x=time_data,
                y=population_data[:, i],
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)], width=3),
                hovertemplate=f'{label}<br>Time: %{{x:.3f}}<br>Population: %{{y:.3f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Population',
            yaxis=dict(range=[0, 1]),
            width=800,
            height=500,
            hovermode='x unified'
        )
        
        html_str = fig.to_html(include_plotlyjs='cdn')
        
        return {
            "plot_type": "population_dynamics",
            "html": html_str,
            "final_populations": population_data[-1].tolist(),
            "max_populations": np.max(population_data, axis=0).tolist(),
        }
    
    def _create_matplotlib_dynamics(self, time_data: np.ndarray, population_data: np.ndarray,
                                  level_labels: List[str], title: str) -> Dict[str, Any]:
        """Create matplotlib population dynamics plot."""
        plt.figure(figsize=(12, 6))
        
        for i, label in enumerate(level_labels):
            plt.plot(time_data, population_data[:, i], linewidth=3, label=label)
        
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Convert to base64 image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "plot_type": "population_dynamics",
            "image_base64": image_base64,
            "final_populations": population_data[-1].tolist(),
            "max_populations": np.max(population_data, axis=0).tolist(),
        }


class SpectrumVisualizer:
    """Visualization tools for spectroscopic data."""
    
    def __init__(self, style: str = "plotly"):
        """Initialize spectrum visualizer."""
        self.style = style
    
    def plot_spectrum(self, frequencies: np.ndarray, spectrum: np.ndarray,
                     title: str = "Spectrum", xlabel: str = "Frequency",
                     ylabel: str = "Intensity") -> Dict[str, Any]:
        """
        Plot spectrum.
        
        Args:
            frequencies: Frequency data
            spectrum: Spectrum data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Dictionary with plot data
        """
        if self.style == "plotly":
            return self._create_plotly_spectrum(frequencies, spectrum, title, xlabel, ylabel)
        else:
            return self._create_matplotlib_spectrum(frequencies, spectrum, title, xlabel, ylabel)
    
    def _create_plotly_spectrum(self, frequencies: np.ndarray, spectrum: np.ndarray,
                              title: str, xlabel: str, ylabel: str) -> Dict[str, Any]:
        """Create Plotly spectrum plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=spectrum,
            mode='lines',
            name='Spectrum',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,100,200,0.2)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=800,
            height=500,
            showlegend=False
        )
        
        html_str = fig.to_html(include_plotlyjs='cdn')
        
        return {
            "plot_type": "spectrum",
            "html": html_str,
            "peak_frequency": float(frequencies[np.argmax(spectrum)]),
            "peak_intensity": float(np.max(spectrum)),
        }
    
    def _create_matplotlib_spectrum(self, frequencies: np.ndarray, spectrum: np.ndarray,
                                  title: str, xlabel: str, ylabel: str) -> Dict[str, Any]:
        """Create matplotlib spectrum plot."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(frequencies, spectrum, 'b-', linewidth=2)
        plt.fill_between(frequencies, spectrum, alpha=0.3)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Convert to base64 image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "plot_type": "spectrum",
            "image_base64": image_base64,
            "peak_frequency": float(frequencies[np.argmax(spectrum)]),
            "peak_intensity": float(np.max(spectrum)),
        }


# Tool handler functions
async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle visualization tool calls."""
    try:
        if name == "plot_bloch_sphere":
            return await plot_bloch_sphere(**arguments)
        elif name == "plot_population_dynamics":
            return await plot_population_dynamics(**arguments)
        elif name == "plot_spectrum":
            return await plot_spectrum(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in visualization tool {name}: {str(e)}")
        raise


async def plot_bloch_sphere(state_vector: List[float], show_trajectory: bool = False,
                          trajectory_data: Optional[List[List[float]]] = None,
                          title: str = "Quantum State") -> Dict[str, Any]:
    """Create Bloch sphere visualization."""
    logger.info(f"Creating Bloch sphere visualization: {title}")
    
    # Convert to numpy array
    state = np.array(state_vector[:2])  # Take first two components
    if len(state_vector) > 2:
        # Handle complex numbers passed as [real, imag, real, imag]
        state = np.array([state_vector[0] + 1j*state_vector[1], 
                         state_vector[2] + 1j*state_vector[3]])
    
    trajectory = None
    if show_trajectory and trajectory_data:
        trajectory = [np.array(traj_point[:2]) for traj_point in trajectory_data]
    
    bloch_sphere = BlochSphere(style="plotly")
    result = bloch_sphere.create_sphere(state, show_trajectory, trajectory, title)
    
    return {
        "success": True,
        "result_type": "bloch_sphere_visualization",
        **result
    }


async def plot_population_dynamics(time_data: List[float], population_data: List[List[float]],
                                 level_labels: Optional[List[str]] = None,
                                 title: str = "Population Dynamics") -> Dict[str, Any]:
    """Create population dynamics plot."""
    logger.info(f"Creating population dynamics plot: {title}")
    
    time_array = np.array(time_data)
    pop_array = np.array(population_data)
    
    if level_labels is None:
        level_labels = [f"Level {i}" for i in range(pop_array.shape[1])]
    
    dynamics = PopulationDynamics(style="plotly")
    result = dynamics.plot_dynamics(time_array, pop_array, level_labels, title)
    
    return {
        "success": True,
        "result_type": "population_dynamics_plot",
        **result
    }


async def plot_spectrum(frequencies: List[float], spectrum: List[float],
                       title: str = "Spectrum", xlabel: str = "Frequency (Hz)",
                       ylabel: str = "Intensity") -> Dict[str, Any]:
    """Create spectrum plot."""
    logger.info(f"Creating spectrum plot: {title}")
    
    freq_array = np.array(frequencies)
    spec_array = np.array(spectrum)
    
    visualizer = SpectrumVisualizer(style="plotly")
    result = visualizer.plot_spectrum(freq_array, spec_array, title, xlabel, ylabel)
    
    return {
        "success": True,
        "result_type": "spectrum_plot",
        **result
    }