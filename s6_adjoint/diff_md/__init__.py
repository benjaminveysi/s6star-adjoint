"""
Differentiable Molecular Dynamics Subpackage with S6*-Adjoint.
Includes:
- Potentials (LennardJonesPotential, HarmonicChainPotential, DoubleWellPotential, NeuralInteratomicPotential)
- Thermodynamic Observables (Energy, Temperature, MSD, Differentiable RDF, Virial Pressure)
- Ensemble Initializers for Batched Parallel Replicas
- XYZ Trajectory IO
- Standalone Simulator
"""
from .potentials import (
    LennardJonesPotential,
    HarmonicChainPotential,
    DoubleWellPotential,
    NeuralInteratomicPotential
)
from .observables import (
    compute_kinetic_energy,
    compute_total_energy,
    compute_temperature,
    compute_mean_squared_displacement,
    compute_differentiable_rdf,
    compute_differentiable_virial_pressure,
    compute_radial_distribution_function,
    generate_ensemble_configurations
)
from .xyz_io import save_xyz_trajectory, read_xyz_trajectory
from .simulator import DifferentiableMDSimulator

__all__ = [
    "LennardJonesPotential",
    "HarmonicChainPotential",
    "DoubleWellPotential",
    "NeuralInteratomicPotential",
    "compute_kinetic_energy",
    "compute_total_energy",
    "compute_temperature",
    "compute_mean_squared_displacement",
    "compute_differentiable_rdf",
    "compute_differentiable_virial_pressure",
    "compute_radial_distribution_function",
    "generate_ensemble_configurations",
    "save_xyz_trajectory",
    "read_xyz_trajectory",
    "DifferentiableMDSimulator",
]
