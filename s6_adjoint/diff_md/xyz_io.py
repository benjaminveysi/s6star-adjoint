"""
Molecular XYZ Trajectory File I/O for standalone molecular dynamics simulation.
Readable by standard molecular viewers (Ovito, VMD, PyMOL, Chimera, and web viewers).
"""
import os
import numpy as np

ELEMENT_NAMES = {0: "Ar", 1: "H", 2: "He", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne"}

def write_xyz_trajectory(filepath: str, coords_history: list, species_list: list, comments: list = None):
    """
    Writes a multi-frame atomic trajectory to an XYZ file.
    coords_history: list of (M, 3) numpy arrays.
    species_list: list of integer atomic numbers or element string symbols.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) if os.path.dirname(os.path.abspath(filepath)) else ".", exist_ok=True)
    num_atoms = len(species_list)
    symbols = [ELEMENT_NAMES.get(s, str(s)) if isinstance(s, int) else str(s) for s in species_list]

    with open(filepath, "w") as f:
        for frame_idx, coords in enumerate(coords_history):
            comment = comments[frame_idx] if comments and frame_idx < len(comments) else f"Frame {frame_idx}"
            f.write(f"{num_atoms}\n")
            f.write(f"{comment}\n")
            for sym, (x, y, z) in zip(symbols, coords):
                f.write(f"{sym:<3} {x:>12.6f} {y:>12.6f} {z:>12.6f}\n")


def save_xyz_trajectory(trajectory_q: list, filepath: str, element: str = "Ar", comments: list = None):
    """
    Convenience wrapper: saves a list of (3 * M,) flattened tensors or (M, 3) arrays to XYZ.
    """
    coords_np = []
    for q in trajectory_q:
        if hasattr(q, "detach"):
            arr = q.detach().cpu().numpy()
        else:
            arr = np.array(q)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        coords_np.append(arr)

    num_atoms = coords_np[0].shape[0]
    species_list = [element] * num_atoms
    write_xyz_trajectory(filepath, coords_np, species_list, comments)


def read_xyz_trajectory(filepath: str):
    """
    Reads a multi-frame atomic trajectory from an XYZ file.
    Returns:
        frames: list of (M, 3) numpy arrays.
        symbols: list of atomic symbols.
    """
    frames = []
    symbols = []
    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                break
            num_atoms = int(line_str)
            comment = f.readline().strip()
            frame_coords = []
            frame_symbols = []
            for _ in range(num_atoms):
                parts = f.readline().split()
                frame_symbols.append(parts[0])
                frame_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            frames.append(np.array(frame_coords))
            symbols = frame_symbols
    return frames, symbols
