from pymatgen import Molecule
from pymatgen.analysis.graphs import MoleculeGraph


def create_LiEC_pymatgen_mol():
    """
            O(3) -- Li(6)
            ||
             C(1)
           /   \
          O(0)  O(4)
          |     |
        C(2) --- C(5)
    """
    atoms = ["O", "C", "C", "O", "O", "C", "Li", "H", "H", "H", "H"]
    coords = [
        [0.3103, -1.1776, -0.3722],
        [-0.6822, -0.5086, 0.3490],
        [1.5289, -0.4938, -0.0925],
        [-1.9018, -0.6327, -0.0141],
        [-0.2475, 0.9112, 0.3711],
        [1.1084, 0.9722, -0.0814],
        [-2.0519, 1.1814, -0.2310],
        [2.2514, -0.7288, -0.8736],
        [1.9228, -0.8043, 0.8819],
        [1.1406, 1.4103, -1.0835],
        [1.7022, 1.5801, 0.6038],
    ]
    charge = 0

    m = Molecule(atoms, coords, charge)

    return m


def create_LiEC_mol_graph():
    bonds = [
        (0, 2),
        (0, 1),
        (2, 5),
        (2, 8),
        (4, 1),
        (5, 4),
        (5, 10),
        (7, 2),
        (9, 5),
        (3, 6),
        (3, 1),
    ]
    bonds = {b: None for b in bonds}

    mol = create_LiEC_pymatgen_mol()
    mol_graph = MoleculeGraph.with_edges(mol, bonds)

    return mol_graph
