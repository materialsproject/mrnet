from typing import List, Tuple
import copy

from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError


def extract_bond_environment(mg: MoleculeGraph, bonds: List[Tuple[int, int]], order=1) -> set:
    """
    Extract the local environment of a particular chemical bond in a MoleculeGraph

    :param bonds:
    :param order:

    :return: set of integers representing the relevant atom indices
    """

    indices = set()
    if order < 0:
        return indices
    elif order == 0:
        for bond in bonds:
            indices.add(bond[0])
            indices.add(bond[1])
        return indices
    else:
        graph = mg.graph.to_undirected()
        for bond in bonds:
            sub_bonds = list()
            for neighbor in graph[bond[0]]:
                sub_bonds.append((bond[0], neighbor))
            for neighbor in graph[bond[1]]:
                sub_bonds.append((bond[1], neighbor))
            indices = indices.union(extract_bond_environment(mg, sub_bonds, order - 1))
        return indices


def fragment_mol_graph(mol_graph, bonds):
    """
    Break a bond in molecule graph and obtain the fragment(s).
    Args:
        mol_graph (MoleculeGraph): molecule graph to fragment
        bonds (list): bond indices (2-tuple)
    Returns:
        dict: with bond index (2-tuple) as key, and a list of fragments (mol_graphs)
            as values. Each list could be of size 1 or 2 and could be empty if the
            mol has no bonds.
    """
    sub_mols = {}

    for edge in bonds:
        edge = tuple(edge)
        try:
            new_mgs = mol_graph.split_molecule_subgraphs(
                [edge], allow_reverse=True, alterations=None
            )
            sub_mols[edge] = new_mgs
        except MolGraphSplitError:  # cannot split, (breaking a bond in a ring)
            new_mg = copy.deepcopy(mol_graph)
            idx1, idx2 = edge
            new_mg.break_edge(idx1, idx2, allow_reverse=True)
            sub_mols[edge] = [new_mg]
    return sub_mols