import copy
import warnings
import logging

import numpy as np
import networkx as nx
from openbabel import openbabel as ob

from rdkit import Chem
from rdkit.Chem import BondType, AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

import pymatgen
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError

from mrnet.utils.graphs import fragment_mol_graph
from mrnet.utils.molecules import remove_metals
from mrnet.utils.utils import to_path, create_directory

logger = logging.getLogger(__name__)


def create_rdkit_mol_from_mol_graph(
    mol_graph, name=None, force_sanitize=False, metals={"Li": 1, "Mg": 2}
):
    """
    Create a rdkit molecule from molecule graph, with bond type perceived by babel.
    Done in the below steps:
    1. create a babel mol without metal atoms.
    2. perceive bond order (conducted by BabelMolAdaptor)
    3. adjust formal charge of metal atoms so as not to violate valence rule
    4. create rdkit mol based on species, coords, bonds, and formal charge
    Args:
        mol_graph (pymatgen MoleculeGraph): molecule graph
        name (str): name of the molecule
        force_sanitize (bool): whether to force sanitization of the rdkit mol
        metals dict: with metal atom (str) as key and the number of valence electrons
            as key.
    Returns:
        m: rdkit Chem.Mol
        bond_types (dict): bond types assigned to the created rdkit mol
    """

    pymatgen_mol = mol_graph.molecule
    species = [str(s) for s in pymatgen_mol.species]
    coords = pymatgen_mol.cart_coords
    bonds = [tuple(sorted([i, j])) for i, j, attr in mol_graph.graph.edges.data()]

    # create babel mol without metals
    pmg_mol_no_metals = remove_metals(pymatgen_mol)
    adaptor = BabelMolAdaptor(pmg_mol_no_metals)
    ob_mol = adaptor.openbabel_mol

    # get babel bond order of mol without metals
    ob_bond_order = {}
    for bd in ob.OBMolBondIter(ob_mol):
        k = tuple(sorted([bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()]))
        v = bd.GetBondOrder()
        ob_bond_order[k] = v

    # create bond type
    atom_idx_mapping = pymatgen_to_babel_atom_idx_map(pymatgen_mol, ob_mol)
    bond_types = {}

    for bd in bonds:
        try:
            ob_bond = [atom_idx_mapping[a] for a in bd]

            # atom not in ob mol
            if None in ob_bond:
                raise KeyError
            # atom in ob mol
            else:
                ob_bond = tuple(sorted(ob_bond))
                v = ob_bond_order[ob_bond]
                if v == 0:
                    tp = BondType.UNSPECIFIED
                elif v == 1:
                    tp = BondType.SINGLE
                elif v == 2:
                    tp = BondType.DOUBLE
                elif v == 3:
                    tp = BondType.TRIPLE
                elif v == 5:
                    tp = BondType.AROMATIC
                else:
                    raise RuntimeError(f"Got unexpected babel bond order: {v}")

        except KeyError:
            atom1_spec, atom2_spec = [species[a] for a in bd]

            if atom1_spec in metals and atom2_spec in metals:
                raise RuntimeError("Got a bond between two metal atoms")

            # bond involves one and only one metal atom (atom not in ob mol case above)
            elif atom1_spec in metals or atom2_spec in metals:
                tp = Chem.rdchem.BondType.DATIVE

                # Dative bonds have the special characteristic that they do not affect
                # the valence on the start atom, but do affect the end atom.
                # Here we adjust the atom ordering in the bond for dative bond to make
                # metal the end atom.
                if atom1_spec in metals:
                    bd = tuple(reversed(bd))

            # bond not found by babel (atom in ob mol)
            else:
                tp = Chem.rdchem.BondType.UNSPECIFIED

        bond_types[bd] = tp

    # a metal atom can form multiple dative bond (e.g. bidentate LiEC), for such cases
    # we need to adjust the their formal charge so as not to violate valence rule
    formal_charge = adjust_formal_charge(species, bonds, metals)

    m = create_rdkit_mol(species, coords, bond_types, formal_charge, name, force_sanitize)

    return m, bond_types


def create_rdkit_mol(
    species, coords, bond_types, formal_charge=None, name=None, force_sanitize=True
):
    """
    Create a rdkit mol from scratch.
    Followed: https://sourceforge.net/p/rdkit/mailman/message/36474923/
    Args:
        species (list): species str of each molecule
        coords (2D array): positions of atoms
        bond_types (dict): with bond indices (2 tuple) as key and bond type
            (e.g. Chem.rdchem.BondType.DOUBLE) as value
        formal_charge (list): formal charge of each atom
        name (str): name of the molecule
        force_sanitize (bool): whether to force the sanitization of molecule.
            If `True` and the sanitization fails, it generally throw an error
            and then stops. If `False`, will try to sanitize first, but if it fails,
            will proceed smoothly giving a warning message.
    Returns:
        rdkit Chem.Mol
    """

    m = Chem.Mol()
    edm = Chem.EditableMol(m)
    conformer = Chem.Conformer(len(species))

    for i, (s, c) in enumerate(zip(species, coords)):
        atom = Chem.Atom(s)
        atom.SetNoImplicit(True)
        if formal_charge is not None:
            cg = formal_charge[i]
            if cg is not None:
                atom.SetFormalCharge(cg)
        atom_idx = edm.AddAtom(atom)
        conformer.SetAtomPosition(atom_idx, Point3D(*c))

    for b, t in bond_types.items():
        edm.AddBond(b[0], b[1], t)

    m = edm.GetMol()
    if force_sanitize:
        Chem.SanitizeMol(m)
    else:
        try:
            Chem.SanitizeMol(m)
        except Exception as e:
            warnings.warn(f"Cannot sanitize molecule {name}, because {str(e)}")
    m.AddConformer(conformer, assignId=False)

    m.SetProp("_Name", str(name))

    return m


def create_rdkit_mol_from_mol_graph(
    mol_graph, name=None, force_sanitize=False, metals={"Li": 1, "Mg": 2}
):
    """
    Create a rdkit molecule from molecule graph, with bond type perceived by babel.
    Done in the below steps:
    1. create a babel mol without metal atoms.
    2. perceive bond order (conducted by BabelMolAdaptor)
    3. adjust formal charge of metal atoms so as not to violate valence rule
    4. create rdkit mol based on species, coords, bonds, and formal charge
    Args:
        mol_graph (pymatgen MoleculeGraph): molecule graph
        name (str): name of the molecule
        force_sanitize (bool): whether to force sanitization of the rdkit mol
        metals dict: with metal atom (str) as key and the number of valence electrons
            as key.
    Returns:
        m: rdkit Chem.Mol
        bond_types (dict): bond types assigned to the created rdkit mol
    """

    pymatgen_mol = mol_graph.molecule
    species = [str(s) for s in pymatgen_mol.species]
    coords = pymatgen_mol.cart_coords
    bonds = [tuple(sorted([i, j])) for i, j, attr in mol_graph.graph.edges.data()]

    # create babel mol without metals
    pmg_mol_no_metals = remove_metals(pymatgen_mol)
    adaptor = BabelMolAdaptor(pmg_mol_no_metals)
    ob_mol = adaptor.openbabel_mol

    # get babel bond order of mol without metals
    ob_bond_order = {}
    for bd in ob.OBMolBondIter(ob_mol):
        k = tuple(sorted([bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()]))
        v = bd.GetBondOrder()
        ob_bond_order[k] = v

    # create bond type
    atom_idx_mapping = pymatgen_to_babel_atom_idx_map(pymatgen_mol, ob_mol)
    bond_types = {}

    for bd in bonds:
        try:
            ob_bond = [atom_idx_mapping[a] for a in bd]

            # atom not in ob mol
            if None in ob_bond:
                raise KeyError
            # atom in ob mol
            else:
                ob_bond = tuple(sorted(ob_bond))
                v = ob_bond_order[ob_bond]
                if v == 0:
                    tp = BondType.UNSPECIFIED
                elif v == 1:
                    tp = BondType.SINGLE
                elif v == 2:
                    tp = BondType.DOUBLE
                elif v == 3:
                    tp = BondType.TRIPLE
                elif v == 5:
                    tp = BondType.AROMATIC
                else:
                    raise RuntimeError(f"Got unexpected babel bond order: {v}")

        except KeyError:
            atom1_spec, atom2_spec = [species[a] for a in bd]

            if atom1_spec in metals and atom2_spec in metals:
                raise RuntimeError("Got a bond between two metal atoms")

            # bond involves one and only one metal atom (atom not in ob mol case above)
            elif atom1_spec in metals or atom2_spec in metals:
                tp = Chem.rdchem.BondType.DATIVE

                # Dative bonds have the special characteristic that they do not affect
                # the valence on the start atom, but do affect the end atom.
                # Here we adjust the atom ordering in the bond for dative bond to make
                # metal the end atom.
                if atom1_spec in metals:
                    bd = tuple(reversed(bd))

            # bond not found by babel (atom in ob mol)
            else:
                tp = Chem.rdchem.BondType.UNSPECIFIED

        bond_types[bd] = tp

    # a metal atom can form multiple dative bond (e.g. bidentate LiEC), for such cases
    # we need to adjust the their formal charge so as not to violate valence rule
    formal_charge = adjust_formal_charge(species, bonds, metals)

    m = create_rdkit_mol(species, coords, bond_types, formal_charge, name, force_sanitize)

    return m, bond_types


def pymatgen_to_babel_atom_idx_map(pmg_mol, ob_mol):
    """
    Create an atom index mapping between pymatgen mol and openbabel mol.
    This does not require pymatgen mol and ob mol has the same number of atoms.
    But ob_mol can have smaller number of atoms.
    Args:
        pmg_mol (pymatgen.Molecule): pymatgen molecule
        ob_mol (ob.Mol): openbabel molecule
    Returns:
        dict: with atom index in pymatgen mol as key and atom index in babel mol as
            value. Value is `None` if there is not corresponding atom in babel.
    """

    pmg_coords = pmg_mol.cart_coords
    ob_coords = [[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(ob_mol)]
    ob_index = [a.GetIdx() for a in ob.OBMolAtomIter(ob_mol)]

    mapping = {i: None for i in range(len(pmg_coords))}

    for idx, oc in zip(ob_index, ob_coords):
        for i, gc in enumerate(pmg_coords):
            if np.allclose(oc, gc):
                mapping[i] = idx
                break
        else:
            raise RuntimeError("Cannot create atom index mapping pymatgen and ob mols")

    return mapping


def adjust_formal_charge(species, bonds, metals):
    """
    Adjust formal charge of metal atoms.
    Args:
        species (list): species string of atoms
        bonds (list): 2-tuple index of bonds
        metals (dict): intial formal charge of models
    Returns:
        list: formal charge of atoms. None for non metal atoms.
    """
    # initialize formal charge first so that atom does not form any bond has its formal
    # charge set
    formal_charge = [metals[s] if s in metals else None for s in species]

    # atom_idx: idx of atoms in the molecule
    # num_bonds: number of bonds the atom forms
    atom_idx, num_bonds = np.unique(bonds, return_counts=True)
    for i, ct in zip(atom_idx, num_bonds):
        s = species[i]
        if s in metals:
            formal_charge[i] = int(formal_charge[i] - ct)

    return formal_charge


class MoleculeWrapper:
    """
    A wrapper of pymatgen Molecule, MoleculeGraph, rdkit Chem.Mol... to make it
    easier to use molecules.
    Arguments:
        mol_graph (MoleculeGraph): pymatgen molecule graph instance
        free_energy (float): free energy of the molecule
        id (str): (unique) identification of the molecule
    """

    def __init__(self, mol_graph, free_energy=None, id=None):
        self.mol_graph = mol_graph
        self.pymatgen_mol = mol_graph.molecule
        self.free_energy = free_energy
        self.id = id

        # set when corresponding method is called
        self._rdkit_mol = None
        self._fragments = None
        self._isomorphic_bonds = None

    @property
    def charge(self):
        """
        Returns:
            int: charge of the molecule
        """
        return self.pymatgen_mol.charge

    @property
    def formula(self):
        """
        Returns:
            str: chemical formula of the molecule, e.g. H2CO3.
        """
        return self.pymatgen_mol.composition.alphabetical_formula.replace(" ", "")

    @property
    def composition_dict(self):
        """
        Returns:
            dict: with chemical species as key and number of the species as value.
        """
        d = self.pymatgen_mol.composition.as_dict()
        return {k: int(v) for k, v in d.items()}

    @property
    def weight(self):
        """
        Returns:
            int: molecule weight
        """
        return self.pymatgen_mol.composition.weight

    @property
    def num_atoms(self):
        """
        Returns:
            int: number of atoms in molecule
        """
        return len(self.pymatgen_mol)

    @property
    def species(self):
        """
        Species of atoms. Order is the same as self.atoms.
        Returns:
            list: Species string.
        """
        return [str(s) for s in self.pymatgen_mol.species]

    @property
    def coords(self):
        """
        Returns:
            2D array: of shape (N, 3) where N is the number of atoms.
        """
        return np.asarray(self.pymatgen_mol.cart_coords)

    @property
    def bonds(self):
        """
        Returns:
            dict: with bond index (a tuple of atom indices) as the key and and bond
                attributes as the value.
        """
        return {tuple(sorted([i, j])): attr for i, j, attr in self.graph.edges.data()}

    @property
    def graph(self):
        """
        Returns:
            networkx graph used by mol_graph
        """
        return self.mol_graph.graph

    @property
    def rdkit_mol(self):
        """
        Returns:
            rdkit molecule
        """
        if self._rdkit_mol is None:
            self._rdkit_mol, _ = create_rdkit_mol_from_mol_graph(
                self.mol_graph, name=str(self), force_sanitize=False
            )
        return self._rdkit_mol

    @rdkit_mol.setter
    def rdkit_mol(self, m):
        self._rdkit_mol = m

    @property
    def fragments(self):
        """
        Get fragments of the molecule by breaking all the bonds.
        Returns:
            A dictionary with bond index (a tuple (idx1, idx2)) as key, and a list
            of the mol_graphs of the fragments as value (each list is of size 1 or 2).
            The dictionary is empty if the mol has no bonds.
        """
        if self._fragments is None:
            bonds = [b for b, _ in self.bonds.items()]
            self._fragments = fragment_mol_graph(self.mol_graph, bonds)
        return self._fragments

    @property
    def isomorphic_bonds(self):
        r"""
        Find isomorphic bonds. Isomorphic bonds are defined as bonds that the same
        fragments (in terms of fragment connectivity) are obtained by breaking the bonds
        separately.
        For example, for molecule
               0     1
            H1---C0---H2
              2 /  \ 3
              O3---O4
                 4
        bond 0 is isomorphic to bond 1, bond 2 is isomorphic to bond 3 , bond 4 is not
        isomorphic to any other bond.
        Note:
            Bond not isomorphic to any other bond is included as a group by itself.
        Returns:
            list of list: each inner list contains the indices (a 2-tuple) of bonds that
                are isomorphic. For the above example, this function
                returns [[(0,1), (0,2)], [(0,3), (0,4)], [(3,4)]]
        """

        if self._isomorphic_bonds is None:

            iso_bonds = []

            for bond1, frags1 in self.fragments.items():
                for group in iso_bonds:

                    # compare to the first in a group to see whether it is isomorphic
                    bond2 = group[0]
                    frags2 = self.fragments[bond2]

                    if len(frags1) == len(frags2) == 1:
                        if frags1[0].isomorphic_to(frags2[0]):
                            group.append(bond1)
                            break
                    elif len(frags1) == len(frags2) == 2:
                        if (
                            frags1[0].isomorphic_to(frags2[0])
                            and frags1[1].isomorphic_to(frags2[1])
                        ) or (
                            frags1[0].isomorphic_to(frags2[1])
                            and frags1[1].isomorphic_to(frags2[0])
                        ):
                            group.append(bond1)
                            break

                # bond1 not in any group
                else:
                    iso_bonds.append([bond1])

            self._isomorphic_bonds = iso_bonds

        return self._isomorphic_bonds

    def is_atom_in_ring(self, atom):
        """
        Whether an atom in ring.
        Args:
            atom (int): atom index
        Returns:
            bool: atom in ring or not
        """
        ring_info = self.mol_graph.find_rings()
        ring_atoms = set([atom for ring in ring_info for bond in ring for atom in bond])
        return atom in ring_atoms

    def is_bond_in_ring(self, bond):
        """
       Whether a bond in ring.
       Args:
           bond (tuple): bond index
       Returns:
           bool: bond in ring or not
        """
        ring_info = self.mol_graph.find_rings()
        ring_bonds = set([tuple(sorted(bond)) for ring in ring_info for bond in ring])
        return tuple(sorted(bond)) in ring_bonds

    def get_sdf_bond_indices(self, zero_based=False, sdf=None):
        """
        Get the indices of bonds as specified in the sdf file.
        zero_based (bool): If True, the atom index will be converted to zero based.
        sdf (str): the sdf string for parsing. If None, it is created from the mol.
        Returns:
            list of tuple: each tuple specifies a bond.
        """
        sdf = sdf or self.write()

        lines = sdf.split("\n")
        start = end = 0
        for i, ln in enumerate(lines):
            if "BEGIN BOND" in ln:
                start = i + 1
            if "END BOND" in ln:
                end = i
                break

        bonds = [
            tuple(sorted([int(i) for i in ln.split()[4:6]])) for ln in lines[start:end]
        ]

        if zero_based:
            bonds = [(b[0] - 1, b[1] - 1) for b in bonds]

        return bonds

    def get_sdf_bond_indices_v2000(self, sdf=None):
        """
        Get the indices of bonds as specified in the sdf file.
        Returns:
            list of tuple: each tuple specifies a bond.
        """
        sdf = sdf or self.write(v3000=False)
        lines = sdf.split("\n")
        split_3 = lines[3].split()
        natoms = int(split_3[0])
        nbonds = int(split_3[1])
        bonds = []
        for line in lines[4 + natoms : 4 + natoms + nbonds]:
            bonds.append(tuple(sorted([int(i) for i in line.split()[:2]])))
        return bonds

    def subgraph_atom_mapping(self, bond):
        """
        Find the atoms in the two subgraphs by breaking a bond in a molecule.
        Returns:
            tuple of list: each list contains the atoms in one subgraph.
        """

        original = copy.deepcopy(self.mol_graph)
        original.break_edge(bond[0], bond[1], allow_reverse=True)

        # A -> B breaking
        if nx.is_weakly_connected(original.graph):
            mapping = list(range(self.num_atoms))
            return mapping, mapping
        # A -> B + C breaking
        else:
            components = nx.weakly_connected_components(original.graph)
            nodes = [original.graph.subgraph(c).nodes for c in components]
            mapping = tuple([sorted(list(n)) for n in nodes])
            if len(mapping) != 2:
                raise Exception("Mol not split into two parts")
            return mapping

    def find_ring(self, by_species=False):
        """
        Find all rings in the molecule.
        Args:
            by_species (bool): If False, the rings will be denoted by atom indices. If
                True, denoted by atom species.
        Returns:
            list of list: each inner list holds the atoms (index or specie) of a ring.
        """
        rings = self.mol_graph.find_rings()

        rings_once_per_atom = []
        for r in rings:
            # the ring is given by the connectivity info. For example, for a 1-2-3 ring,
            # r would be something like [(1,2), (2,3), (3,1)]
            # here we remove the repeated atoms and let each atom appear only once
            atoms = []
            for i in r:
                atoms.extend(i)
            atoms = list(set(atoms))
            if by_species:
                atoms = [self.species[j] for j in atoms]
            rings_once_per_atom.append(atoms)

        return rings_once_per_atom

    def write(self, filename=None, name=None, format="sdf", kekulize=True, v3000=True):
        """Write a molecule to file or as string using rdkit.
        Args:
            filename (str): name of the file to write the output. If None, return the
                output as string.
            name (str): name of a molecule. If `file_format` is sdf, this is the first
                line the molecule block in the sdf.
            format (str): format of the molecule, supporting: sdf, pdb, and smi.
            kekulize (bool): whether to kekulize the mol if format is `sdf`
            v3000 (bool): whether to force v3000 form if format is `sdf`
        """
        if filename is not None:
            create_directory(filename)
            filename = str(to_path(filename))

        name = str(self.id) if name is None else name
        self.rdkit_mol.SetProp("_Name", name)

        if format == "sdf":
            if filename is None:
                sdf = Chem.MolToMolBlock(
                    self.rdkit_mol, kekulize=kekulize, forceV3000=v3000
                )
                return sdf + "$$$$\n"
            else:
                return Chem.MolToMolFile(
                    self.rdkit_mol, filename, kekulize=kekulize, forceV3000=v3000
                )
        elif format == "pdb":
            if filename is None:
                sdf = Chem.MolToPDBBlock(self.rdkit_mol)
                return sdf + "$$$$\n"
            else:
                return Chem.MolToPDBFile(self.rdkit_mol, filename)
        elif format == "smi":
            return Chem.MolToSmiles(self.rdkit_mol)
        else:
            raise ValueError(f"format {format} currently not supported")

    def draw(self, filename=None, show_atom_idx=False):
        """
        Draw the molecule.
        Args:
            filename (str): path to the save the generated image. If `None` the
                molecule is returned and can be viewed in Jupyter notebook.
        """
        m = copy.deepcopy(self.rdkit_mol)
        AllChem.Compute2DCoords(m)

        if show_atom_idx:
            for a in m.GetAtoms():
                a.SetAtomMapNum(a.GetIdx() + 1)
        # d.drawOptions().addAtomIndices = True

        if filename is None:
            return m
        else:
            create_directory(filename)
            filename = str(to_path(filename))
            Draw.MolToFile(m, filename)

    def draw_with_bond_note(self, bond_note, filename="mol.png", show_atom_idx=True):
        """
        Draw molecule using rdkit and show bond annotation, e.g. bond energy.
        Args:
            bond_note (dict): {bond_index: note}. The note to show for the
                corresponding bond.
            filename (str): path to the save the generated image. If `None` the
                molecule is returned and can be viewed in Jupyter notebook.
        """
        m = self.draw(show_atom_idx=show_atom_idx)

        # set bond annotation
        highlight_bonds = []
        for bond, note in bond_note.items():
            if isinstance(note, (float, np.floating)):
                note = "{:.3g}".format(note)
            idx = m.GetBondBetweenAtoms(*bond).GetIdx()
            m.GetBondWithIdx(idx).SetProp("bondNote", note)
            highlight_bonds.append(idx)

        # set highlight color
        bond_colors = {b: (192 / 255, 192 / 255, 192 / 255) for b in highlight_bonds}

        d = rdMolDraw2D.MolDraw2DCairo(400, 300)

        # smaller font size
        d.SetFontSize(0.8 * d.FontSize())

        rdMolDraw2D.PrepareAndDrawMolecule(
            d, m, highlightBonds=highlight_bonds, highlightBondColors=bond_colors
        )
        d.FinishDrawing()

        create_directory(filename)
        with open(to_path(filename), "wb") as f:
            f.write(d.GetDrawingText())

    def pack_features(self, broken_bond=None):
        feats = dict()
        feats["charge"] = self.charge
        return feats

    def __expr__(self):
        return f"{self.id}_{self.formula}"

    def __str__(self):
        return self.__expr__()


def create_wrapper_mol_from_atoms_and_bonds(
    species, coords, bonds, charge=0, free_energy=None, identifier=None
):
    """
    Create a :class:`MoleculeWrapper` from atoms and bonds.
    Args:
        species (list of str): atom species str
        coords (2D array): positions of atoms
        bonds (list of tuple): each tuple is a bond (atom indices)
        charge (int): chare of the molecule
        free_energy (float): free energy of the molecule
        identifier (str): (unique) identifier of the molecule
    Returns:
        MoleculeWrapper instance
    """

    pymatgen_mol = pymatgen.Molecule(species, coords, charge)
    bonds = {tuple(sorted(b)): None for b in bonds}
    mol_graph = MoleculeGraph.with_edges(pymatgen_mol, bonds)

    return MoleculeWrapper(mol_graph, free_energy, identifier)


def rdkit_mol_to_wrapper_mol(m, charge=None, free_energy=None, identifier=None):
    """
    Convert an rdkit molecule to a :class:`MoleculeWrapper` molecule.
    This constructs a molecule graph from the rdkit mol and assigns the rdkit mol
    to the molecule wrapper.
    Args:
        m (Chem.Mol): rdkit molecule
        charge (int): charge of the molecule. If None, inferred from the rdkit mol;
            otherwise, the provided charge will override the inferred.
        free_energy (float): free energy of the molecule
        identifier (str): (unique) identifier of the molecule
    Returns:
        MoleculeWrapper instance
    """

    species = [a.GetSymbol() for a in m.GetAtoms()]

    # coords = m.GetConformer().GetPositions()
    # NOTE, the above way to get coords results in segfault on linux, so we use the
    # below workaround
    conformer = m.GetConformer()
    coords = [[x for x in conformer.GetAtomPosition(i)] for i in range(m.GetNumAtoms())]

    bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in m.GetBonds()]
    bonds = {tuple(sorted(b)): None for b in bonds}

    charge = Chem.GetFormalCharge(m) if charge is None else charge

    pymatgen_mol = pymatgen.Molecule(species, coords, charge)
    mol_graph = MoleculeGraph.with_edges(pymatgen_mol, bonds)

    if identifier is None:
        identifier = m.GetProp("_Name")
    mw = MoleculeWrapper(mol_graph, free_energy, identifier)
    mw.rdkit_mol = m

    return mw


class GenerateCoordsError(Exception):
    def __init__(self, msg=None):
        self.msg = msg
        super(GenerateCoordsError, self).__init__(msg)

    def __repr__(self):
        return f"cannot generate 3D coords, {self.msg}"


class RdkitMolCreationError(Exception):
    def __init__(self, msg=None):
        self.msg = msg
        super(RdkitMolCreationError, self).__init__(msg)

    def __repr__(self):
        return f"cannot create rdkit mol, {self.msg}"
