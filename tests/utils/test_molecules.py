from mrnet.utils.molecules import remove_metals

from .utils import create_LiEC_pymatgen_mol


def test_remove_metals():

    mol = create_LiEC_pymatgen_mol()
    mol = remove_metals(mol)
    assert len(mol) == 10
    assert mol.charge == -1