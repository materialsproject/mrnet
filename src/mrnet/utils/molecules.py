from collections import defaultdict

from pymatgen.core.structure import Molecule


def remove_metals(mol, metals={"Li": 1, "Mg": 2}):
    """
    Check whether metals are in a pymatgen molecule. If yes, create a new Molecule
    with metals removed.
    Args:
        mol (pymatgen mol): molecule
        metals (dict): with metal specie are key and charge as value
    Returns:
        pymatgen mol
    """
    species = [str(s) for s in mol.species]

    if set(species).intersection(set(metals.keys())):
        charge = mol.charge

        species = []
        coords = []
        properties = defaultdict(list)
        for site in mol:
            s = str(site.specie)
            if s in metals:
                charge -= metals[s]
            else:
                species.append(s)
                coords.append(site.coords)
                for k, v in site.properties:
                    properties[k].append(v)

        # do not provide spin_multiplicity, since we remove an atom
        mol = Molecule(species, coords, charge, site_properties=properties)

    return mol
