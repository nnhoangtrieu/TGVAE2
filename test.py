import rdkit
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as get_mol
from pathlib import Path



def get_pharmacophore(smi, feature) : 
    "feature: Acceptor, Donor, Hydrophobe, Aromatic"

    
    feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    
    mol = rdkit.Chem.MolFromSmiles(smi) 
    rdkit.Chem.SanitizeMol(mol)
    mol_h = rdkit.Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h) 
    
    features = feature_factory.GetFeaturesForMol(mol_h)
    coor = [(f.GetAtomIds(), f.GetType()) for f in features]
    print(coor)
    # return features.GetAtomMatch()

print(len("CCCC"))
get_pharmacophore("CCCC", feature="Acceptor")