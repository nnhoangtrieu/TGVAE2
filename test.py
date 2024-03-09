# import rdkit
# from rdkit import RDConfig
# from rdkit.Chem import AllChem
# from rdkit.Chem import MolFromSmiles as get_mol
# from pathlib import Path



# def get_pharmacophore(smi, feature) : 
#     "feature: Acceptor, Donor, Hydrophobe, Aromatic"


#     feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    
#     mol = rdkit.Chem.MolFromSmiles(smi) 
#     rdkit.Chem.SanitizeMol(mol)
#     mol_h = rdkit.Chem.AddHs(mol)
#     AllChem.EmbedMolecule(mol_h) 
    
#     features = feature_factory.GetFeaturesForMol(mol_h)
#     coor = [(f.GetAtomIds(), f.GetType()) for f in features]
#     print(coor)
#     # return features.GetAtomMatch()

# get_pharmacophore("C=CCN1C(=O)C(O)(CC(=O)c2cccc(N)c2)c2ccccc21", feature="Acceptor")


from data import ProcessData

Data = ProcessData('data/train.txt', 22)
data_list = Data.process()

print(data_list[0])