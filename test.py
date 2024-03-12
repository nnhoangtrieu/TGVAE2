import rdkit
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as get_mol
from pathlib import Path
import re 
import torch
from rdkit.Chem import rdDistGeom


test_smi = 'CCCC'



def get_ei(smi) : 
    mol = rdkit.Chem.MolFromSmiles(smi) 
    ei = []
    for bond in mol.GetBonds() :
        b = bond.GetBeginAtomIdx() 
        e = bond.GetEndAtomIdx() 
        ei.append([b,e])
    for bond in mol.GetBonds() :
        b = bond.GetBeginAtomIdx() 
        e = bond.GetEndAtomIdx() 
        ei.append([e, b])
    return torch.tensor(ei).T

def tokenizer(smile):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

def get_pharmacophore(smi) : 
    "feature: Acceptor, Donor, Hydrophobe, Aromatic"
    feature_dic = {
        'Donor': 'A',
        'Acceptor': 'B',
        'NegIonizable': 'C',
        'PosIonizable': 'D',
        'ZnBinder': 'E',
        'Aromatic': 'F',
        'Hydrophobe': 'G',
        'LumpedHydrophobe': 'H'
    }

    feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    mol = rdkit.Chem.MolFromSmiles(smi) 
    rdkit.Chem.SanitizeMol(mol)
    mol_h = rdkit.Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h) 
    
    res = []
    idx = [] 

    for key, value in feature_dic.items() : 
        features = feature_factory.GetFeaturesForMol(mol_h, includeOnly=key)
        out = [f.GetAtomIds() for f in features]
        temp = []
        for o in out : 
            for i in o : 
                temp.append(i)
                idx.append(i)
        temp = (temp, value)
        res.append(temp)
    return res, set(idx)

feat, idx = get_pharmacophore(test_smi)















def get_pf(smi) : 
    token = tokenizer(smi)
    feature_option = ['Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe']
    feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    mol = rdkit.Chem.MolFromSmiles(smi) 
    rdkit.Chem.SanitizeMol(mol)
    mol_h = rdkit.Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h) 
    
    features = []
    res = [] 

    for option in feature_option :
        factory = feature_factory.GetFeaturesForMol(mol_h, includeOnly=option)

        feat = [f.GetAtomIds() for f in factory]
        coor = [f.GetPos() for f in factory]
        temp = []
        for f in feat : 
            for i in f : 
                temp.append(i)
        
        features.append(temp)

    count = 0 
    for t in token : 
        temp = []
        if t.isalpha() : 
            for f in features : 
                if count not in f : 
                    temp.append(0.0) 
                else :
                    temp.append(1.0) 
            count += 1
            res.append(temp)
    return torch.tensor(res)




def get_pnf(smi) : 
    
    feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    feature_opt = list(feature_factory.GetFeatureDefs().keys())
    feature_opt = [f.split('.')[1] for f in feature_opt]
    feature_dic = {f:i for i, f in enumerate(feature_opt)}
    node_f = [0] * len(list(feature_factory.GetFeatureDefs().keys()))
    mol = rdkit.Chem.MolFromSmiles(smi) 
    rdkit.Chem.SanitizeMol(mol)
    mol_h = rdkit.Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h) 

    factory = feature_factory.GetFeaturesForMol(mol_h, confId=-1)

    

    feat = [[f.GetAtomIds(), f.GetType()] for f in factory]
    token = [node_f] * mol.GetNumAtoms()
    for idx, f in feat : 
        for i in idx : 
            token[i][feature_dic[f]] = 1.0
    return torch.tensor(token)
        


# pnf = get_pnf('CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1')


# print(pnf, pnf.shape)




# mol = rdkit.Chem.MolFromSmiles('CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1')

# for atom in mol.GetAtoms() : 
#     print(atom.GetSymbol())





def get_coor(smi) : 
    mol = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol) 
    AllChem.UFFOptimizeMolecule(mol) 
    coor = []
    for i, atom in enumerate(mol.GetAtoms()) : 
        if atom.GetSymbol() != 'H' :
            pos = mol.GetConformer().GetAtomPosition(i) 
            coor.append([round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)])
    return torch.tensor(coor)





mol = rdkit.Chem.MolFromSmiles('CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1')

for atom in mol.GetAtoms() : 
    print(atom.GetFormalCharge())


# from data import ProcessData
# from torch_geometric.loader import DataLoader as gDataLoader

# Data = ProcessData('data/train.txt', 22,coor=True)

# data_list = Data.process()

# train_loader = gDataLoader(data_list, batch_size=16, shuffle=True)
# for i in train_loader : 
#     print(i)