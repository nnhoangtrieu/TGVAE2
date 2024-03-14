import rdkit
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as get_mol
from pathlib import Path
import re 
import torch
from rdkit.Chem import rdDistGeom
import multiprocessing
from tqdm import tqdm


test_smi = 'CCCC'

def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)

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
    print(smi)
    return torch.tensor(token)
        





import pickle


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



# with open('data/train.txt', 'r') as f :
#     smi_list = [s.strip() for s in f.readlines()]



# coor_list = []

# for i in tqdm(range(0, 1000)) : 
#     coor = get_coor(smi_list[i])
#     coor_list.append(coor)

# with open('data/coor/data0_1k.pkl', 'wb') as f : 
#     pickle.dump(coor_list, f)


with open('data/coor/data0_1k.pkl', 'rb') as f :
    coor_list = pickle.load(f)

    print(coor_list)