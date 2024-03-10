import rdkit
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as get_mol
from pathlib import Path
import re 
import torch


test_smi = 'COc1cc2c(cc1OC)C(c1c(-c3ccccc3)noc1C)=NCC2'



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









def mask_smi(smi, idx) : 
    print(smi)
    smi = tokenizer(smi)
    count = 0 
    res = ""
    for s in smi : 
        if s.isalpha() : 
            if count not in idx : 
                res += "M"
            else : 
                res += s 
            count += 1

        else : 
            res += s 


    return res







def get_pf(smi) : 
    token = tokenizer(smi)
    print(f'token: {token}')
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
    feature_option = ['Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe']
    feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    mol = rdkit.Chem.MolFromSmiles(smi) 
    rdkit.Chem.SanitizeMol(mol)
    mol_h = rdkit.Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h) 
    
    features = []
    res = [] 

    for option in feature_option :
    # for key, value in feature_dic.items() : 
        factory = feature_factory.GetFeaturesForMol(mol_h, includeOnly=option)

        feat = [f.GetAtomIds() for f in factory]

        temp = []
        for f in feat : 
            for i in f : 
                temp.append(i)
        
        features.append(temp)

    print(f'features: {features}')


    count = 0 
    for t in token : 
        temp = []
        for f in features : 
            if count not in f : 
                temp.append(0.0) 
            else :
                temp.append(1.0) 
        count += 1
        res.append(temp)
    return torch.tensor(res)


pf = get_pf(test_smi)