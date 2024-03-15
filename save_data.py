import rdkit 
import rdkit.Chem
import torch 
import re 
from torch_geometric.data import Data

class MyData(Data) : 
    def __cat_dim__(self, key, value, *args, **kwargs) : 
        if key == 'token' :
            return None 
        return super().__cat_dim__(key, value, *args, **kwargs) 


def tokenizer(smile):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens


def get_graph_data(smi, gvocab, vocab, max_len) : 
    dic = {
        'SINGLE': 0,
        'DOUBLE': 1,
        'TRIPLE': 2,
        'AROMATIC': 3
    }
    mol = rdkit.Chem.MolFromSmiles(smi) 
    node_feat, edge_idx, edge_feat = [], [], []

    for atom in mol.GetAtoms() :
        node_feat.append(gvocab[atom.GetSymbol()])
    
    for bond in mol.GetBonds() :
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() 
        edge_idx += [[b, e], [e, b]]

        edge_feat.append(dic[str(bond.GetBondType())])
        edge_feat.append(dic[str(bond.GetBondType())])
    
    token = tokenizer(smi) 
    token = [0] + [vocab[t] for t in token] + [1] 
    token = token + [2] * (max_len - len(token))


    return MyData(x=torch.tensor(node_feat),
                  edge_index=torch.tensor(edge_idx),
                  edge_attr=torch.tensor(edge_feat),
                  token=torch.tensor(token, dtype=torch.long),
                  smi=smi)




gvocab = torch.load('data/moses_gvocab.pt')
vocab = torch.load('data/moses_vocab.pt')
maxlen = torch.load('data/moses_maxlen.pt')


with open('data/train.txt', 'r') as f : 
    smi_list = [s.strip() for s in ]









































































# def get_gvocab(path) : 
#     gvocab = {}
#     with open(path, 'r') as f : 
#         smi_list = [s.strip() for s in f.readlines()]
#         for smi in smi_list : 
#             print(smi)
#             mol = rdkit.Chem.MolFromSmiles(smi) 
#             for atom in mol.GetAtoms() : 
#                 symbol = atom.GetSymbol()
#                 if symbol not in gvocab : 
#                     gvocab[symbol] = len(gvocab)

#     return gvocab 

# def get_ei(smi) : 
#     edge_idx = []
#     mol = rdkit.Chem.MolFromSmiles(smi)
#     for bond in mol.GetBonds() : 
#         b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() 
#         edge_idx += [[b, e], [e, b]]
#     return torch.tensor(edge_idx).T


# def get_nf(smi, gvocab) : 
#     node_feat = []
#     mol = rdkit.Chem.MolFromSmiles(smi) 

#     for atom in mol.GetAtoms() : 
#         node_feat.append(gvocab[atom.GetSymbol()])
#     return node_feat


# def get_ef(smi) : 
#     edge_feat = [] 
#     mol = rdkit.Chem.MolFromSmiles(smi) 


