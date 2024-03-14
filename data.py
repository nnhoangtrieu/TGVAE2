import utils 
from utils import * 
import pickle
import multiprocessing
import re 
import torch_geometric 
from torch_geometric.data import Data
import rdkit 

# def parallel_f(f, input_list) :
#     pool = multiprocessing.Pool()
#     return pool.map(f, input_list)


class MyData(Data) : 
    def __cat_dim__(self, key, value, *args, **kwargs) : 
        if key == 'smi' :
            return None 
        return super().__cat_dim__(key, value, *args, **kwargs) 


class ProcessData() : 
    def __init__(self, path, max_len, node_feature_list = None, edge_index_list = None, token_list = None, pharmacophore=False, coor=False)  :
        self.path = path 
        self.max_len = max_len 
        self.node_feature_list = node_feature_list 
        self.edge_index_list = edge_index_list 
        self.token_list = token_list
        self.pharmacophore = pharmacophore
        self.coor = coor
    def extract(self) : 
        if self.path.lower().endswith('.txt') : 
            with open(self.path, 'r') as f : 
                data = [line.strip() for line in f]
                data = [x for x in data if len(x) < self.max_len]
                
                return data  
        elif self.path.lower().endswith('.pickle')  : 
            with open(self.path, 'rb') as f : 
                data = pickle.load(f)
                data = [x for x in data if len(x) < self.max_len]
                return data
        else : 
            raise ValueError('File format currently not supported. Only .txt and .pickle are supported')

    def get_gvocab(self, smi_list) :
        dic = {}
        for smi in smi_list :
            mol = rdkit.Chem.MolFromSmiles(smi)
            for atom in mol.GetAtoms() : 
                symbol = atom.GetSymbol() 
                if symbol not in dic : 
                    dic[symbol] = len(dic) 
        return dic 

    def get_vocab(self, smi_list) :
        dic = {'<START>': 0, '<END>': 1, '<PAD>': 2}
        for smi in smi_list :
            for char in smi :
                if char not in dic :
                    dic[char] = len(dic) 
        return dic 
    
    def tokenizer(self, smile):
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regezz = re.compile(pattern)
        tokens = [token for token in regezz.findall(smile)]
        assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
        return tokens
    
    def pad(self, smi) :
        smi = smi + [2] * (self.max_len - len(smi))
        return torch.tensor(smi, dtype=torch.long)

    def encode(self, smi) :
        return [0] + [self.vocab[char] for char in smi] + [1]
    
    def get_ew(self, smi) : 
        dic = {
            'SINGLE': 0,
            'DOUBLE': 1,
            'TRIPLE': 2,
            'AROMATIC': 3
        }
        mol = rdkit.Chem.MolFromSmiles(smi) 
        return torch.tensor([dic[str(bond.GetBondType())] for bond in mol.GetBonds()] * 2)
    


    def get_coor(self, smi) : 
        mol = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol) 
        AllChem.UFFOptimizeMolecule(mol) 
        coor = []
        for i, atom in enumerate(mol.GetAtoms()) : 
            if atom.GetSymbol() != 'H' :
                pos = mol.GetConformer().GetAtomPosition(i) 
                coor.append([round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)])
        return torch.tensor(coor)


    def get_ei(self, smi) : 
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

    def get_nf(self, smi) : 
        mol = rdkit.Chem.MolFromSmiles(smi) 
        atom_list = [self.gvocab[atom.GetSymbol()] for atom in mol.GetAtoms()]
        return torch.tensor(atom_list)

    def process(self) : 
        self.smi_list = self.extract() 
        token_list = [self.tokenizer(smi) for smi in self.smi_list]

        self.gvocab = self.get_gvocab(self.smi_list)
        self.vocab = self.get_vocab(token_list)
        self.inv_vocab = {v:k for k,v in self.vocab.items()}
        token_list = [self.encode(smi) for smi in token_list]
        self.max_len = len(max(token_list, key=len))
        token_list = [self.pad(t) for t in token_list]
        node_feature_list = [self.get_nf(smi) for smi in self.smi_list]
        edge_index_list = [self.get_ei(smi) for smi in self.smi_list]
        edge_weight_list = [self.get_ew(smi) for smi in self.smi_list]


        # if self.pharmacophore : 
        #     data_list = [MyData(x=torch.cat((node_feature_list[i].unsqueeze(-1), node_pharmacophore_list[i]), dim=-1),
        #                         edge_index=edge_index_list[i],
        #                         edge_attr=edge_weight_list[i],
        #                         smi=token_list[i]) for i in range(len(self.smi_list)) if node_feature_list[i].size(0) == node_pharmacophore_list[i].size(0)]
            
        # elif self.coor : 
        #     data_list = [MyData(x=node_feature_list[i],
        #                         edge_index=edge_index_list[i],
        #                         edge_attr=edge_weight_list[i],
        #                         coor=coor_list[i],
        #                         smi=token_list[i]) for i in range(len(self.smi_list))]
        # else :
        data_list = [MyData(x=node_feature_list[i],
                            edge_index=edge_index_list[i],
                            edge_attr=edge_weight_list[i],
                            smi=token_list[i]) for i in range(len(self.smi_list))]


        



        return data_list