import os 
import torch 
import argparse
from model.base import Transformer as TransformerBaseComplete
from model.bond_s import Transformer as TransformerSimpleBond
from model.bond_l import Transformer as TransformerLearnableBond
from model.GAT import Transformer as TransformerGATEmbedding
from model.GCN import Transformer as TransformerGCNEmbedding
from model.GATb import Transformer as TransformerGATEmbedding_LearnableBond
from model.GCNb import Transformer as TransformerGCNEmbedding_LearnableBond
from model.test import Transformer as TransformerTest
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from metric.metrics import get_all_metrics
from tqdm import tqdm
import warnings
from prettytable import PrettyTable 
warnings.filterwarnings("ignore")
table = PrettyTable(["Epoch", "Valid", "Unique@10k", "Novelty", "IntDiv1", "IntDiv2", "Scaf"]) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_gen_smi(t) : 
    smiles = ''.join([inv_vocab[i] for i in t])
    smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")
    return smiles 
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
def get_mask( target, smi_dic) :
        mask = (target != smi_dic['<PAD>']).unsqueeze(-2)
        return mask & subsequent_mask(target.size(-1)).type_as(mask.data)
def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)

parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default="None")
parser.add_argument('--start', type=int, default=20)
arg = parser.parse_args()


if not os.path.exists(f'checkpoint/{arg.save_name}') and arg.save_name != 'None' : 
    print('Path not exists')
    exit()

if not os.path.exists(f'genmol/{arg.save_name}') :
    os.makedirs(f'genmol/{arg.save_name}')

config = torch.load(f'checkpoint/{arg.save_name}/config.pt') if arg.save_name != 'None' else torch.load(f'{arg.save_path}/config.pt')
inv_vocab = {v: k for k, v in config['vocab'].items()}





if arg.save_name[:6] == 'base_c': 
    model = TransformerBaseComplete(
        d_model=config['d_model'],
        d_latent=config['d_latent'],
        d_ff=config['d_ff'],
        e_heads=config['e_heads'],
        d_heads=config['d_heads'],
        num_layer=config['n_layers'],
        dropout=config['dropout'],
        vocab=config['vocab'],
        gvocab=config['gvocab']
    ).to(device)
    print('Model: TransformerBaseComplete')

elif arg.save_name[:6] == 'bond_s' :
    model = TransformerSimpleBond(
        d_model=config['d_model'],
        d_latent=config['d_latent'],
        d_ff=config['d_ff'],
        e_heads=config['e_heads'],
        d_heads=config['d_heads'],
        num_layer=config['n_layers'],
        dropout=config['dropout'],
        vocab=config['vocab'],
        gvocab=config['gvocab']
    ).to(device)
    print('Model: TransformerSimpleBond')

elif arg.save_name[:6] == 'bond_l' :
    model = TransformerLearnableBond(
        d_model=config['d_model'],
        d_latent=config['d_latent'],
        d_ff=config['d_ff'],
        e_heads=config['e_heads'],
        d_heads=config['d_heads'],
        num_layer=config['n_layers'],
        dropout=config['dropout'],
        vocab=config['vocab'],
        gvocab=config['gvocab']
    ).to(device)
    print('Model: TransformerLearnableBond')

elif arg.save_name[:6] == 'ge_gat' :
    model = TransformerGATEmbedding(
        d_model=config['d_model'],
        d_latent=config['d_latent'],
        d_ff=config['d_ff'],
        e_heads=config['e_heads'],
        d_heads=config['d_heads'],
        num_layer=config['n_layers'],
        dropout=config['dropout'],
        vocab=config['vocab'],
        gvocab=config['gvocab']
    ).to(device)
    print('Model: TransformerGATEmbedding')

elif arg.save_name[:6] == 'ge_gcn' :
    model = TransformerGCNEmbedding(
        d_model=config['d_model'],
        d_latent=config['d_latent'],
        d_ff=config['d_ff'],
        e_heads=config['e_heads'],
        d_heads=config['d_heads'],
        num_layer=config['n_layers'],
        dropout=config['dropout'],
        vocab=config['vocab'],
        gvocab=config['gvocab']
    ).to(device)
    print('Model: TransformerGCNEmbedding')

elif arg.save_name[:4] == 'GATb' or arg.save_name[:4] == 'BASE' :
    model = TransformerGATEmbedding_LearnableBond(
        d_model=config['d_model'],
        d_latent=config['d_latent'],
        d_ff=config['d_ff'],
        e_heads=config['e_heads'],
        d_heads=config['d_heads'],
        num_layer=config['n_layers'],
        dropout=config['dropout'],
        vocab=config['vocab'],
        gvocab=config['gvocab']
    ).to(device)
    print('Model: TransformerGATEmbedding_LearnableBond')

elif arg.save_name[:13] == 'ge_bond_l_gcn' :
    model = TransformerGCNEmbedding_LearnableBond(
        d_model=config['d_model'],
        d_latent=config['d_latent'],
        d_ff=config['d_ff'],
        e_heads=config['e_heads'],
        d_heads=config['d_heads'],
        num_layer=config['n_layers'],
        dropout=config['dropout'],
        vocab=config['vocab'],
        gvocab=config['gvocab']
    ).to(device)
    print('Model: TransformerGCNEmbedding_LearnableBond')


writer = SummaryWriter(f'tensorboard/{arg.save_name}')

model.eval()
for epoch in range(arg.start, config['n_epochs'] + 1) :

    try :
        model.load_state_dict(torch.load(f'checkpoint/{arg.save_name}/snapshot_{epoch}.pt')['MODEL_STATE'])
        print(f'Loaded snapshot_{epoch}.pt')
    except :
        print("Snapshot not found")
        continue


    gen_mol = torch.empty(0).to(device)
    with torch.no_grad() : 
        print(f'Epoch {epoch} generating molecules...')
        for _ in range(10) :
            z = torch.randn(3000, config['d_latent']).to(device)
            tgt = torch.zeros(3000, 1, dtype=torch.long).to(device)

            for _ in range(config['max_token_len']-1) : 
                pred = model.inference(z, tgt, None, get_mask(tgt, config['vocab']).to(device))
                _, idx = torch.topk(pred, 1, dim=-1)
                idx = idx[:, -1, :]
                tgt = torch.cat([tgt, idx], dim=1)

            gen_mol = torch.cat([gen_mol, tgt], dim=0)
            torch.cuda.empty_cache()
        gen_mol = gen_mol.tolist() 

        gen_mol = parallel_f(read_gen_smi, gen_mol)
        result = get_all_metrics(gen_mol, pool=4)
        table.add_row([epoch,
                       round(result['valid'],4),
                       round(result['unique@10000'],4),
                       round(result['Novelty'],4), 
                       round(result['IntDiv'],4), 
                       round(result['IntDiv2'],4), 
                       round(result['Scaf/TestSF'],4)])
        print(table)
        for name, value in result.items() : 
            writer.add_scalar(name, value, epoch)
            # print(f'\t{name}: {value:.4f}')

        # with open(f'genmol/{arg.save_name}/e_{epoch}', 'w') as f : 
        #     for i, mol in enumerate(gen_mol[:1000]) : 
        #         f.write(f'{i+1}. {mol}\n')
        #     f.write(f'Epoch {epoch}:\n{result}')


