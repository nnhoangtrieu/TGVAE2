import os 
import torch 
import argparse 
from model.base import Transformer as TransformerBase
from model.base_complete import Transformer as TransformerBaseComplete
from model.bond import Transformer as TransformerBond
import multiprocessing 
from tqdm import tqdm
import datetime 
import warnings
import rdkit
from metric.metrics import get_all_metrics
from metric.plot import get_plot
from helper.table import table1, table2, table3, table4
warnings.filterwarnings("ignore")

current = datetime.datetime.now()
current = current.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default='None')
parser.add_argument('--num_gen', type=int, default=30000)
parser.add_argument('--get_metric', type=bool, default=True)
parser.add_argument('--num_cpu', type=int, default=4)

arg = parser.parse_args()

if not os.path.exists(f'checkpoint/{arg.save_name}') and arg.save_name != 'None' :
    print('Path not exists')
    print('Make sure there is a folder with the name you provided, which inside has "config.pt" and "model.pt"')
    exit()


print('Model found, loading model...\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Currently using {device}\n')

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



config = torch.load(f'checkpoint/{arg.save_name}/config.pt') 
inv_vocab = {v: k for k, v in config['vocab'].items()}

if arg.save_name[0] == 'b' : 
    model = TransformerBond(
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
elif arg.save_name[0] == 'c' : 
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
else :
    model = TransformerBase(
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

try :
    model.load_state_dict(torch.load(f'checkpoint/{arg.save_name}/model.pt')['MODEL_STATE'])
    print('Model loaded successfully')
except : 
    print('Model not found')
    exit()


rdkit.rdBase.DisableLog('rdApp.*')
model.eval() 
gen_mol = torch.empty(0).to(device)
with torch.no_grad() :
    for _ in tqdm(range(arg.num_gen // 1000), desc='Generating molecules') :
        z = torch.randn(arg.num_gen // (arg.num_gen // 1000), config['d_latent']).to(device)
        tgt = torch.zeros(arg.num_gen // (arg.num_gen // 1000), 1, dtype=torch.long).to(device)

        for _ in range(config['max_token_len']-1) : 
            pred = model.inference(z, tgt, None, get_mask(tgt, config['vocab']).to(device))
            _, idx = torch.topk(pred, 1, dim=-1)
            idx = idx[:, -1, :]
            tgt = torch.cat([tgt, idx], dim=1)

        gen_mol = torch.cat([gen_mol, tgt], dim=0)
        torch.cuda.empty_cache()
    gen_mol = gen_mol.tolist() 
    gen_mol = parallel_f(read_gen_smi, gen_mol)

    # print('Generated Molecules: ')
    # for i, mol in enumerate(gen_mol) : 
    #     print(f'{i+1}. {mol}')

    os.makedirs(f'output/{current}', exist_ok=True)


    with open(f'output/{current}/molecules.txt', 'w') as f :
        for i, mol in enumerate(gen_mol) : 
            f.write(f'{mol}\n')

    if arg.get_metric == True :
        print('Calculating metrics...')
        result = get_all_metrics(gen_mol, k=(10000, 20000, 25000, 30000), pool=arg.num_cpu)
        get_plot(gen_mol, f'output/{current}')
    else :
        print('Skip calculating metrics...')


    for name, value in result.items() : 
        print(f'\t{name}: {value:.4f}')


    table1.add_row(["TGVAE", round(result['valid'], 4), round(result['unique@10000'],4), round(result['Novelty'],4)])
    table2.add_row(["TGVAE", round(result['Filters'], 4), round(result['IntDiv'], 4), round(result['IntDiv2'], 4)])
    table3.add_row(["TGVAE", round(result['FCD/Test'], 4), round(result['FCD/TestSF'], 4), round(result['SNN/Test'], 4), round(result['SNN/TestSF'], 4)])
    table4.add_row(["TGVAE", round(result['Frag/Test'], 4), round(result['Frag/TestSF'], 4), round(result['Scaf/Test'], 4), round(result['Scaf/TestSF'], 4)])

    with open(f'output/{current}/chart.txt', 'w') as f : 
        f.write(table1.get_string() + '\n\n')
        f.write(table2.get_string() + '\n\n')
        f.write(table3.get_string() + '\n\n')
        f.write(table4.get_string() + '\n\n')
    print(table1)
    print(table2)
    print(table3)
    print(table4)




