import os
import argparse
from tqdm import tqdm
import torch 
import torch_geometric 
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader as gDataLoader
from model.base import Transformer
from helper.data import ProcessData
from helper.utils import monotonic_annealer, get_mask, seed_torch, cyclic_annealer

def loss_fn(pred, tgt, mu, sigma, beta) :
    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(vocab)), tgt.reshape(-1), ignore_index=vocab['<PAD>'])
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()).mean() / arg.batch
    return  reconstruction_loss + kl_loss * beta, reconstruction_loss, kl_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch()
parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=44)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_latent', type=int, default=512)
parser.add_argument('--d_ff', type=int, default=1024)
parser.add_argument('--e_heads', type=int, default=1)
parser.add_argument('--d_heads', type=int, default=16)
parser.add_argument('--n_layers', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--wd', type=float, default=1e-6)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--kl_type', type=str, default="monotonic")
parser.add_argument('--kl_w_start', type=float, default=0.00005)
parser.add_argument('--kl_w_end', type=float, default=0.0001)
parser.add_argument('--kl_cycle', type=int, default=4)
parser.add_argument('--kl_ratio', type=float, default=1.0)
parser.add_argument('--save_name', type=str, default='savename')
arg = parser.parse_args()

print('\nArguments:')
print(vars(arg))


if not os.path.exists(f'checkpoint/{arg.save_name}') : 
    os.makedirs(f'checkpoint/{arg.save_name}')
if not os.path.exists(f'genmol/{arg.save_name}') :
    os.makedirs(f'genmol/{arg.save_name}')
Data = ProcessData('data/train.txt', arg.max_len)
data_list, smi_list, vocab, inv_vocab, gvocab, max_len = (Data.process(),
                                                         Data.smi_list,
                                                         Data.vocab,
                                                         Data.inv_vocab,
                                                         Data.gvocab,
                                                         Data.max_len)
config = vars(arg)
config['vocab'] = vocab 
config['gvocab'] = gvocab
config['max_token_len'] = max_len 
torch.save(vars(arg), f'checkpoint/{arg.save_name}/config.pt')

print(f'\nNumber of data: {len(data_list)}')
train_loader = gDataLoader(data_list, batch_size=arg.batch, shuffle=True)  





model = Transformer(d_model=arg.d_model,
                    d_latent=arg.d_latent,
                    d_ff=arg.d_ff,
                    e_heads=arg.e_heads,
                    d_heads=arg.d_heads,
                    num_layer=arg.n_layers,
                    dropout=arg.dropout,
                    vocab=vocab,
                    gvocab=gvocab).to(device)

optim = torch.optim.Adam(model.parameters(),
                         lr=arg.lr,
                         weight_decay=arg.wd)


if arg.kl_type == 'monotonic' : 
    annealer = monotonic_annealer(n_epoch=arg.n_epochs,
                                  kl_start=0,
                                  kl_w_start=arg.kl_w_start,
                                  kl_w_end=arg.kl_w_end)
elif arg.kl_type == 'cyclic' :
    annealer = cyclic_annealer(start=arg.kl_w_start,
                               stop=arg.kl_w_end,
                               n_epoch=arg.n_epochs,
                               n_cycle=arg.kl_cycle,
                               ratio=arg.kl_ratio)




print('\n\n\n')
print('#########################################################################')
print('######################### START TRAINING ################################')
print('#########################################################################')
print('\n\n\n')





for epoch in range(1, arg.n_epochs + 1) :
    print(f'Starting Epoch {epoch}...')
    train_loss, val_loss, recon_loss, kl_loss = 0, 0, 0, 0
    beta = annealer[epoch-1]

    model.train()
    for src in tqdm(train_loader, desc=f'Epoch {epoch}') :
        src = src.to(device)
        tgt = src.clone().smi.to(device)
        tgt_mask = get_mask(tgt[:, :-1], vocab) 
        pred, mu, sigma = model(src, tgt[:, :-1], None, tgt_mask)
        loss, recon, kl = loss_fn(pred, tgt[:, 1:], mu, sigma, beta)
        loss.backward(), optim.step(), optim.zero_grad(), clip_grad_norm_(model.parameters(), 5)
    print(f'Finished Training Epoch {epoch}...')


    if (epoch < 50) or (epoch >= 50 and epoch % 5 == 0) :
        snapshot = {
            "MODEL_STATE": model.state_dict(),
            "OPTIMIZER_STATE": optim.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, f'checkpoint/{arg.save_name}/snapshot_{epoch}.pt')
        print(f"Training snapshot saved at checkpoint/{arg.save_name}/snapshot_{epoch}.pt")
    
    else : 
        print('Skip saving snapshot....')


    

    







