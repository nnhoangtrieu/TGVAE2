from data import ProcessData
import torch 
import torch.nn as nn 
from torch_geometric.loader import DataLoader as gDataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from model.GATb import Transformer, PositionalEncoding, Embeddings
from utils import monotonic_annealer, get_mask, seed_torch, cyclic_annealer
import argparse 
from tqdm import tqdm 
import os  


parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--batch', type=int, default=8)
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
parser.add_argument('--kl', type=str, default="M")
parser.add_argument('--kl_w_start', type=float, default=0.00005)
parser.add_argument('--kl_w_end', type=float, default=0.0001)
parser.add_argument('--kl_cycle', type=int, default=4)
parser.add_argument('--kl_ratio', type=float, default=1.0)
parser.add_argument('--pharmacophore', type=bool, default=False)
parser.add_argument('--coor', type=bool, default=False)
parser.add_argument('--save_name', type=str, default='ge_bond_l_gat')
arg = parser.parse_args()

os.makedirs(f'checkpoint/finetune_{arg.save_name}', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def loss_fn(pred, tgt, mu, sigma, beta) :
    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(vocab)), tgt.reshape(-1), ignore_index=vocab['<PAD>'])
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()).mean() / arg.batch
    return  reconstruction_loss + kl_loss * beta, reconstruction_loss, kl_loss

Data = ProcessData('data/adagrasib6k_train.txt', 100)
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
torch.save(vars(arg), f'checkpoint/finetune_{arg.save_name}/config.pt')
train_loader = gDataLoader(data_list, batch_size=arg.batch, shuffle=True)  


old_config = torch.load('checkpoint/ge_bond_l_gat|||n_layers8/config.pt')
embedding = Embeddings(arg.d_model, len(vocab))
pe = PositionalEncoding(arg.d_model, arg.dropout)
tgt_embedding = nn.Sequential(embedding, pe).to(device)
generator = nn.Linear(arg.d_model, len(vocab)).to(device)


annealer = monotonic_annealer(n_epoch=arg.n_epochs,
                                kl_start=0,
                                kl_w_start=0.00005,
                                kl_w_end=0.0001)

model = Transformer(d_model=arg.d_model,
                        d_latent=arg.d_latent,
                        d_ff=arg.d_ff,
                        e_heads=arg.e_heads,
                        d_heads=arg.d_heads,
                        num_layer=arg.n_layers,
                        dropout=arg.dropout,
                        vocab=old_config['vocab'],
                        gvocab=old_config['gvocab']).to(device)





model.load_state_dict(torch.load('checkpoint/ge_bond_l_gat|||n_layers8/snapshot_80.pt')['MODEL_STATE'])
model.tgt_embedding = tgt_embedding 
model.generator = generator




for param in model.encoder.layers[:6].parameters() : 
    param.requires_grad = False
for param in model.decoder.layers[:6].parameters() :
    param.requires_grad = False



optim = torch.optim.Adam(model.parameters(),
                         lr=5e-4,
                         weight_decay=1e-6)















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


    snapshot = {
        "MODEL_STATE": model.state_dict(),
        "OPTIMIZER_STATE": optim.state_dict(),
        "EPOCHS_RUN": epoch,
    }
    torch.save(snapshot, f'checkpoint/finetune_{arg.save_name}/snapshot_{epoch}.pt')
    print(f"Training snapshot saved at checkpoint/finetune_{arg.save_name}/snapshot_{epoch}.pt")
    
