import re

scripts = [
'train.py --n_layers 9',
'train.py --n_layers 8 --kl "M" --kl_w_start 0.00005 --kl_w_end 0.000075',
'train.py --n_layers 8 --kl "M" --kl_w_start 0.000075 --kl_w_end 0.000125',
'train.py --n_layers 8 --kl "M" --kl_w_start 0.0001 --kl_w_end 0.00015',
'train.py --n_layers 8 --kl "M" --kl_w_start 0.0001 --kl_w_end 0.0003',

'train.py --n_layers 8 --kl "C" --kl_cycle 4 --kl_w_start 0.00005 --kl_w_end 0.000075',
'train.py --n_layers 8 --kl "C" --kl_cycle 6',
'train.py --n_layers 8 --kl "C" --kl_cycle 8'

]


# model = ['base_o', 'base_c', 'bond_s', 'bond_l', 'ge_bond_l_gat', 'ge_bond_l_gcn', 'ge_gat', 'ge_gcn']
model = ['ge_bond_l_gat']

check = []

for i, script in enumerate(scripts) :
    for j, m in enumerate(model) :
        split = script.split('--')[1:]
        split = [s.replace(' ','').replace('"','').replace("'",'') for s in split]
        save_name = '|'.join(split)
        if save_name in check : 
            print('Name existed')
            print(save_name)
            # exit()
        else :
            check.append(save_name)

        
        with open(f'./jobsubmission{i}{j}.sub', 'w') as f : 
            f.write(f'''#!/bin/bash
                    
#SBATCH --job-name=trieu-nguyen
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --output=blabla%j.out
                
ml Python 
python {script} --save_name "{m}|||{save_name}"
''')
    






# import os 

# for i, save_name in enumerate(os.listdir('./checkpoint')) : 
#     with open(f'moses{i}.sub', 'w') as f : 
#         f.write(f'''#!/bin/bash
                
# #SBATCH --job-name=trieu-nguyen
# #SBATCH --cpus-per-task=4
# #SBATCH --gres=gpu:a30:1
# #SBATCH --output=moses%j.out

# ml Python 
# python all-metric.py --save_name "{save_name}"
#  ''')




# # for file in /home/80027464/TGVAE/jobsubmission*; do [ -f "$file" ] && sbatch "$file"; done
# # for file in /home/80027464/TGVAE/moses*; do [ -f "$file" ] && sbatch "$file"; done

