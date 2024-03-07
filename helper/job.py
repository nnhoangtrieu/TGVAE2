import re

scripts = [

# 'train.py --max_len 44 --n_layers 4',
# 'train.py --max_len 44 --n_layers 6',
# 'train.py --max_len 44 --n_layers 8',
# 'train.py --max_len 44 --n_layers 9',
# 'train.py --max_len 44 --n_layers 10',



'train.py --max_len 100 --n_layers 8'
# 'train.py --max_len 100 --n_layers 6',
# 'train.py --max_len 100 --n_layers 8',
# 'train.py --max_len 100 --n_layers 9',
# 'train.py --max_len 100 --n_layers 10'
]


check = []

for i, script in enumerate(scripts) :
    split = script.split('--')[1:]
    split = [s.replace(' ','').replace('"','').replace("'",'') for s in split]
    save_name = '|'.join(split)
    if save_name in check : 
        print('Name existed')
        print(save_name)
        exit()
    else :
        check.append(save_name)

    with open(f'./job{i}.sub', 'w') as f : 
        f.write(f'''#!/bin/bash
                
#SBATCH --job-name=trieu-nguyen
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --output=job-out/single-gpu/train%j.out
                
ml Python 
python {script} --save_name "v2{save_name}"
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
# python get-metrics.py --save_name "{save_name}"
#  ''')




# # for file in /home/80027464/graphvae/job*; do [ -f "$file" ] && sbatch "$file"; done
# # for file in /home/80027464/graphvae/moses*; do [ -f "$file" ] && sbatch "$file"; done

