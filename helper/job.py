import re

scripts = [

'train.py --n_layers 8 --kl "M" --kl_w_start 0.0002 --kl_w_end 0.00025',
'train.py --n_layers 8 --kl "M" --kl_w_start 0.0003 --kl_w_end 0.00035',
'train.py --n_layers 8 --kl "M" --kl_w_start 0.0004 --kl_w_end 0.00045',
'train.py --n_layers 8 --kl "M" --kl_w_start 0.0005 --kl_w_end 0.00055',
'train.py --n_layers 8 --kl "M" --kl_w_start 0.0006 --kl_w_end 0.00065',



'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 2 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 2 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 2 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 2 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 2 --kl_ratio 0.25',

'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 2 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 2 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 2 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 2 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 2 --kl_ratio 0.5',


'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 2 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 2 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 2 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 2 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 2 --kl_ratio 1.0',




'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 4 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 4 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 4 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 4 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 4 --kl_ratio 0.25',

'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 4 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 4 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 4 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 4 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 4 --kl_ratio 0.5',


'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 4 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 4 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 4 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 4 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 4 --kl_ratio 1.0',


'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 6 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 6 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 6 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 6 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 6 --kl_ratio 0.25',

'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 6 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 6 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 6 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 6 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 6 --kl_ratio 0.5',


'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 6 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 6 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 6 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 6 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 6 --kl_ratio 1.0',



'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 8 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 8 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 8 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 8 --kl_ratio 0.25',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 8 --kl_ratio 0.25',

'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 8 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 8 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 8 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 8 --kl_ratio 0.5',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 8 --kl_ratio 0.5',


'train.py --n_layers 8 --kl "C" --kl_w_start 0.0002 --kl_w_end 0.00025 --kl_cycle 8 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0003 --kl_w_end 0.00035 --kl_cycle 8 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0004 --kl_w_end 0.00045 --kl_cycle 8 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0005 --kl_w_end 0.00055 --kl_cycle 8 --kl_ratio 1.0',
'train.py --n_layers 8 --kl "C" --kl_w_start 0.0006 --kl_w_end 0.00065 --kl_cycle 8 --kl_ratio 1.0'

]

model = ['GATb']

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

