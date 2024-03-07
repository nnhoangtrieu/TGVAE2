import os
import argparse
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import rdkit 
from rdkit.Chem.QED import qed
import sascorer 
from rdkit import Chem
from rdkit.Chem import Descriptors
from multiprocessing import Pool

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol
def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return sascorer.calculateScore(mol)
def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)
def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)
def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)
def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def get_plot(gen_set, out_path, num_cpu=4) :
    rdkit.rdBase.DisableLog('rdApp.*')

    with open('data/test.txt','r') as f :
        test = f.readlines()
        test = test[1:]
        test = [c[:-6] for c in test]

    generated = OrderedDict(
        {'MOSES': pd.DataFrame({'SMILES': test})})
    

    generated['TGVAE'] = pd.DataFrame({'SMILES': gen_set})
    metrics = {
        'weight': weight,
        'logP': logP,
        'SA': SA,
        'QED': QED
    }

    for s in generated.values():
        s['ROMol'] = mapper(num_cpu)(get_mol, s['SMILES'])

    distributions = OrderedDict()
    for metric_name, metric_fn in metrics.items():
        distributions[metric_name] = OrderedDict()
        for _set, _molecules in generated.items():
            distributions[metric_name][_set] = mapper(num_cpu)(
                metric_fn, _molecules['ROMol'].dropna().values
            )

    for metric_i, metric_name in enumerate(metrics):
        for model, d in distributions[metric_name].items():
            dist = wasserstein_distance(distributions[metric_name]['MOSES'], d)
            sns.distplot(
                d, hist=False, kde=True,
                kde_kws={'shade': True, 'linewidth': 3},
                label='{0} ({1:0.2g})'.format(model, dist))
        plt.title(metric_name, fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_path, metric_name+'.png'),
            dpi=250
        )
        plt.close()