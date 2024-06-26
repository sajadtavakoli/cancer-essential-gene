import numpy as np
import pandas as pd 
from scipy.stats import pearsonr

dir = '/work3/sajata/ACB-course/cancer-essential-gene/data/'

scores_path = dir+'Achilles_v2.11_training_phase3.gct'
copy_path = dir+'CCLE_copynumber_training_phase3.gct'
exp_path = dir+'CCLE_expression_training_phase3.gct'
mutation_path = dir+'CCLE_hybridmutation_training_phase3.gct'
pr_genes_path = dir + 'prioritized_gene_list_phase3.txt'

# Read files

ess_all = pd.read_csv(scores_path, sep='\t', skiprows=2).sort_values(by='Description').drop('Name', axis=1)
ess_all = ess_all.dropna()
exp_all = pd.read_csv(exp_path, sep='\t', skiprows=2).sort_values(by='Description').drop('Name', axis=1)
exp_all = exp_all.dropna()
name_all = exp_all['Description']
cop_all = pd.read_csv(copy_path, sep='\t', skiprows=2).sort_values(by='Description').drop('Name', axis=1)
#cop_all = cop_all[cop_all['Description'].isin(name_all)]
cop_all = cop_all.dropna()
mut_all = pd.read_csv(mutation_path, sep='\t', skiprows=2).sort_values(by='Description').drop('Name', axis=1)
mut_all = mut_all.dropna()

common_genes = set(ess_all['Description']).intersection(exp_all['Description']).intersection(cop_all['Description'])
common_genes = list(common_genes)

ess_all = ess_all[ess_all['Description'].isin(common_genes)]
ess_all = ess_all.sort_values(by='Description')
exp_all = exp_all[exp_all['Description'].isin(common_genes)]
exp_all = exp_all.sort_values(by='Description')
cop_all = cop_all[cop_all['Description'].isin(common_genes)]
cop_all = cop_all.sort_values(by='Description')



name_pr = pd.read_table(pr_genes_path, header=None)[0].tolist()

for name in name_pr:
    if name in common_genes:
        continue
    else: 
        name_pr = [x for x in name_pr if x != name]

# name_pr = pd.DataFrame(data={'Description': name_pr})
ess_pr = ess_all[ess_all['Description'].isin(name_pr)].sort_values(by='Description')
cop_pr = cop_all[cop_all['Description'].isin(name_pr)].sort_values(by='Description')
exp_pr = exp_all[exp_all['Description'].isin(name_pr)].sort_values(by='Description')
mut_pr = mut_all[mut_all['Description'].isin(name_pr)].sort_values(by='Description')


# correlation between
# shape: (#pr_genes, #all_genes)
# 
import time
common_genes = list(ess_all['Description']) # updated -> bug fixed (sorted now)



sajad
###
# -> creating image and labels for training and saving in memory
###

import joblib as gb
paths = gb.load(dir+'corr/'+'*.npy')
loaded_names = []
cell_lines = ess_all.keys().to_list()[1:]
x, y = [], []
labels = {}
for i, path in enumerate(paths):
    gene_name = path.split('.')[0]
    if gene_name in loaded_names: 
        continue
    loaded_names.append(gene_name)
    corr = np.load(path)
    img = np.zeros((8, len(common_genes)), dtype=np.float32)
    
    for j, cell in enumerate(cell_lines):
        label = ess_pr[ess_pr['Description']==gene_name][cell].values[0]
        exp_all_genes = exp_all[cell].to_numpy()   
        exp_target_gene = np.array([exp_pr[exp_pr['Description']==gene_name][cell].values[0]] * len(common_genes))
        cop_all_genes = cop_all[cell].to_numpy()  
        cop_target_gene = cop_pr[cop_pr['Description']==gene_name][cell].values[0]
        cop_target_gen = np.array([cop_target_gene]*len(common_genes))
        img[0, :] = corr[:, 0]
        img[1, :] = corr[:, 1]
        img[2, :] = corr[:, 2]
        img[3, :] = corr[:, 3]
        img[4, :] = exp_all_genes
        img[5, :] = exp_target_gene
        img[6, :] = cop_all_genes
        img[7, :] = cop_target_gen
        np.save(dir+'imgs/'+f'img-f{gene_name}-{cell}', img)
        labels[gene_name+'-'+cell] = label
        break 
    break 

gb.dump('labels.joblib', labels)
