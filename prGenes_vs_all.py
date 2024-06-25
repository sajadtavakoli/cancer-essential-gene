import numpy as np
import pandas as pd 
from scipy.stats import pearsonr

dir = './data/'

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
corr_exp_ess = {}
for name1 in name_pr: 
    temp = []
    ess = ess_pr.loc[ess_pr['Description']==name1].to_numpy()
    ess = ess.reshape(-1)[1:]
    for name2 in common_genes:
        exp = exp_all.loc[exp_all['Description']==name2].to_numpy()
        exp = exp.reshape(-1)[1:]
        corr, p_value = pearsonr(ess, exp)
        temp.extend([corr, p_value])
    temp = np.array(temp)
    corr_exp_ess[name1] = temp


corr_exp_exp = {}
for name1 in name_pr: 
    temp = []
    exp1 = exp_pr.loc[exp_pr['Description']==name1].to_numpy()
    exp1 = exp1.reshape(-1)[1:]
    for name2 in common_genes:
        exp2 = exp_all.loc[exp_all['Description']==name2].to_numpy()
        exp2 = exp2.reshape(-1)[1:]
        corr, p_value = pearsonr(exp1, exp2)
        temp.extend([corr, p_value]) ### BUG BUG BUG
    temp = np.array(temp)
    corr_exp_exp[name1] = temp

ali

cell_lines = ess_all.keys().to_list()[1:]
x, y = [], []
for i, gene_name in enumerate(name_pr):
    corr_exp = np.array(corr_exp_exp[i])
    corr_ess = np.array(corr_exp_ess[i])
    for j , cell in enumerate(cell_lines):
        label = ess_pr[ess_pr['Description']==gene_name][cell].values[0]
        exp_all_genes = exp_all[cell].to_numpy()
        exp_target_gene = np.array([exp_pr[exp_pr['Description']==gene_name][cell].values[0]] * len(name_all))
        #cop_null = cop_pr[cop_pr['Description']==gene_name][cell].isnull().values[0]
        cop_target_gene = cop_pr[cop_pr['Description']==gene_name][cell].values[0]
        cop_target_gen = np.array([cop_target_gene]*len(name_all))
        y.append(label)
        temp = []
        temp.extend[corr_exp, corr_ess, exp_all_genes, exp_target_gene, cop_target_gen]
        x.append(temp)

x, y = np.array(x), np.array(y)

np.save('x.npy', x)
np.save('y.npy', y)







# corr between ess score of each gene and all genes -> matrix: [#pr_genes, #all_genes]
# corr between exp of pr_genes and all genes -> [#pr_genes, #all_genes]
# exp
# copy
# mutation

# matrix: [#pr_genes * # all_genes, ]


