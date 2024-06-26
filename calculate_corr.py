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


#sajad
###
# -> calculating correlations and p-values and save in memeory 
###

counter = 0
corr_exp_ess, corr_exp_exp = {}, {}
for name1 in name_pr:
    t1 = time.time()
    
    #temp_ess_exp = []
    ess1 = ess_pr.loc[ess_pr['Description']==name1].to_numpy()
    ess1 = ess1.reshape(-1)[1:]
    
    #temp_exp_exp = []
    exp1 = exp_pr.loc[exp_pr['Description']==name1].to_numpy()
    exp1 = exp1.reshape(-1)[1:]

    temp = []
    for name2 in common_genes:
        exp2 = exp_all.loc[exp_all['Description']==name2].to_numpy()
        exp2 = exp2.reshape(-1)[1:]

        corr_ess_exp, p_value_ess_exp = pearsonr(ess1, exp2)
        #temp_ess_exp.append([corr_ess_exp, p_value_ess_exp])

        corr_exp_exp, p_value_exp_exp = pearsonr(exp1, exp2)
        temp.append([corr_ess_exp, p_value_ess_exp, corr_exp_exp, p_value_exp_exp])
        
    
    #temp_ess_exp = np.array(temp_ess_exp)
    #temp_exp_exp = np.array(temp_exp_exp) 
    #np.save(dir+name1+'corr_ess_exp', temp_ess_exp)
    #np.save(dir+name1+'corr_exp_exp', temp_exp_exp)
    #corr_exp_ess[name1] = temp_ess_exp
    #corr_exp_exp[name1] = temp_exp_exp
    
    temp = np.array(temp)
    np.save(dir+'corr/'+name1, temp)

    t2 = time.time()
    counter+=1
    print(counter, 'priorotized gene:', name1) 
    print(f'time: {t2-t1}\n')
    print('-'*50)
