import numpy as np 
import pandas as pd 
import glob as gb 

data_dir = '/home/sajata/courses/computational-biology/project/data/'
dir = '/home/sajata/courses/computatinal-biology/project/cancer-essential-gene/'


import pandas as pd

scores_file = 'Achilles_v2.11_training_phase3.gct'
copy_num_file = 'CCLE_copynumber_training_phase3.gct'
exp_data_file = 'CCLE_expression_training_phase3.gct'
mutation_file = 'CCLE_hybridmutation_training_phase3.gct'

# Read files
scores = pd.read_csv(scores_file, sep='\t', skiprows=2)
copy_num = pd.read_csv(copy_num_file, sep='\t', skiprows=2)
exp_data = pd.read_csv(exp_data_file, sep='\t', skiprows=2)
mutation_data = pd.read_csv(mutation_file, sep='\t', skiprows=2)

scores_genes = scores['Description']
copy_num_genes = copy_num['Description']
exp_data_genes = exp_data['Description']
mutation_genes = mutation_data['Description']

genes1 = set(scores_genes).intersection(set(copy_num_genes))
genes2 = set(exp_data_genes).intersection(genes1)
final_genes = set(mutation_genes).intersection(genes2)


scores_final = scores[scores['Description'].isin(final_genes)]
copy_num_final = copy_num[copy_num['Description'].isin(final_genes)]
exp_data_final = exp_data[exp_data['Description'].isin(final_genes)]
mutation_final = mutation_data[mutation_data['Description'].isin(final_genes)]

print(scores_final['Description'].head())
print(copy_num_final['Description'].head())
print(exp_data_final['Description'].head())
print(mutation_final['Description'].head())

scores_sorted = scores_final.sort_values(by="Description")
copy_num_sorted = copy_num_final.sort_values(by="Description")
exp_data_sorted = exp_data_final.sort_values(by="Description")
mutation_sorted = mutation_final.sort_values(by="Description")

scores_sorted = scores_final.drop(columns=['Name'])
copy_num_sorted = copy_num_final.drop(columns=['Name'])
exp_data_sorted = exp_data_final.drop(columns=['Name'])
mutation_sorted = mutation_final.drop(columns=['Name'])

print('-'*100)
print(scores_sorted['Description'].head())
print(copy_num_sorted['Description'].head())
print(exp_data_sorted['Description'].head())
print(mutation_sorted['Description'].head())


exp_corr = exp_data_sorted.T.corr(method='pearson')

import matplotlib.pyplot as plt
import seaborn as sns

# plt.figure(figsize=(8, 6))
# sns.heatmap(exp_corr, annot=True, cmap='coolwarm', center=0)
# plt.title('Pearson Correlation Matrix')
# plt.show()

cells = exp_data_sorted.columns.tolist()[2:]

num_best = 20
top_k = 10
best_score_list = []
for cell in cells:
    temp = scores_sorted[['Description', cell]]
    sorted = temp.sort_values(by=cell)[:top_k]
    best_score_list.append(sorted)

genes = exp_data_sorted['Description'].tolist()
gene_occurence = [0]*len(genes)
for i, gene in enumerate(genes): 
    temp = []
    for j, cell in enumerate(cells):
        best_score_temp = best_score_list[j]
        best_genes = best_score_temp['Description'].tolist()
        temp.extend(best_genes)
    count = temp.count(gene)
    gene_occurence[i] = count

     
selected_genes_name, selected_genes_score = [], []
gene_occurence = np.array(gene_occurence)
sorted_index = np.argsort(gene_occurence)[::-1][:num_best]

for idx in sorted_index:
    selected_genes_name.append(genes[idx])
    selected_genes_score.append(gene_occurence[idx])

import matplotlib.pyplot as plt 
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_genes_score)))
plt.bar(selected_genes_name, selected_genes_score, color=colors)

# Adding titles and labels
plt.title('Occurrence of Genes')
plt.xlabel('Genes')
plt.ylabel('Occurrence')
plt.xticks(rotation=45, ha='right')

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a color bar to indicate the color mapping
# sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(selected_genes_score), vmax=max(selected_genes_score)))
# sm.set_array([])
# cbar = plt.colorbar(sm)
# cbar.set_label('Occurrences', fontsize=14)
plt.tight_layout()
# Display the plot
plt.savefig('best 10 genes.jpg', dpi=540)
plt.show()
    

# exp_data_sorted = exp_data_sorted.

scores_selected = scores_sorted[scores_sorted['Description'].isin(selected_genes_name)]
copy_num_selected = copy_num_sorted[copy_num_sorted['Description'].isin(selected_genes_name)]
exp_data_selected = exp_data_sorted[exp_data_sorted['Description'].isin(selected_genes_name)]
mutation_selected = mutation_sorted[mutation_sorted['Description'].isin(selected_genes_name)]



### expression boxplot 
df_transposed = exp_data_selected.set_index('Description').transpose()[selected_genes_name]
plt.figure(figsize=(12, 8))
df_transposed.boxplot()

# Adding titles and labels
plt.title('Expression of each gene in all cell lines', fontsize=16, fontweight='bold')
plt.xlabel('gene', fontsize=14)
plt.ylabel('expression', fontsize=14)

# Display the plot
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('expression boxplot.jpg', dpi=540)
plt.show()




### essentiality score boxplot 
df_transposed = scores_selected.set_index('Description').transpose()[selected_genes_name]
plt.figure(figsize=(12, 8))
df_transposed.boxplot()

# Adding titles and labels
plt.title('essentiality score for each gene in all cell lines', fontsize=16, fontweight='bold')
plt.xlabel('gene', fontsize=14)
plt.ylabel('score', fontsize=14)

# Display the plot
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('scores boxplot.jpg', dpi=540)
plt.show()



### copy number boxplot 
df_transposed = copy_num_selected.set_index('Description').transpose()[selected_genes_name]
plt.figure(figsize=(12, 8))
df_transposed.boxplot()

# Adding titles and labels
plt.title('copy number variation for each gene in all cell lines', fontsize=16, fontweight='bold')
plt.xlabel('gene', fontsize=14)
plt.ylabel('copy number variation', fontsize=14)

# Display the plot
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('copy number boxplot.jpg', dpi=540)
plt.show()


### Correlation between exp of genes

df_transposed = exp_data_selected.set_index('Description').transpose()[selected_genes_name]
correlation_matrix = df_transposed.corr(method='pearson')

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Gene expression', fontsize=16, fontweight='bold')
plt.xlabel('Genes', fontsize=14)
plt.ylabel('Genes', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.savefig('expression correlation.jpg', dpi=540)
plt.show()



