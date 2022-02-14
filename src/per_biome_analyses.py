import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mode = 'beta'

if mode == 'alpha':
    pred_per_biome_file = 'results/production_models/production_model_alpha_ensemble/pred_with_biome_info.txt'
    cutoff = 0.3033
elif mode == 'beta':
    pred_per_biome_file = 'results/production_models/production_model_beta_ensemble/pred_with_biome_info.txt'
    cutoff = 0.0891
elif mode == 'gamma':
    pred_per_biome_file = 'results/production_models/production_model_gamma_ensemble/pred_with_biome_info.txt'
    cutoff = 0.3655
pred_per_biome = pd.read_csv(pred_per_biome_file,sep='\t',header=None)
pred_per_biome = pred_per_biome[pred_per_biome.iloc[:,4]<1]

biome_ids = []
biom_div_estimates = []
for biome_data in pred_per_biome.groupby(2):
    div_estimates = biome_data[1].values[:,4:]
    mean_div_estimates = np.mean(div_estimates,axis=1)
    biome_ids.append(biome_data[0])
    biom_div_estimates.append(mean_div_estimates)

# subtract 1 for correct python indexing
biome_ids = np.array(biome_ids)-1
# sort by decreasing biodiv
biodiv_means = [np.mean(i) for i in biom_div_estimates]
sorted_ids = np.argsort(biodiv_means)[::-1]
# apply sorted ids
biom_div_estimates_sorted = np.array(biom_div_estimates)[sorted_ids]
biome_ids_sorted = biome_ids[sorted_ids]
print([len(i) for i in np.array(biom_div_estimates)[sorted_ids]])

color_list = ['#31a354',
               '#BDB336',
               '#9EC748',
               '#addd8e',
               '#006878',
               '#83B797',
               '#EFA32B',
               '#FAD71D',
               '#6BC0B4',
               '#C4A172',
               '#BCD88F',
               '#D9232A',
               '#E0785B',
               '#D81774']
colors = np.array(color_list)[biome_ids_sorted]
# plot the figure
fig = plt.figure(figsize=(4, 4))
violin_plots = plt.violinplot(biom_div_estimates_sorted,showmeans=True,showmedians=False,showextrema=False)
for i,pc in enumerate(violin_plots['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(1)
vp = violin_plots['cmeans']
vp.set_edgecolor('black')
vp.set_linewidth(1)
plt.gca().set_axisbelow(True)
plt.xticks(np.arange(len(biom_div_estimates_sorted))+1, biome_ids_sorted+1, rotation=0)
plt.grid()
plt.gca().axes.xaxis.set_ticklabels([])
fig.savefig('plots/%s_div_by_biome.pdf'%mode,
            bbox_inches='tight',
            dpi=500)
plt.show()

