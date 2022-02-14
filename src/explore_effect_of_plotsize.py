import os,glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

target_dir = 'data/true_div_data/true_diversity_data_proj_aus_*m2.txt'
plotsize_dfs = glob.glob(target_dir)
plot_sizes = []
div_counts = []
for file in plotsize_dfs:
    splot_df = pd.read_csv(file,sep='\t')
    div = splot_df.species_richness.values
    plotsize = int(os.path.basename(file).split('_')[-1].split('m2')[0])
    plot_sizes.append(plotsize)
    div_counts.append(div)

plot_sizes = np.array(plot_sizes)
div_counts = np.array(div_counts)

new_order = np.argsort(plot_sizes)
plot_sizes_sorted = plot_sizes[new_order]
div_counts_sorted = div_counts[new_order]

fig = plt.figure(figsize=(10,5))
plt.boxplot(div_counts_sorted,patch_artist=True,labels=plot_sizes_sorted)
fig.savefig('data/true_div_data/div_per_plotsize.pdf')




