import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import cust_func

# read file with splot data
splot_file = 'data/true_div_data/true_diversity_data_proj_aus.txt'
#splot_data = pd.read_csv(splot_file,sep='\t')

seed = 1234
data_obj = cust_func.prep_data(seed)
data_obj._true_div_data_file = splot_file

# define how many neighbours to extract
for n in [10,30,50,100,200,500,1000]:
    data_obj.load_splot_data(extract_richness=True)
    data_obj.get_beta_and_gamma_div_n_nearest_neighbours(n, beta_mode='sorensen')

    coords = data_obj._true_div_coords
    alpha_div = data_obj._alpha_div
    beta_div = data_obj._beta_div
    gamma_div = data_obj._gamma_div
    radius = data_obj._avg_dist/1000 # in km


    # plot figure
    fig = plt.figure(figsize=(18, 12))
    subplot1 = fig.add_subplot(221)
    plt.scatter(coords[:,0],
                coords[:,1],
                c=alpha_div,
                cmap='inferno_r',
                marker='s',
                edgecolors='none',
                s=20)
    plt.title('Alpha diversity')
    plt.colorbar()

    subplot2 = fig.add_subplot(222)
    #beta_divs = gamma_divs/selected_splot_data.species_richness.values
    plt.scatter(coords[:,0],
                coords[:,1],
                c=beta_div,
                cmap='inferno_r',
                marker='s',
                edgecolors='none',
                s=20)
    plt.title('Beta diversity')
    plt.colorbar()

    subplot3 = fig.add_subplot(223)
    plt.scatter(coords[:,0],
                coords[:,1],
                c=gamma_div,
                cmap='inferno_r',
                marker='s',
                edgecolors='none',
                s=20)
    plt.title('Gamma diversity')
    plt.colorbar()

    subplot4 = fig.add_subplot(224)
    plt.scatter(coords[:,0],
                coords[:,1],
                c=radius,
                cmap='inferno_r',
                marker='s',
                edgecolors='none',
                s=20)
    plt.title('Neighbourhood radius (in km)')
    plt.colorbar()

    plt.subplots_adjust(wspace=0.2,hspace=0.2)
    outfile = 'plots/map_alpha_gamma_div_training_data_n_%i.png'%n
    fig.savefig(outfile,
                bbox_inches='tight',
                dpi=500,
                transparent=True)




