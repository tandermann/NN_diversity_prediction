import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# wcvp reference file
wcvp_file = 'data/wcvp/wcvp_v5_jun_2021.txt'
wcvp_data = pd.read_csv(wcvp_file,sep='|')
wcvp_data_species = wcvp_data[wcvp_data['rank']=='SPECIES']
wcvp_data_species_accepted = wcvp_data_species[wcvp_data_species.taxonomic_status=='Accepted']

# species list file
species_list_file = 'data/true_div_data/true_diversity_data_proj_aus.txt'
species_list_data = pd.read_csv(species_list_file,sep='\t')
coords = species_list_data[['lon','lat']].values
species_list_strings = species_list_data.species_list.values
species_lists = [i.split(', ') for i in species_list_strings]
genus_lists = [[j.split(' ')[0] for j in i] for i in species_lists]

# provide family name
target_families = ['Orchidaceae', 'Asteraceae', 'Gramineae', 'Fabaceae', 'Rubiaceae', 'Myrtaceae', 'Proteaceae', 'Leguminosae', 'Lauraceae', 'Euphorbiaceae', 'Malvaceae', 'Melastomataceae', 'Annonaceae', 'Arecaceae', 'Sapotaceae', 'Poaceae']
for target_family in target_families:
    # extract all species for family from wcvp
    target_rows = wcvp_data_species_accepted[wcvp_data_species_accepted.family.values == target_family]
    target_species_list = (target_rows.genus + ' ' + target_rows.species).values.astype(str)
    target_species_list = np.array([i for i in target_species_list if not i.startswith('xx')])
    target_genus_list = np.unique(target_rows.genus).astype(str)
    target_genus_list = np.delete(target_genus_list,np.where(target_genus_list=='xx'))

    # count how many species belonging to genera of the target group were recorded
    target_ids = np.array([np.intersect1d(i, target_genus_list,return_indices=True)[1] for i in genus_lists],dtype=object)
    np.array([np.array(species_list)[target_ids[i]] for i,species_list in enumerate(species_lists)],dtype=object)
    n_target_matches_genus = np.array([len(i) for i in target_ids])
    # just in case some records are only identified to family level, extract those here
    n_target_matches_family = np.array([sum(np.array(i)==target_family) for i in genus_lists])
    div_in_target_group = n_target_matches_genus+n_target_matches_family

    fig = plt.figure(figsize=(4, 4))
    plt.hist(div_in_target_group,np.arange(0,30))
    plt.title('%s species diversity'%target_family)
    fig.savefig('plots/training_data_per_family/histogram_splot_species_div_%s.pdf'%target_family,bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(9, 5.9))
    plt.scatter(coords[:,0],
                coords[:,1],
                c=div_in_target_group,
                cmap = 'inferno_r',
                marker = 's',
                edgecolors = 'none',
                s = 30)
    plt.title('%s species diversity'%target_family)
    fig.savefig('plots/training_data_per_family/map_splot_species_div_%s.png'%target_family,
                bbox_inches='tight',
                dpi=500,
                transparent=True)
    plt.close()



