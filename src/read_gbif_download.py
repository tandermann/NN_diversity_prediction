import pandas as pd
import numpy as np

gbif_df = pd.read_csv('data/gbif/australia.csv',sep='\t',error_bad_lines=False)
gbif_df_reduced = gbif_df.loc[:,['species','decimalLongitude','decimalLatitude']]
dtypes = {'species':str,'decimalLongitude':float,'decimalLatitude':float}
for c in gbif_df_reduced.columns:
    gbif_df_reduced[c] = gbif_df_reduced[c].astype(dtypes[c])
gbif_df_reduced.to_csv('data/gbif/australia_formatted.txt',sep='\t',index=False)


wcvp = pd.read_csv('data/wcvp/wcvp_v5_jun_2021.txt',sep='|')
# exclude all rows of df that do not contain species epithet
wcvp = wcvp.loc[~pd.isna(wcvp.species),]
species_names = (wcvp.genus + ' ' + wcvp.species).values
species_names = sorted(species_names)
# only keep those names with upper case genus name
species_names = [i for i in species_names if i[0].isupper()]
species_names = np.unique(species_names)
np.savetxt('data/wcvp/wcvp_species_list.txt',species_names,fmt='%s')
