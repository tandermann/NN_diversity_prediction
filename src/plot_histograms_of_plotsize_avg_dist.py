import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import cust_func


# random place for code for plotting Topaza node age histograms
# factor = 0.05
# fig = plt.figure(figsize=(10,4))
# a = np.random.normal(15*factor,2*factor,10000)
# #b = np.random.normal(5*factor,1.2*factor,10000)
# c = np.random.normal(25*factor,2*factor,10000)
# plt.hist(a,np.arange(0,30*factor,0.1*factor),zorder=0)
# #plt.axvline(np.mean(a),color='black',linestyle='--',zorder=1)
# #plt.hist(b,np.arange(0,30*factor,0.1*factor),zorder=0)
# #plt.axvline(np.mean(b),color='black',linestyle='--',zorder=1)
# plt.hist(c,np.arange(0,30*factor,0.1*factor),zorder=0)
# #plt.axvline(np.mean(c),color='black',linestyle='--',zorder=1)
# #plt.axvline(14*factor,color='red',zorder=1)
# plt.xticks([])
# plt.yticks([])
# plt.show()
# fig.savefig('/Users/tobiasandermann/Desktop/hist.png',transparent=True)
#
#
# name = ['a','b','c']
# price = [0.3,0.1,0.4]
# plt.barh(np.arange(len(price)),price,0.1)
# plt.show()








#___________________________PLOTSIZE INFO_______________________________
plotsize_info_file = 'data/true_div_data/true_diversity_data_proj_aus.txt'

plotsize_data = pd.read_csv(plotsize_info_file,sep='\t').plotsize.values
plotsizes,counts = np.unique(plotsize_data,return_counts=True)

sorted_ids = np.argsort(counts)[::-1]
plotsizes = plotsizes[sorted_ids]
counts = counts[sorted_ids]

n_groups = 10
sel_plotsizes = plotsizes[:n_groups]
sel_counts = counts[:n_groups]

new_order = np.argsort(sel_plotsizes)
sel_plotsizes = sel_plotsizes[new_order]
sel_counts = sel_counts[new_order]

fig = plt.figure()
x=sel_plotsizes.astype(str)
y=sel_counts
plt.gca().bar(x,y,label='Training data plotsizes')
plt.axvline(np.where(sel_plotsizes==500)[0][0],color='red',linestyle='--',label='Plotsize for prediction')
plt.legend()
plt.xlabel('Plotsize in m2')
plt.ylabel('Count of vegetation plots')
plt.show()
fig.savefig('plots/distribution_splot_plotsizes.pdf',bbox_inches='tight', dpi = 500,transparent=True)



#___________________________NEIGHBORHOOD RADIUS TRAINING DATA_______________________________
true_div_data_file = 'data/true_div_data/true_diversity_data_proj_aus.txt'
data_obj_path = 'data/cnn_input/l_100000_g_20/data_obj.pkl'
# load model and compile distances
data_obj = cust_func.load_obj(data_obj_path)
data_obj.prep_labels_train(true_div_data_file,
                           target_family=None,
                           n_neighbours=50,
                           beta_mode='sorensen')
# get avg distance measures
avg_dist_values = data_obj._avg_dist.flatten()/1000

# plot figure
fig = plt.figure()
plt.hist(avg_dist_values,bins=100,label='Neighborhood radius training data')
plt.axvline(5000/1000,color='red',linestyle='--',label='Neighborhood radius for predictions')
plt.legend()
plt.xlabel('Neighborhood radius in km')
plt.ylabel('Count of vegetation plots')
plt.show()
fig.savefig('plots/distribution_neighborhood_radiuses.pdf',bbox_inches='tight', dpi = 500,transparent=True)








