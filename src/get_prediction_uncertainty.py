import numpy as np
import os, glob
import matplotlib.pyplot as plt
import cmocean

dropout=False
mode='alpha'

if dropout:
    if mode == 'alpha':
        target_dir = 'results/production_models/production_model_alpha_mcdropout/nn_1234_2_states_27_features_30_neighbours_30_15_5_nodes_0.2_0.1_0.0_dropout_sigmoid_40_batchsize_500_epochs_alpha'
    elif mode == 'beta':
        target_dir = 'results/production_models/production_model_beta_mcdropout/nn_1234_5_states_27_features_30_neighbours_30_15_5_nodes_0.2_0.1_0.0_dropout_sigmoid_40_batchsize_1100_epochs_beta'
    elif mode == 'gamma':
        target_dir = 'results/production_models/production_model_gamma_mcdropout/nn_1234_7_states_27_features_50_neighbours_30_15_5_nodes_0.2_0.1_0.0_dropout_sigmoid_40_batchsize_2000_epochs_gamma'
    all_data = np.loadtxt(os.path.join(target_dir,'predicted_labels_%s_div.txt'%mode),delimiter=',')
    all_div_estimates = all_data[:,2:]
    all_coords = all_data[:,:2].astype(int)
else:
    target_dir = 'results/production_models/production_model_%s_ensemble'%mode
    target_folders = glob.glob(os.path.join(target_dir,'nn_*'))
    all_data = np.array([np.loadtxt(os.path.join(i,'predicted_labels_%s_div.txt'%mode),delimiter=',') for i in target_folders])
    all_div_estimates = all_data[:,:,2].T
    all_coords = all_data[0,:,:2]


# remove columns where there is no variation in predictions (was the case for one of the alpha models)
all_div_estimates = np.array([i for i in all_div_estimates.T if np.max(i)-np.min(i) > 0]).T

mean_div_estimates = np.mean(all_div_estimates,axis=1)
std_estimates = np.std(all_div_estimates,axis=1)
uncertainty_measure = std_estimates/mean_div_estimates
median=False
if median:
    lower = np.quantile(all_div_estimates,0.25,axis=1)
    upper = np.quantile(all_div_estimates,0.75,axis=1)
    delta = upper-lower
    mean_div_estimates = np.median(all_div_estimates,axis=1)
    uncertainty_measure = delta/mean_div_estimates

# #scale all diversity predictions between 0 and 1
# rescale=True
# if rescale:
#     all_div_estimates_scaled = ((all_div_estimates.T-np.min(all_div_estimates.T,axis=1).reshape(len(all_div_estimates.T),1))/(np.max(all_div_estimates.T,axis=1)-np.min(all_div_estimates.T,axis=1)).reshape(len(all_div_estimates.T),1)).T
#     mean_div_estimates = np.mean(all_div_estimates_scaled,axis=1)
#     std_estimates = np.std(all_div_estimates_scaled,axis=1)

#uncertainty_measure = np.mean(np.abs(all_div_estimates-mean_div_estimates.reshape(len(all_div_estimates),1))/mean_div_estimates.reshape(len(all_div_estimates),1),axis=1)
#uncertainty_measure = np.var(all_div_estimates,axis=1)
plt.hist(uncertainty_measure)
plt.show()

cutoff = np.median(uncertainty_measure)
print('Median uncertainty: %.4f'%cutoff)
out_data = np.hstack([all_coords,mean_div_estimates.reshape(len(mean_div_estimates),1),uncertainty_measure.reshape(len(uncertainty_measure),1)])
np.savetxt(os.path.join(target_dir,'%s_uncertainty_predictions.txt'%mode),out_data,fmt='%.4f')

cmap = 'inferno_r' #'cmo.speed'

# select only cells with acceptable std
stepsize=0.05
for i in np.arange(0,1+stepsize,stepsize):
    selected_ids = np.where(uncertainty_measure<=i)[0]
    selected_coords = all_coords[selected_ids]
    selected_divs = mean_div_estimates[selected_ids]

    # plot on map
    width=9.5
    height=6.4
    pointsize=1
    fig = plt.figure(figsize=(width, height))
    plt.scatter(all_coords[:, 0],
                all_coords[:, 1],
                c=mean_div_estimates,
                cmap='gray_r',
                alpha=1,
                marker='s',
                edgecolors='none',
                s=pointsize)
    plt.scatter(selected_coords[:, 0],
                selected_coords[:, 1],
                c=selected_divs,
                cmap=cmap,
                marker='s',
                edgecolors='none',
                s=pointsize)
    cbar = plt.colorbar()
#    plt.gca().set_facecolor('#4D629A')
    plt.axis('off')
#    cbar.set_label("Predicted species diversity", labelpad=+5)
    outfile = os.path.join(target_dir,'plot_prediction_uncertainty_cutoff_%.2f.png'%i)
    fig.savefig(outfile,
                bbox_inches='tight',
                dpi=500)
    plt.close()


# plot on map
width=9.5
height=6.4
pointsize=1

fig = plt.figure(figsize=(width, height))
plt.scatter(all_coords[:, 0],
            all_coords[:, 1],
            c=mean_div_estimates,
            cmap=cmap,
            marker='s',
            edgecolors='none',
            s=pointsize)
cbar = plt.colorbar()
#    plt.gca().set_facecolor('#4D629A')
plt.axis('off')
#    cbar.set_label("Predicted species diversity", labelpad=+5)
outfile = os.path.join(target_dir,'plot_prediction.png')
fig.savefig(outfile,
            bbox_inches='tight',
            dpi=500)
plt.show()
plt.close()

selected_ids = np.where(uncertainty_measure<=cutoff)[0]
selected_coords = all_coords[selected_ids]
selected_divs = mean_div_estimates[selected_ids]

fig = plt.figure(figsize=(width, height))
plt.scatter(all_coords[:, 0],
            all_coords[:, 1],
            c='grey',
            #c=mean_div_estimates,
            #cmap='gray_r',
            alpha=1,
            marker='s',
            edgecolors='none',
            s=pointsize)
plt.scatter(selected_coords[:, 0],
            selected_coords[:, 1],
            c=selected_divs,
            cmap=cmap,#'cmo.speed'
            marker='s',
            edgecolors='none',
            s=pointsize)
cbar = plt.colorbar()
#    plt.gca().set_facecolor('#4D629A')
plt.axis('off')
#    cbar.set_label("Predicted species diversity", labelpad=+5)
outfile = os.path.join(target_dir,'plot_prediction_uncertainty_mean_%.2f.png'%cutoff)
fig.savefig(outfile,
            bbox_inches='tight',
            dpi=500)
plt.show()
plt.close()


fig = plt.figure(figsize=(width, height))
plt.scatter(all_coords[:, 0],
            all_coords[:, 1],
            c=mean_div_estimates,
            cmap='gray_r',
            alpha=1,
            marker='s',
            edgecolors='none',
            s=pointsize)
plt.scatter(selected_coords[:, 0],
            selected_coords[:, 1],
            c=selected_divs,
            cmap=cmap,#'cmo.speed'
            marker='s',
            edgecolors='none',
            s=pointsize)
cbar = plt.colorbar()
#    plt.gca().set_facecolor('#4D629A')
plt.axis('off')
#    cbar.set_label("Predicted species diversity", labelpad=+5)
outfile = os.path.join(target_dir,'plot_prediction_uncertainty_mean_%.2f_detail.png'%cutoff)
fig.savefig(outfile,
            bbox_inches='tight',
            dpi=500)
plt.show()
plt.close()

uncertainty_cmap = 'gray_r' #'Reds'

max_uncertainty = 0.5
capped_uncertainty = uncertainty_measure.copy()
capped_uncertainty[capped_uncertainty>max_uncertainty] = max_uncertainty
fig = plt.figure(figsize=(width, height))
plt.scatter(all_coords[:, 0],
            all_coords[:, 1],
            c=capped_uncertainty,
            cmap=uncertainty_cmap,#'cmo.speed'
            marker='s',
            edgecolors='none',
            s=pointsize)
cbar = plt.colorbar()
#    plt.gca().set_facecolor('#4D629A')
plt.axis('off')
#cbar.set_label("Standard deviation of predictions (scaled by mean)", labelpad=+5)
outfile = os.path.join(target_dir,'plot_uncertainty.png')
fig.savefig(outfile,
            bbox_inches='tight',
            dpi=500)
plt.show()
plt.close()

fig = plt.figure(figsize=(width, height))
plt.scatter(all_coords[:, 0],
            all_coords[:, 1],
            c=np.log(uncertainty_measure),
            cmap=uncertainty_cmap,#'cmo.speed'
            marker='s',
            edgecolors='none',
            s=pointsize)
cbar = plt.colorbar()
#    plt.gca().set_facecolor('#4D629A')
plt.axis('off')
#cbar.set_label("Standard deviation of predictions (scaled by mean, log-transformed)", labelpad=+5)
outfile = os.path.join(target_dir,'plot_uncertainty_log.png')
fig.savefig(outfile,
            bbox_inches='tight',
            dpi=500)
plt.show()
plt.close()

