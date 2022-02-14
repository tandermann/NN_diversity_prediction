from src import cust_func
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import multiprocessing
from functools import partial
#import importlib
#importlib.reload(cust_func)

# ________________________________SETTINGS_______________________
# settings for feature extraction________________________________
n_instances = None #  define how many instances to extract data for (set to None for all)
window_length = 100000  # define side-length of square in m
n_cells_per_row = 20  # define grid-cell size in cells per square (one side)
selected_bioclim_values = ['bio1', 'bio12'] # this selection is only applied for the CNN features, since loading all bioclim layers is very memory intense
occ_count_square_size = 10000 #define window edge length for counting gbif occurrences for NN features
# settings for label calculation_________________________________
n_neighbours = 50 # only for beta and gamma label_mode
beta_mode = 'sorensen'
target_family = None #'Fabaceae' #None # select taxonomic family for which to prepare div labels for
# training settings______________________________________________
seed = 1234
n_epochs = 5000
batch_size = 40
patience = 200
nodes = [30,15,5]
dropout = [0.0]
mc_dropout = False
n_train_instances = None
test_fraction = 0.2
val_fraction = 0.2
criterion = 'mae'
model_choice = 'nn'
output_actfun = 'softplus' # None, softplus, or sigmoid
selected_state_ids = None #[5,7], [5,6,7,8,10]
selected_channels = ['occs', 'div', 'land', 'hfp']
#selected_additional_features = ['bio_1','bio_12','elevation','hfp','plotsize','avgdist']
#selected_additional_features = ['n_occs','n_species','lon','lat','bio_1','bio_12','elevation','hfp','plotsize','avgdist']
selected_additional_features = None
#selected_additional_features = data_obj._add_feature_list
model_outdir = 'results/'
modes = ['alpha','beta','gamma']
modeltest = False
production_model = True
# modeltest settings_____________________________________________
cpus=6
if modeltest:
    if target_family is None:
        model_outdir = 'results/modeltesting/'
    else:
        model_outdir = 'results/modeltesting/%s' % target_family
    test_n_neighbours = [
        #30,
        50
    ]
    test_features = [
        ['bio_1','bio_12','elevation','hfp','plotsize','avgdist'],
        ['lon','lat','bio_1','bio_12','elevation','hfp','plotsize','avgdist'],
        None
    ]
    test_states = [
        #[5,7],
        #[5,7,8,10],
        #[5,6,7,8,10],
        None
    ]
    nodes = [
        [30],
        [30, 5],
        [30, 15, 5],
        [30, 20, 10, 5]
    ]
    dropout = [
        [0.0],
        [0.1],
        [0.3]
    ]
    modeltest_settings = [
        [seed],
        test_n_neighbours,
        test_features,
        test_states,
        nodes,
        dropout
    ]
    permutations = list(itertools.product(*modeltest_settings))
#_______________________________________________________________


# feature extraction ___________________________________________
# define input files
true_div_data_file = 'data/true_div_data/true_diversity_data_proj_aus.txt'
gbif_occs_file = 'data/gbif/australia_formatted_cleaned_proj.txt'
land_cells_file = 'data/land_vs_water_data/land_water_coords.txt'
climate_data_file = 'data/climate/land_cells_coords_climate.txt'
hfp_data_file = 'data/human_footprint/land_cells_coords_hfp.txt'
pred_data_file = 'data/prediction_data/prediction_cells_coords_res_10000.txt'

# run feature extraction
extract_nn_features = False
extract_cnn_features = False
if extract_nn_features or extract_cnn_features:
    # initialize a data_obj with a seed___________________________
    seed = 7288036
    data_obj = cust_func.prep_data(seed)

    # train data__________________________________________________
    # extract features for CNN
    coords = pd.read_csv(true_div_data_file, sep='\t')[['lon', 'lat']].values
    data_obj.extract_cnn_features_train(coords,
                                        window_length = window_length,
                                        n_cells_per_row = n_cells_per_row,
                                        gbif_occs_file = gbif_occs_file,
                                        land_cells_file = land_cells_file,
                                        climate_data_file = climate_data_file,
                                        hfp_data_file = hfp_data_file,
                                        select_bioclim_columns = selected_bioclim_values,
                                        extract_features=extract_cnn_features)
    # extract additional features for input to fully connected layers
    data_obj.extract_additional_features_train(coords,
                                               true_div_data_file,
                                               occ_count_square_size = occ_count_square_size,
                                               bioclim_ids = None,
                                               n_samples=n_instances,
                                               extract_features = extract_nn_features)

    # prediction data_____________________________________________
    pred_data = pd.read_csv(pred_data_file, sep='\t')
    pred_data_coords = pred_data[['lon', 'lat']].values
    pred_data_state_info = pred_data[['aus_state_info']].values
    pred_data_elevation = pred_data[['elevation']].values
    pred_data_hfp = pred_data[['hfp']].values
    pred_data_clim = pred_data[pred_data.columns[7:]].values

    # load the data files necessary for extracting features
    data_obj.extract_cnn_features_predict(pred_data_coords,
                                          extract_features=extract_cnn_features)
    data_obj.extract_additional_features_predict(pred_data_state_info=pred_data_state_info,
                                                 pred_data_elevation=pred_data_elevation,
                                                 pred_data_hfp=pred_data_hfp,
                                                 pred_data_clim=pred_data_clim,
                                                 extract_features=extract_nn_features)
    # save all objects to file
    data_obj.save_objects_to_file()
#_______________________________________________________________



# define the whole training and prediction workflow
def run_all(args):
    seed, n_neighbours, selected_additional_features, selected_state_ids, nodes, dropout = args
    # load the train features
    data_obj_path = 'data/cnn_input/l_%i_g_%i/data_obj.pkl' % (window_length, n_cells_per_row)
    data_obj = cust_func.load_obj(data_obj_path)
    # data_obj.load_cnn_features()

    # extract_labels
    data_obj.prep_labels_train(true_div_data_file,
                               target_family=target_family,
                               n_neighbours=n_neighbours,
                               beta_mode=beta_mode)

    # prep data for NN exercise
    data_out = np.hstack([data_obj._gamma_div,data_obj._additional_features])
    colnames = ['species_div'] + data_obj._add_feature_list
    # only export instances with reasonable avg_dist
    data_out = data_out[data_out[:,-1]<100000]
    df_out = pd.DataFrame(data_out, columns=colnames)
    target_cols = [i for i in df_out.columns if not 'state_info' in i]
    target_cols = [i for i in target_cols if not 'plotsize' in i]
    df_out_final = df_out[target_cols]
    df_out_final.to_csv(
        '/Users/tobiasandermann/Documents/teaching/ai_course_geosciences/biodiv_exercise/data/div_data_all_features.txt',
        sep='\t',
        index=False)

    # rescale features and labels
    # data_obj.scale_cnn_features()
    data_obj.scale_additional_features()
    data_obj.scale_train_labels(mode='custom',factor_list=[100,
                                                           1,
                                                           800])

    # iterate through label modes
    for label_mode in modes:
        # initialize a model
        model_obj = cust_func.model_obj(data_obj,seed)
        # define train and test data
        model_obj.select_train_and_test(test_fraction=test_fraction,
                                        validation_fraction=val_fraction,
                                        mode=label_mode,
                                        target_states=selected_state_ids,
                                        n_instances=n_instances,
                                        shuffle=True,
                                        select_cnn_features=selected_channels,
                                        select_additional_features=selected_additional_features)

        # build and train the model
        if model_choice == 'nn':
            ## build a NN model only using additional features and train
            model_obj.build_nn_model(   nodes=nodes,
                                        dropout=dropout,
                                        output_actfun=output_actfun,
                                        loss=criterion,
                                        mc_dropout=mc_dropout)
            model_obj.train_nn( epochs=n_epochs,
                                batch_size=batch_size,
                                patience=patience,
                                criterion=criterion,
                                outdir=model_outdir)
        elif model_choice == 'cnn':
            # build a CNN model and train
            model_obj.build_cnn_model(  filters=[5,1],
                                        kernels_conv=[5,3],
                                        conv_strides=[1,2],
                                        pool_size=[2,2],
                                        pool_strides=[1,2],
                                        nodes=[20,10],
                                        dropout=[0.3,0.1],
                                        pooling_strategy='max',
                                        padding='valid')
            #model_obj.build_cnn_model(filters=[6],nodes=[5,5],kernels_conv=(3,3),pool_size=(2,2),strides=(2,2),pooling_strategy='avg')
            #model_obj.build_cnn_model(filters=[3,1],nodes=[40,10],kernels_conv=(10,10),pool_size=(3,3),strides=(1,1),pooling_strategy='max')
            model_obj.train_cnn(epochs=200,
                                batch_size=50,
                                patience=10)

        # make plots
        model_obj.plot_train_history(criterion=criterion,save_to_file=True)
        model_obj.plot_prediction_accuracy('train')
        model_obj._model.summary()

        # estimate mean error
        if not production_model:
            model_obj.plot_prediction_accuracy('val')
            model_obj.plot_prediction_accuracy('test')
            model_obj.summarize_errors(error='mape')

        # load the prediction features and labels
        #data_obj.load_prediction_cnn_features

        # set the size of the predicted plots
        pred_plot_size = 500
        data_obj.update_pred_alpha_plotsize(pred_plot_size)
        pred_gamma_radius = 5000
        data_obj.update_pred_gamma_radius(pred_gamma_radius)

        # rescale features
        #data_obj.scale_pred_cnn_features()
        data_obj.scale_pred_additional_features()

        # make predictions for the prediction set
        pred_div = model_obj.make_predictions(  data_obj._cnn_input_data_pred,
                                                data_obj._additional_features_pred_rescaled,
                                                target_states = None,
                                                mcdropout_reps = 100)#list(model_obj._target_states)+[2])

        # plot the predicted div
        model_obj.plot_predicted_div(filename_add_string='plotsize_%i_avgdist_%i_%s'%(pred_plot_size,pred_gamma_radius,label_mode),
                                     pointsize=1)
                                     #width=5.8,
                                     #height=6.4,

# modeltesting__________________________________________________
if modeltest:
    pool = multiprocessing.Pool(cpus)
    args = permutations
    all_output = np.array(pool.map(partial(run_all), args))
    pool.close()
#_______________________________________________________________

# production model, using different seeds_______________________
if production_model:
    mc_dropout = False
    target_div_type = 1
    modes = ['alpha','beta','gamma']
    test_fraction = 0. #use all data for training
    val_fraction = 0. #use all data for training
    # select the settings for the chosen div type
    modes = [modes[target_div_type]]
    n_epochs = [1500, 750, 1700]
    n_epochs = n_epochs[target_div_type]
    patience = 0
    if mc_dropout:
        model_outdir = 'results/production_models/production_model_%s_mcdropout'%modes[0]
        dropout = [0.2, 0.1, 0.]
        args = [
            [seed, 50, ['lon','lat','bio_1','bio_12','elevation','hfp','plotsize','avgdist'], None, [30, 20, 10, 5], [0.1]],
            [seed, 50, None, None, [30, 15, 5], [0.1]],
            [seed, 50, None, None, [30, 15, 5], [0.0]]
                ]
        args = args[target_div_type]
        run_all(args)
    else:
        n_reps = 50
        test_seeds = list(np.random.choice(np.arange(9999999),n_reps,replace=False))
        model_outdir = 'results/production_models/production_model_%s_ensemble'%modes[0]
        # define settings for input
        args = [
            [test_seeds, [50], [['lon','lat','bio_1','bio_12','elevation','hfp','plotsize','avgdist']], [None], [[30, 5]], [[0.1]]],
            [test_seeds, [50], [None], [None], [[30, 15, 5]], [[0.0]]],
            [test_seeds, [50], [None], [None], [[30, 15, 5]], [[0.0]]]
                ]
        args = args[target_div_type]
        permutations = list(itertools.product(*args))
        if cpus == 1:
            all_output = [run_all(i) for i in permutations]
        else:
            pool = multiprocessing.Pool(cpus)
            args = permutations
            all_output = np.array(pool.map(partial(run_all), args))
            pool.close()

#_______________________________________________________________

plot_figure = False

data_obj_path = 'data/cnn_input/l_%i_g_%i/data_obj.pkl' % (window_length, n_cells_per_row)
data_obj = cust_func.load_obj(data_obj_path)
# extract_labels
data_obj.prep_labels_train(true_div_data_file,
                           target_family=target_family,
                           n_neighbours=n_neighbours,
                           beta_mode=beta_mode)

if plot_figure:
    markersize = 1
    # plot figure of beta_divs
    fig = plt.figure(figsize=(10, 2))

    subplot = fig.add_subplot(1, 3, 1)
    plt.scatter(data_obj._true_div_coords[:, 0],
                data_obj._true_div_coords[:, 1],
                c=data_obj._alpha_div,
                cmap='inferno_r',
                marker='s',
                edgecolors='none',
                s=markersize)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.title('Alpha diversity')

    subplot = fig.add_subplot(1, 3, 2)
    plt.scatter(data_obj._true_div_coords[:, 0],
                data_obj._true_div_coords[:, 1],
                c=data_obj._beta_div,
                cmap='inferno_r',
                marker='s',
                edgecolors='none',
                s=markersize)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.title('Beta diversity')

    subplot = fig.add_subplot(1, 3, 3)
    plt.scatter(data_obj._true_div_coords[:, 0],
                data_obj._true_div_coords[:, 1],
                c=data_obj._gamma_div,
                cmap='inferno_r',
                marker='s',
                edgecolors='none',
                s=markersize)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.title('Gamma diversity')

    fig.savefig('plots/training_labels.png',
                bbox_inches='tight',
                dpi=500,
                transparent=False)

    plt.show()