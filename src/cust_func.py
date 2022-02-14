import os
import ast
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cmocean
import itertools

class prep_data():
    def __init__(self, seed=None):
        if not seed:
            seed = np.random.randint(0,9999999)
        self._seed = seed
        self._labels_loaded = False
        self._cnn_features_loaded = False
        self._additional_features_loaded = False
        self._cnn_features_predict_loaded = False
        self._appended_plotsize_feature = False
        self._appended_avgdist_feature = False
        self._additional_features_predict_loaded = False
        self._gbif_occs_coords = None
        self._plotsize_feature_index = None
        self._avgdist_feature_index = None
        self._additional_features_scaled = None
        np.random.seed(self._seed)
        #TODO: is random seed preserved across the whole class or does it have to be redefined?

    def prep_labels_train(self,
                          true_div_data_file,
                          target_family=None,
                          n_neighbours=50,
                          beta_mode='sorenson'):
        self._true_div_data_file = true_div_data_file
        self._target_family = target_family
        self._n_neighbours = n_neighbours
        # load input data
        self.load_splot_data(target_family=self._target_family)
        self.get_beta_and_gamma_div_n_nearest_neighbours(self._n_neighbours,beta_mode=beta_mode)
        # append the plotsize and avgdist columns to the additional features
        self.append_additional_feature_columns()
        self._labels_loaded = True

    def append_additional_feature_columns(self):
        # plotsize
        if self._plotsize_feature_index is None:
            self._additional_features = np.hstack([self._additional_features,self._true_div_plotsize])
            self._plotsize_feature_index = self._additional_features.shape[1]-1
            self._add_feature_list.append('plotsize')
        else:
            self._additional_features[:,self._plotsize_feature_index] = self._true_div_plotsize
        # avgdist
        if self._avgdist_feature_index is None:
            self._additional_features = np.hstack([self._additional_features,self._avg_dist])
            self._avgdist_feature_index = self._additional_features.shape[1]-1
            self._add_feature_list.append('avgdist')
        else:
            self._additional_features[:,self._avgdist_feature_index] = self._avg_dist

    def extract_cnn_features_train(self,
                                   target_coords,
                                   window_length,
                                   n_cells_per_row,
                                   gbif_occs_file = None,
                                   land_cells_file = None,
                                   climate_data_file = None,
                                   hfp_data_file = None,
                                   select_bioclim_columns = None,
                                   extract_features=True):
        # read and store settings
        self._rescaled_cnn_features = False
        self._window_length = window_length
        self._n_cells_per_row = n_cells_per_row
        self._gbif_occs_file = gbif_occs_file
        self._land_cells_file = land_cells_file
        self._climate_data_file = climate_data_file
        self._hfp_data_file = hfp_data_file
        if select_bioclim_columns is None:
            self._select_bioclim_columns = []
        else:
            self._select_bioclim_columns = select_bioclim_columns
        # get channel names
        self.get_channel_list()
        if extract_features:
            self.load_input_data_to_memory(gbif=True,landinfo=True,climate=True,hfp=True)
            cnn_input_data = self.extract_cnn_input(target_coords)
            self._cnn_input_data = cnn_input_data
            self._cnn_features_loaded = True

    def extract_additional_features_train(self,
                                          target_coords,
                                          true_div_data_file,
                                          gbif_occs_features = True,
                                          lonlat_feature = True,
                                          aus_state_feature = True,
                                          elevation_feature = True,
                                          hfp_feature = True,
                                          bioclim_features = True,
                                          occ_count_square_size = 100000,
                                          possible_states = None,
                                          bioclim_ids = None,
                                          n_samples = None,
                                          extract_features=True):
        self._true_div_data_file = true_div_data_file
        self._gbif_occs_features = gbif_occs_features
        self._occ_count_square_size = occ_count_square_size
        self._lonlat_feature = lonlat_feature
        self._aus_state_feature = aus_state_feature
        self._elevation_feature = elevation_feature
        self._hfp_feature = hfp_feature
        self._bioclim_features = bioclim_features
        if bioclim_ids is None:
            bioclim_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        self._bioclim_ids = bioclim_ids
        if possible_states is None:
            self._possible_states = [2, 5, 6, 7, 8, 9, 10, 11, 999]
        else:
            self._possible_states = possible_states
        self._onehot_key_present = False  # this will be determined when encoding training data for the first time
        # get the list of feature names
        add_feature_list = self.get_feature_list()
        self._add_feature_list = add_feature_list
        # extract the features
        if extract_features:
            self.load_splot_data(extract_richness=False,n_samples=n_samples)
            if n_samples is not None:
                target_coords = target_coords[:n_samples]
            additional_features = self.extract_additional_features( target_coords,
                                                                    self._true_div_state_info.copy(),
                                                                    self._true_div_points_elevation.copy(),
                                                                    self._true_div_points_hfp.copy(),
                                                                    self._true_div_points_clim.copy())
            self._additional_features = additional_features
            self._additional_features_loaded = True

    def save_objects_to_file(self, outdir=None):
        # define output files
        if outdir is None:
            outdir = 'data/cnn_input/l_%i_g_%i' % (self._window_length, self._n_cells_per_row)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self._data_outdir = outdir
        pred_outdir = os.path.join(self._data_outdir,'pred')
        if not os.path.exists(pred_outdir):
            os.makedirs(pred_outdir)
        self._pred_data_outdir = pred_outdir
        cnn_array_path, data_obj_path = self.get_paths(self._data_outdir)
        self._cnn_features_path_train = cnn_array_path
        self._data_obj_path = data_obj_path
        if self._cnn_features_loaded:
            # save CNN features
            np.save(self._cnn_features_path_train, self._cnn_input_data)
        if self._cnn_features_predict_loaded:
            np.save(self._cnn_features_path_pred, self._cnn_input_data_pred)
        # empty memory
        self.purge_memory_large_objects() # remove unnecessary large objects from data_obj before saving
        # save data_object to file
        save_obj(self,self._data_obj_path)

    def get_beta_and_gamma_div_n_nearest_neighbours(self, n, beta_mode='sorensen'):
        # get coordinates and species lists of selected splot sites
        splot_coords = self._true_div_coords.copy()
        species_lists = self._species_lists.copy()
        # for each splot site determine the N closest neighbours
        beta_div_list = []
        gamma_div_list = []
        avg_dist_list = []
        for i, __ in enumerate(splot_coords):
            coords_p1 = splot_coords[i]
            coords_p2 = splot_coords
            distances = get_distance_between_points(coords_p1, coords_p2)
            neighbour_ids = np.argpartition(distances, n + 1)[:n + 1]  # these are not sorted!!! +1 because the 0-distance to itself
            # return avg distance to neighbours
            avg_dist = np.max(distances[neighbour_ids[neighbour_ids > 0]])
            # get species list of all neighbours
            selected_species_lists = species_lists[neighbour_ids]
            # get gamma div summarized across neighbours
            total_species_list = np.unique(np.concatenate(selected_species_lists))
            gamma_div = len(total_species_list)
            # get beta div as avg number of species differences to neighbours
            selected_point_species_list = selected_species_lists[np.where(neighbour_ids == i)[0][0]]
            if beta_mode == 'avg_diff':
                diffs = [len(list(set(spl) ^ set(selected_point_species_list))) for spl in selected_species_lists]
                beta_div = np.mean(diffs)
            elif beta_mode == 'sorensen':
                # b_ij = [len(set(selected_species_lists[i])-set(selected_species_lists[i+1])) if i <50 else len(set(selected_species_lists[i])-set(selected_species_lists[0])) for i in np.arange(len(selected_species_lists))]
                # b_ji = [len(set(selected_species_lists[i+1])-set(selected_species_lists[i])) if i <50 else len(set(selected_species_lists[0])-set(selected_species_lists[i])) for i in np.arange(len(selected_species_lists))]
                #ij_pairs = np.array(list(itertools.combinations(np.arange(len(selected_species_lists)), 2)))
                #b_ij = [len(set(selected_species_lists[i][0])-set(selected_species_lists[i][1])) for i in ij_pairs]
                #b_ji = [len(set(selected_species_lists[i][1])-set(selected_species_lists[i][0])) for i in ij_pairs]
                # b_ij = [len(set(selected_point_species_list)-set(splist)) for i, splist in enumerate(selected_species_lists)]
                # b_ji = [len(set(splist)-set(selected_point_species_list)) for i, splist in enumerate(selected_species_lists)]
                #beta_div = (component_b + component_c) / (2*component_a + component_b + component_c)
                # # a is overlap between species compositions
                # # b is set of species only present in neighbouring site
                # # c is set of species only present in focal site
                # a = np.array([len(list(set(spl) & set(selected_point_species_list))) for spl in selected_species_lists])
                # b = np.array([len(list(set(selected_point_species_list)-set(spl))) for spl in selected_species_lists])
                # c = np.array([len(list(set(spl) - set(selected_point_species_list))) for spl in selected_species_lists])
                s_t = gamma_div
                s_i = np.array([len(i) for i in selected_species_lists])
                b_ij = [len(set(selected_species_lists[i])-set(selected_point_species_list)) for i in np.arange(len(selected_species_lists))]
                b_ji = [len(set(selected_point_species_list)-set(selected_species_lists[i])) for i in np.arange(len(selected_species_lists))]
                component_a = np.sum(s_i)-s_t
                component_b = np.sum(np.min(np.vstack([b_ij,b_ji]),axis=0))
                component_c = np.sum(np.max(np.vstack([b_ij, b_ji]), axis=0))
                beta_div = (component_b + component_c)  / (2*component_a + component_b + component_c)
                # simpsons_index = a/(a+np.min([b,c],axis=0))
                # beta_div = np.mean(1-simpsons_index)
            elif beta_mode == 'whittaker':
                beta_div = gamma_div/self._alpha_div[i]
            # caveat: we lose some taxa here, if they were only identified to genus level (~10% of data)
            # identical genus names between and within splot sites will be seen as the same taxon and only counted once
            beta_div_list.append(beta_div)
            gamma_div_list.append(gamma_div)
            avg_dist_list.append(avg_dist)
        self._beta_div = np.array(beta_div_list).reshape(len(beta_div_list), 1)
        self._gamma_div = np.array(gamma_div_list).reshape(len(gamma_div_list),1)
        self._avg_dist = np.array(avg_dist_list).reshape(len(avg_dist_list),1)
        # save memory by setting these objects to None
        self._species_lists = None
        self._genus_lists = None

    def load_splot_data(self, target_family = None, extract_richness=True, n_samples = None):
        # load splot data
        true_div_points_df = pd.read_csv(self._true_div_data_file, sep='\t')
        if n_samples is not None:
            true_div_points_df = true_div_points_df[:n_samples]
        true_div_points_coords = true_div_points_df[['lon', 'lat']].values.astype(int)
        true_div_points_plotsize = true_div_points_df[['plotsize']].values.astype(int)
        true_div_points_state_info = true_div_points_df[['aus_state_info']].values.astype(int)
        true_div_points_elevation = true_div_points_df[['elevation']].values.astype(int)
        true_div_points_hfp = true_div_points_df[['hfp']].values.astype(float)
        true_div_points_clim = true_div_points_df[[i for i in true_div_points_df.columns if i.startswith('bio_')]].values.astype(float)
        self._true_div_coords = true_div_points_coords
        self._true_div_plotsize = true_div_points_plotsize
        self._true_div_state_info = true_div_points_state_info
        self._true_div_points_elevation = true_div_points_elevation
        self._true_div_points_hfp = true_div_points_hfp
        self._true_div_points_clim = true_div_points_clim
        if extract_richness:
            # get list of species and genera
            species_list_strings = true_div_points_df.species_list.values.astype(str)
            species_lists = np.array([np.array(i.split(', '), dtype=str) for i in species_list_strings], dtype=object)
            genus_lists = [[j.split(' ')[0] for j in i] for i in species_lists]
            if target_family is None:
                true_div_points_sr = true_div_points_df[['species_richness']].values.astype(int)
                self._species_lists = species_lists
            else:
                # get target taxa in specified family
                wcvp_file = 'data/wcvp/wcvp_v5_jun_2021.txt'
                wcvp_data = pd.read_csv(wcvp_file, sep='|')
                target_species_list, target_genus_list = get_list_of_species_and_genera_from_wcvp(wcvp_data, target_family)
                # count how many species belonging to genera of the target group were recorded
                target_ids = np.array([np.intersect1d(i, target_genus_list, return_indices=True)[1] for i in genus_lists],dtype=object)
                target_species_lists = np.array([np.array(species_list)[target_ids[i]] for i, species_list in enumerate(species_lists)],dtype=object)
                self._species_lists = target_species_lists
                n_target_matches_genus = np.array([len(i) for i in target_ids])
                # just in case some records are only identified to family level, extract those here
                n_target_matches_family = np.array([sum(np.array(i) == target_family) for i in genus_lists])
                div_in_target_group = n_target_matches_genus + n_target_matches_family
                true_div_points_sr = div_in_target_group.reshape(len(div_in_target_group),1)
            self._alpha_div = true_div_points_sr

    def load_input_data_to_memory(self,gbif=False,landinfo=False,climate=False,hfp=False):
        print('Loading data files into memory.')
        if self._gbif_occs_file is not None:
            if gbif:
                # laod gbif data
                gbif_occs_df = pd.read_csv(self._gbif_occs_file, sep='\t')
                gbif_occs_df_complete_rows = gbif_occs_df[~gbif_occs_df['species'].isnull()].copy()
                gbif_occs_df_complete_rows[['lon', 'lat']] = gbif_occs_df_complete_rows[['lon', 'lat']].astype(int)
                gbif_occs_coords = gbif_occs_df_complete_rows.values[:, 1:3].astype(int)
                gbif_occs_species = gbif_occs_df_complete_rows.values[:, 0].astype(str)
                self._gbif_occs_coords = gbif_occs_coords
                self._gbif_occs_species = gbif_occs_species
                del gbif_occs_coords
                del gbif_occs_species
        if self._land_cells_file is not None:
            if landinfo:
                # load land cell info
                land_cell_coords = pd.read_csv(self._land_cells_file, sep='\t').values
                self._land_cell_coords = land_cell_coords[:,:2]
                self._land_water_info = land_cell_coords[:,-1]
                del land_cell_coords
            else:
                self._land_cell_coords = None
                self._land_water_info = None
        # load climate data
        if self._climate_data_file is not None:
            if climate:
                if not self._select_bioclim_columns:
                    climate_data = pd.read_csv(self._climate_data_file, sep='\t')
                    bioclim_cols = list(climate_data.columns[2:].values)
                    climate_data = climate_data.values
                    self._climate_data_coords = climate_data[:, :2]
                    self._climate_data_values = climate_data[:,2:]
                    self._select_bioclim_columns = bioclim_cols
                    del climate_data
                else:
                    self._climate_data_coords = pd.read_csv(self._climate_data_file,sep='\t',usecols=['lon','lat']).values
                    self._climate_data_values = pd.read_csv(self._climate_data_file,sep='\t',usecols=self._select_bioclim_columns).values
                    self._select_bioclim_columns = list(self._select_bioclim_columns)
        else:
            self._climate_data_coords = None
            self._climate_data_values = None
            self._select_bioclim_columns = []
        # load human footprint data
        if self._hfp_data_file is not None:
            if hfp:
                hfp_data = pd.read_csv(self._hfp_data_file, sep='\t').values
                self._hfp_data_coords = hfp_data[:,:2]
                self._hfp_data_values = hfp_data[:,2:]
                del hfp_data
        else:
            self._hfp_data_coords = None
            self._hfp_data_values = None

    def get_channel_list(self):
        channel_list = []
        if self._gbif_occs_file is not None:
            channel_list.append('occs')
            channel_list.append('div')
        if self._land_cells_file is not None:
            channel_list.append('land')
        if self._climate_data_file is not None:
            if not self._select_bioclim_columns:
                bioclim_cols = list(pd.read_csv(self._climate_data_file, sep='\t',nrows=1).columns[2:].values)
            else:
                bioclim_cols = self._select_bioclim_columns
            for i in bioclim_cols:
                channel_list.append(i)
        if self._hfp_data_file is not None:
            channel_list.append('hfp')
        self._channel_list = channel_list

    def purge_memory_large_objects(self,
                                   gbif=True,
                                   landinfo=True,
                                   climate=True,
                                   hfp=True,
                                   cnn_features=True):
        if gbif:
            self._gbif_occs_coords = None
            self._gbif_occs_species = None
        if landinfo:
            self._land_cell_coords = None
            self._land_water_info = None
        if climate:
            self._climate_data_coords = None
            self._climate_data_values = None
        if hfp:
            self._hfp_data_coords = None
            self._hfp_data_values = None
        if cnn_features:
            self._cnn_input_data = None
            self._cnn_input_data_pred = None
            self._cnn_features_loaded = False
            self._cnn_features_predict_loaded = False

    def load_cnn_features(self,cnn_features=True):
        self._rescaled_cnn_features = False
        if cnn_features:
            self._cnn_input_data = np.load(self._cnn_features_path_train).astype(float)
            self._cnn_features_loaded = True

    def one_hot_encode(self, data, possible_states):
        if not self._onehot_key_present:
            # define a mapping of chars to integers
            char_to_int = dict((c, i) for i, c in enumerate(possible_states))
            int_to_char = dict((i, c) for i, c in enumerate(possible_states))
            self._onehot_encoding_key = char_to_int
            self._onehot_decoding_key = int_to_char
            self._onehot_key_present = True
        # integer encode input data
        integer_encoded = [self._onehot_encoding_key[char] for char in data]
        # one hot encode
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(possible_states))]
            letter[value] = 1
            onehot_encoded.append(letter)
        return np.array(onehot_encoded)
        ## invert encoding
        #inverted = int_to_char[np.argmax(onehot_encoded[0])]
        #print(inverted)

    def get_paths(self, indir):
        cnn_array_path = os.path.join(indir, 'cnn_features.npy')
        data_obj_path = os.path.join(indir, 'data_obj.pkl')
        return cnn_array_path,data_obj_path

    def plot_selected_occurrences(self, index=0):
        if self._gbif_occs_coords is None:
            self.load_splot_data(extract_richness=False)
            self.load_input_data_to_memory(gbif=True)
        div_point = self._true_div_coords[index]
        lon, lat = div_point  # cea coords are in meters
        square_coords = get_square_corner_points(lon, lat, self._window_length).astype(int)
        selected_occs_ids = find_occs_in_square(self._gbif_occs_coords, square_coords)
        fig = plt.figure(figsize=(8, 8))
        plt.plot(square_coords[0], square_coords[1], '.', color='black')
        plt.plot(square_coords[2], square_coords[3], '.', color='black')
        plt.plot(np.array([square_coords[0], square_coords[2], square_coords[2], square_coords[0], square_coords[0]]),
                 np.array([square_coords[1], square_coords[1], square_coords[3], square_coords[3], square_coords[1]]),
                 '-',
                 color='black')
        plt.scatter(lon, lat, marker='x', color='r', s=100, linewidths=2, zorder=3)
        plt.plot(self._gbif_occs_coords[selected_occs_ids][:, 0],
                 self._gbif_occs_coords[selected_occs_ids][:, 1], '.',
                 color='black')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        gridcells = split_square_in_grid_cells(square_coords, self._n_cells_per_row)
        for i, gridcell in enumerate(gridcells):
            selected_occs_ids_grid_cell = find_occs_in_square(self._gbif_occs_coords[selected_occs_ids], gridcell)
            selected_occs_ids_grid_cell_abs = selected_occs_ids[selected_occs_ids_grid_cell]
            plt.fill(np.array([gridcell[0],gridcell[2],gridcell[2],gridcell[0]]),
                     np.array([gridcell[1],gridcell[1],gridcell[3],gridcell[3]]),
                     '.',
                     color='C%i' % (i % 10),
                     alpha=0.2)
            plt.plot(self._gbif_occs_coords[selected_occs_ids_grid_cell_abs][:, 0],
                     self._gbif_occs_coords[selected_occs_ids_grid_cell_abs][:, 1],
                     '.',
                     color='C%i' % (i % 10))
        plt.show()
        self._point_plot = fig

    def plot_training_instance(self, index=0):
        img_data = np.array([
            self._cnn_input_data[index, :, :, 0].T,
            self._cnn_input_data[index, :, :, 1].T,
            self._cnn_input_data[index, :, :, 2].T
            ]).T
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(img_data)
        plt.show()
        self._instance_plot = fig

    def plot_all_channels(self, index=0, n_cols = 3, zoom_factor=1.0):
        n_channels = self._cnn_input_data.shape[-1]
        n_rows = int(np.ceil(n_channels / n_cols))
        fig = plt.figure(figsize=(n_cols*zoom_factor, n_rows*zoom_factor))
        for i in np.arange(n_channels):
            channel = self._cnn_input_data[index, :, :, i]
            subplot2 = fig.add_subplot(n_rows, n_cols, i + 1)
            img = plt.imshow(channel,cmap='binary',vmin=0,vmax=1)
            img.axes.get_xaxis().set_visible(False)
            img.axes.get_yaxis().set_visible(False)
            plt.title(self._channel_list[i])
        plt.show()
        self._channel_plot = fig

    def extract_cnn_features_predict(self, pred_data_coords, extract_features=True):
        self._pred_data_coords = pred_data_coords
        self._rescaled_pred = False
        if extract_features:
            self.load_input_data_to_memory(gbif=True,landinfo=True,climate=True,hfp=True)
            pred_cnn_input = self.extract_cnn_input(self._pred_data_coords)
            self._cnn_input_data_pred = pred_cnn_input
            self._cnn_features_predict_loaded = True

    def extract_additional_features_predict(self,
                                            pred_data_state_info=None,
                                            pred_data_elevation=None,
                                            pred_data_hfp=None,
                                            pred_data_clim=None,
                                            extract_features=True):
        self._pred_data_state_info = pred_data_state_info
        if extract_features:
            additional_features = self.extract_additional_features( self._pred_data_coords.copy(),
                                                                    pred_data_state_info.copy(),
                                                                    pred_data_elevation.copy(),
                                                                    pred_data_hfp.copy(),
                                                                    pred_data_clim.copy())
            self._additional_features_pred = additional_features
            self._additional_features_predict_loaded = True
            self._appended_plotsize_feature = False
            self._appended_avgdist_feature = False

    def load_prediction_cnn_features(self,target_dir=None):
        self._rescaled_pred = False
        self._cnn_input_data_pred = np.load(self._cnn_features_path_pred).astype(float)
        self._cnn_features_predict_loaded = True

    def get_paths_pred(self):
        cnn_array_path = os.path.join(self._pred_data_outdir, 'cnn_data_pred.npy')
        self._cnn_features_path_pred = cnn_array_path

    def scale_cnn_features( self,
                            log_occs=True,
                            log_sr=True,
                            log_hfp=True,
                            scale_factor_occs=None,
                            scale_factor_sr=None,
                            scale_factor_nland=None,
                            scale_factor_clim=None,
                            scale_factor_hfp=None):
        if not self._rescaled_cnn_features:
            self._log_occs = log_occs
            self._log_sr = log_sr
            self._log_hfp = log_hfp
            # determine scale factors, if not manually provided
            # CNN features
            if self._cnn_features_loaded:
                if not scale_factor_occs:
                    scale_factor_occs = [np.min(self._cnn_input_data[:, :, :, 0]),np.max(self._cnn_input_data[:, :, :, 0])]
                    self._use_manual_occ_scale_factor = False
                else:
                    self._use_manual_occ_scale_factor = True
                if not scale_factor_sr:
                    scale_factor_sr = [np.min(self._cnn_input_data[:, :, :, 1]),np.max(self._cnn_input_data[:, :, :, 1])]
                    self._use_manual_sr_scale_factor = False
                else:
                    self._use_manual_sr_scale_factor = True
                if not scale_factor_nland:
                    scale_factor_nland = [0,1] #int(np.max(self._cnn_input_data[:, :, :, 2]))
                if not scale_factor_clim:
                    scale_factor_clim = []
                    for i,__ in enumerate(self._select_bioclim_columns):
                        scale_factor_clim.append([np.min(self._cnn_input_data[:, :, :, 3+i]),np.max(self._cnn_input_data[:, :, :, 3+i])])
                if not scale_factor_hfp:
                    scale_factor_hfp = [np.min(self._cnn_input_data[:, :, :, -1]),np.max(self._cnn_input_data[:, :, :, -1])]
                    self._use_manual_hfp_scale_factor = False
                else:
                    self._use_manual_hfp_scale_factor = True
                self._scale_factor_occs = np.array(scale_factor_occs).astype(float)
                self._scale_factor_sr = np.array(scale_factor_sr).astype(float)
                self._scale_factor_nland = np.array(scale_factor_nland).astype(float)
                self._scale_factor_clim = np.array(scale_factor_clim).astype(float)
                self._scale_factor_hfp = np.array(scale_factor_hfp).astype(float)
            # apply the scaling factors
            scaled_cnn_input = self.apply_scaling_cnn_features(self._cnn_input_data)
            self._cnn_input_data = scaled_cnn_input
            self._rescaled_cnn_features = True
        else:
            print('CNN features have already been rescaled. Reload raw data in case you want to apply current rescaling settings.')

    def scale_additional_features(self,addfeat_factor=None):
        # additional features
        if self._additional_features_loaded:
            if addfeat_factor is None:
                addfeat_factor = np.array([np.min(self._additional_features,axis=0),np.max(self._additional_features,axis=0)]).T
                onehot_encoded_rows = [i for i, featname in enumerate(self._add_feature_list) if featname.startswith('state_info')]
                addfeat_factor[onehot_encoded_rows] = [0.,1.] # set the rescaling factor of one-hot-encoded features to [0,1]
        self._addfeat_factor = np.array(addfeat_factor).astype(float)
        # apply the scaling factors
        scaled_add_features = self.apply_scaling_additional_features(self._additional_features)
        self._additional_features_scaled = scaled_add_features

    def scale_train_labels(self,mode='custom',factor_list=[]):
        self._label_scale_mode = mode
        # scale the labels
        if self._label_scale_mode == 'log':
            self._label_scale_factor_alpha = None
            self._label_scale_factor_beta = None
            self._label_scale_factor_gamma = None
            self._alpha_div_scaled = np.log(self._alpha_div.copy())
            self._beta_div_scaled = np.log(self._beta_div.copy())
            self._gamma_div_scaled = np.log(self._gamma_div.copy())
        elif self._label_scale_mode == 'minmax':
            self._label_scale_factor_alpha = np.array([np.min(self._alpha_div),np.max(self._alpha_div)])
            self._label_scale_factor_beta = np.array([np.min(self._beta_div),np.max(self._beta_div)])
            self._label_scale_factor_gamma = np.array([np.min(self._gamma_div),np.max(self._gamma_div)])
            self._alpha_div_scaled = (self._alpha_div-self._label_scale_factor_alpha[0])/(self._label_scale_factor_alpha[1]-self._label_scale_factor_alpha[0])
            self._beta_div_scaled = (self._beta_div-self._label_scale_factor_beta[0])/(self._label_scale_factor_beta[1]-self._label_scale_factor_beta[0])
            self._gamma_div_scaled = (self._gamma_div-self._label_scale_factor_gamma[0])/(self._label_scale_factor_gamma[1]-self._label_scale_factor_gamma[0])
        elif self._label_scale_mode == 'custom':
            self._label_scale_factor_alpha = factor_list[0]
            self._label_scale_factor_beta = factor_list[1]
            self._label_scale_factor_gamma = factor_list[2]
            self._alpha_div_scaled = self._alpha_div.copy()/self._label_scale_factor_alpha
            self._beta_div_scaled = self._beta_div.copy()/self._label_scale_factor_beta
            self._gamma_div_scaled = self._gamma_div.copy()/self._label_scale_factor_gamma
        else:
            self._label_scale_factor_alpha = None
            self._label_scale_factor_beta = None
            self._label_scale_factor_gamma = None
            self._alpha_div_scaled = self._alpha_div.copy()
            self._beta_div_scaled = self._beta_div.copy()
            self._gamma_div_scaled = self._gamma_div.copy()

    def scale_pred_cnn_features(self):
        if not self._rescaled_pred:
            scaled_cnn_input = self.apply_scaling_cnn_features(self._cnn_input_data_pred)
            self._cnn_input_data_pred = scaled_cnn_input
            self._rescaled_pred = True
        else:
            print('Prediction data has already been rescaled.')

    def scale_pred_additional_features(self):
        scaled_add_features = self.apply_scaling_additional_features(self._additional_features_pred)
        self._additional_features_pred_rescaled = scaled_add_features

    def update_pred_alpha_plotsize(self,new_plotsize):
        if type(new_plotsize) is int:
            self._pred_plotsize = new_plotsize
            pred_data_plotsize = np.array([new_plotsize] * len(self._additional_features_pred))
        else: # if array of new plotsize values is provided
            pred_data_plotsize = np.array(new_plotsize)
            if len(np.unique(pred_data_plotsize)) > 1:
                self._pred_plotsize = 'custom'
            else:
                self._pred_plotsize = pred_data_plotsize[0]
        # rescale the values
        if self._rescaled_pred:
            pred_data_plotsize = (pred_data_plotsize-self._addfeat_factor[self._plotsize_feature_index,0])/(self._addfeat_factor[self._plotsize_feature_index,1] - self._addfeat_factor[self._plotsize_feature_index,0])
        if not self._appended_plotsize_feature:
            self._additional_features_pred = np.hstack([self._additional_features_pred,pred_data_plotsize.reshape(len(pred_data_plotsize),1)])
            self._appended_plotsize_feature = True
        else:
            self._additional_features_pred[:,self._plotsize_feature_index] = pred_data_plotsize

    def update_pred_gamma_radius(self,new_radius):
        if type(new_radius) is int:
            self._pred_avgdist = new_radius
            pred_data_avgdist = np.array([new_radius] * len(self._additional_features_pred))
        else: # if array of new radius values is provided
            pred_data_avgdist = new_radius
            if len(np.unique(pred_data_avgdist)) > 1:
                self._pred_avgdist = 'custom'
            else:
                self._pred_avgdist = pred_data_avgdist[0]
        # rescale the values
        if self._rescaled_pred:
            pred_data_avgdist = (pred_data_avgdist-self._addfeat_factor[self._avgdist_feature_index, 0])/(self._addfeat_factor[self._avgdist_feature_index, 1] - self._addfeat_factor[self._avgdist_feature_index, 0])
        if not self._appended_avgdist_feature:
            self._additional_features_pred = np.hstack([self._additional_features_pred,pred_data_avgdist.reshape(len(pred_data_avgdist),1)])
            self._appended_avgdist_feature = True
        else:
            self._additional_features_pred[:,self._avgdist_feature_index] = pred_data_avgdist

    def extract_cnn_input(self, coordinate_list):
        # for each point with species div data
        cnn_input_data = []
        for j, div_point in enumerate(coordinate_list):
            print('Extracting data for point %i/%i' % (j + 1, len(coordinate_list)), flush=True, end='\r')
            lon, lat = div_point  # cea coords are in meters
            square_coords = get_square_corner_points(lon, lat, self._window_length).astype(int)
            # make a pre-selection for faster processing
            if self._gbif_occs_file is not None:
                selected_occs_ids = find_occs_in_square(self._gbif_occs_coords, square_coords)
            if self._land_cells_file is not None:
                selected_landgrid_cell_ids = find_occs_in_square(self._land_cell_coords, square_coords)
            if self._climate_data_file is not None:
                selected_clim_coords_ids = find_occs_in_square(self._climate_data_coords, square_coords)
            if self._hfp_data_file is not None:
                selected_hfp_coords_ids = find_occs_in_square(self._hfp_data_coords, square_coords)
            gridcells = split_square_in_grid_cells(square_coords, self._n_cells_per_row)
            layers_all_gridcells = []
            for i, gridcell in enumerate(gridcells):
                layers = []
                if self._gbif_occs_file is not None:
                    selected_occs_ids_grid_cell = find_occs_in_square(self._gbif_occs_coords[selected_occs_ids],gridcell)
                    selected_occs_ids_grid_cell_abs = selected_occs_ids[selected_occs_ids_grid_cell]
                    # n_occs
                    n_occs_count = len(selected_occs_ids_grid_cell_abs)
                    layers.append(n_occs_count)
                    # n_species
                    n_species_count = len(np.unique(self._gbif_occs_species[selected_occs_ids_grid_cell_abs]))
                    layers.append(n_species_count)
                if self._land_cells_file is not None:
                    land_cells_ids_in_grid_cell = find_occs_in_square(self._land_cell_coords[selected_landgrid_cell_ids], gridcell)
                    land_cells_ids_in_grid_cell_abs = selected_landgrid_cell_ids[land_cells_ids_in_grid_cell]
                    if len(land_cells_ids_in_grid_cell_abs) == 0:
                        landcell_fraction = 0.0
                    else:
                        landcell_fraction = np.mean(self._land_water_info[land_cells_ids_in_grid_cell_abs])
                    layers.append(landcell_fraction)
                if self._climate_data_file is not None:
                    clim_coord_ids_in_grid_cell = find_occs_in_square(self._climate_data_coords[selected_clim_coords_ids], gridcell)
                    clim_coords_ids_in_grid_cell_abs = selected_clim_coords_ids[clim_coord_ids_in_grid_cell]
                    for bioclim_col in np.arange(self._climate_data_values.shape[1]):
                        clim_values = self._climate_data_values[clim_coords_ids_in_grid_cell_abs,bioclim_col]
                        if len(clim_values) > 0:
                            clim_value = np.mean(clim_values)
                        else:
                            clim_value = 0
                        layers.append(clim_value)
                if self._hfp_data_file is not None:
                    hfp_coords_ids_in_grid_cell = find_occs_in_square(self._hfp_data_coords[selected_hfp_coords_ids],gridcell)
                    hfp_coords_ids_in_grid_cell_abs = selected_hfp_coords_ids[hfp_coords_ids_in_grid_cell]
                    hfp_values = self._hfp_data_values[hfp_coords_ids_in_grid_cell_abs]
                    if len(hfp_values) > 0:
                        hfp_value = np.mean(hfp_values)
                    else:
                        hfp_value = 0
                    layers.append(hfp_value)
                layers_all_gridcells.append(layers)
            # reshape all channels from flat array to correct dimensions
            reshaped_arrays = []
            for channel in np.array(layers_all_gridcells).T:
                reshaped_arrays.append(channel.reshape([self._n_cells_per_row, self._n_cells_per_row]).T)
            # put all together into one joined array with multiple channels
            cnn_input_data.append(np.array(reshaped_arrays).T)
        cnn_input_data = np.array(cnn_input_data)
        return cnn_input_data

    def extract_additional_features(self,
                                    coords,
                                    state_info_values=None,
                                    elevation_values=None,
                                    hfp_values=None,
                                    bioclim_values=None):
        # create dummy array for additional features (this first column will be removed)
        add_features = np.zeros(len(coords)).reshape(len(coords),1)
        if self._gbif_occs_features:
            if self._gbif_occs_coords is None:
                self.load_input_data_to_memory(gbif=True)
            occs_values = []
            sr_values = []
            for j, point in enumerate(coords):
                print('Extracting additional features for point %i/%i' % (j + 1, len(coords)), flush=True, end='\r')
                lon, lat = point  # cea coords are in meters
                square_coords = get_square_corner_points(lon, lat, self._occ_count_square_size).astype(int)
                # make a pre-selection for faster processing
                #TODO: select only GBIF occurrences belonging to specified target family
                selected_occs_ids = find_occs_in_square(self._gbif_occs_coords, square_coords)
                # n occs
                n_occs_count = len(selected_occs_ids)
                occs_values.append(n_occs_count)
                # n species
                n_species_count = len(np.unique(self._gbif_occs_species[selected_occs_ids]))
                sr_values.append(n_species_count)
            add_features = np.hstack([add_features,np.array(occs_values).reshape(len(occs_values),1),np.array(sr_values).reshape(len(sr_values),1)])
        # add lon-lat features
        add_features = np.hstack([add_features,coords])
        # add one-hot-encoding of australia state features
        if self._aus_state_feature:
            onehot_encoded_state_features = self.one_hot_encode(state_info_values.flatten(), self._possible_states)
            add_features = np.hstack([add_features,onehot_encoded_state_features])
        # elevation
        if self._elevation_feature:
            add_features = np.hstack([add_features,elevation_values])
        # hfp
        if self._hfp_feature:
            add_features = np.hstack([add_features,hfp_values])
        # climate
        if self._bioclim_features:
            bioclim_values_selected = bioclim_values[:,list(np.array(self._bioclim_ids)-1)]
            add_features = np.hstack([add_features,bioclim_values_selected])
        additional_features = add_features[:,1:] # drop the first dummy column
        return (additional_features)

    def get_feature_list(self):
        add_feature_list = []
        if self._gbif_occs_features:
            add_feature_list.append('n_occs')
            add_feature_list.append('n_species')
        if self._lonlat_feature:
            add_feature_list.append('lon')
            add_feature_list.append('lat')
        if self._aus_state_feature:
            for state_info in ['state_info_%i'%i for i in self._possible_states]:
                add_feature_list.append(state_info)
        if self._elevation_feature:
            add_feature_list.append('elevation')
        if self._hfp_feature:
            add_feature_list.append('hfp')
        if self._bioclim_features:
            bioclim_names = ['bio_%i'%i for i in np.arange(19)+1]
            bioclim_names_selected = np.array(bioclim_names)[list(np.array(self._bioclim_ids)-1)]
            for i in bioclim_names_selected:
                add_feature_list.append(i)
        return add_feature_list

    def apply_scaling_cnn_features(self,cnn_features):
        image_array = cnn_features.copy().astype(float)
        # log-transform occ counts and div values
        if self._log_occs:
            image_array[:, :, :, 0] = np.log(image_array[:, :, :, 0]+1) # +1 to avoid -inf in log-space
            if not self._use_manual_occ_scale_factor:
                scale_factor_occs = [np.min(image_array[:, :, :, 0]), np.max(image_array[:, :, :, 0])]
                self._scale_factor_occs = np.array(scale_factor_occs).astype(float)
        if self._log_sr:
            image_array[:, :, :, 1] = np.log(image_array[:, :, :, 1]+1) # +1 to avoid -inf in log-space
            if not self._use_manual_occ_scale_factor:
                scale_factor_sr = [np.min(image_array[:, :, :, 1]), np.max(image_array[:, :, :, 1])]
                self._scale_factor_sr = np.array(scale_factor_sr).astype(float)
        if self._log_hfp:
            image_array[:, :, :, -1] = np.log(image_array[:, :, :, -1]+1) # +1 to avoid -inf in log-space
            if not self._use_manual_hfp_scale_factor:
                scale_factor_hfp = [np.min(image_array[:, :, :, -1]), np.max(image_array[:, :, :, -1])]
                self._scale_factor_hfp = np.array(scale_factor_hfp).astype(float)
        # apply scale-factors
        image_array[:, :, :, 0] = (image_array[:, :, :, 0] - self._scale_factor_occs[0]) / (self._scale_factor_occs[1] - self._scale_factor_occs[0])
        image_array[:, :, :, 1] = (image_array[:, :, :, 1] - self._scale_factor_sr[0]) / (self._scale_factor_sr[1] - self._scale_factor_sr[0])
        image_array[:, :, :, 2] = (image_array[:, :, :, 2] - self._scale_factor_nland[0]) / (self._scale_factor_nland[1] - self._scale_factor_nland[0])
        for i, __ in enumerate(self._select_bioclim_columns):
            image_array[:, :, :, 3+i] = (image_array[:, :, :, 3+i] - self._scale_factor_clim[i,0]) / (self._scale_factor_clim[i,1] - self._scale_factor_clim[i,0])
        image_array[:, :, :, -1] = (image_array[:, :, :, -1] - self._scale_factor_hfp[0]) / (self._scale_factor_hfp[1] - self._scale_factor_hfp[0])
        return(image_array)

    def apply_scaling_additional_features(self, additional_features):
        flat_features = additional_features.copy().astype(float)
        flat_features = (flat_features - self._addfeat_factor[:,0]) / (self._addfeat_factor[:,1] - self._addfeat_factor[:,0])
        return(flat_features)

class model_obj():
    def __init__(self,data_obj,seed=None):
        if not seed:
            seed = np.random.randint(0, 9999999)
        self._seed = seed
        tf.random.set_seed(self._seed)
        np.random.seed(self._seed)
        # load features and labels
        self._label_scale_factor_alpha = data_obj._label_scale_factor_alpha
        self._label_scale_factor_beta = data_obj._label_scale_factor_beta
        self._label_scale_factor_gamma = data_obj._label_scale_factor_gamma
        self._add_feature_list = data_obj._add_feature_list
        self._channel_list = data_obj._channel_list
        self._true_div_state_info = data_obj._true_div_state_info.copy()
        self._alpha_div = data_obj._alpha_div_scaled.copy()
        self._beta_div = data_obj._beta_div_scaled.copy()
        self._gamma_div = data_obj._gamma_div_scaled.copy()
        self._pred_data_state_info = data_obj._pred_data_state_info.copy()
        self._n_neighbours = data_obj._n_neighbours
        # read CNN features if present
        if data_obj._cnn_features_loaded:
            self._image_array = data_obj._cnn_input_data.copy().astype(float)
        # read additional features if present
        if data_obj._additional_features_loaded:
            self._additional_features = data_obj._additional_features_scaled.copy().astype(float)
            if len(self._additional_features) == 0:
                self._n_addfeatures = 0
            else:
                if len(self._additional_features.shape) == 1:
                    self._n_addfeatures = 1
                else:
                    self._n_addfeatures = self._additional_features.shape[1]
        self._label_scale_mode = data_obj._label_scale_mode
        self._pred_data_coords = data_obj._pred_data_coords
        self._plotsize_feature_index = data_obj._plotsize_feature_index
        self._avgdist_feature_index = data_obj._avgdist_feature_index
        # load other objects that are required
        self._addfeat_factor = data_obj._addfeat_factor
        self._cnn_features_present = data_obj._cnn_features_loaded
        self._additional_features_present = data_obj._additional_features_loaded

    def select_train_and_test(self,
                              test_fraction=0.2,
                              validation_fraction=0.2,
                              mode='gamma',
                              target_states = None,
                              n_instances = None,
                              shuffle=True,
                              select_cnn_features = None,
                              select_additional_features = None,
                              exclude_state_id = True):
        self._mode = mode
        if target_states is None:
            target_states = np.unique(self._true_div_state_info)
        self._target_states = target_states
        self._test_fraction = test_fraction
        self._validation_fraction = validation_fraction
        # get pool of indices to pick from
        # target state ids
        all_indices = np.arange(len(self._true_div_state_info))
        # select those of target states
        selected_indices = all_indices[np.in1d(self._true_div_state_info, self._target_states)]
        # select n instances
        if n_instances is not None:
            if n_instances > len(selected_indices):
                quit('Not enough instances. Reduce n_instances to max of %i.'%len(selected_indices))
            self._n_instances = self._n_instances
        else:
            self._n_instances = len(selected_indices)
        selected_indices = selected_indices[:self._n_instances]
        # shuffle these indices
        if shuffle:
            # shuffle all input data and labels
            np.random.seed(self._seed)
            shuffled_indices = np.random.choice(selected_indices, len(selected_indices), replace=False)
        else:
            shuffled_indices = selected_indices
        self._shuffled_indices = shuffled_indices
        # select train, validation, and test data
        n_test_instances = np.round(len(self._shuffled_indices)*self._test_fraction).astype(int)
        n_validation_instances = np.round(len(self._shuffled_indices)*self._validation_fraction).astype(int)
        test_ids = self._shuffled_indices[:n_test_instances]
        validation_ids = self._shuffled_indices[n_test_instances:n_test_instances+n_validation_instances]
        train_ids = self._shuffled_indices[n_test_instances+n_validation_instances:]
        self._n_test_ids = len(test_ids)
        self._n_val_ids = len(validation_ids)
        self._n_train_ids = len(train_ids)
        # labels
        if self._mode == 'alpha':
            labels = self._alpha_div.copy()
        elif self._mode == 'beta':
            labels = self._beta_div.copy()
        elif self._mode == 'gamma':
            labels = self._gamma_div.copy()
        labels_test = labels[test_ids]
        labels_val = labels[validation_ids]
        labels_train = labels[train_ids]
        # cnn features
        if self._cnn_features_present:
            image_array = self._image_array.copy()
            if select_cnn_features is not None:
                selected_cnn_feature_column_ids = [np.where(np.array(self._channel_list) == i)[0][0] for i in select_cnn_features]
                self._selected_cnn_channel_ids = selected_cnn_feature_column_ids
                self._selected_cnn_channel_names = select_cnn_features
                # select only defined channels
                image_array = image_array[:,:,:,self._selected_cnn_channel_ids]
            x_test = image_array[test_ids]
            x_val = image_array[validation_ids]
            x_train = image_array[train_ids]
        else:
            x_test = None
            x_val = None
            x_train = None
        # additional features
        if self._additional_features_present:
            additional_features = self._additional_features.copy()

            if select_additional_features is None:

                if exclude_state_id:
                    select_additional_features = [i for i in self._add_feature_list if not i.startswith('state_info')]
                    selected_additional_feature_column_ids = [np.where(np.array(self._add_feature_list) == i)[0][0] for i in select_additional_features]
                else:
                    select_additional_features = self._add_feature_list
                    selected_additional_feature_column_ids = list(np.arange(len(self._add_feature_list)))
                self._selected_additional_feature_names = select_additional_features
                self._selected_additional_feature_ids = selected_additional_feature_column_ids
            else:
                selected_additional_feature_column_ids = [np.where(np.array(self._add_feature_list) == i)[0][0] for i in select_additional_features]
                self._selected_additional_feature_ids = selected_additional_feature_column_ids
                self._selected_additional_feature_names = select_additional_features
            additional_features = additional_features[:,self._selected_additional_feature_ids]
            self._n_addfeatures = additional_features.shape[1] # update this to new number
            x2_test = additional_features[test_ids]
            x2_val = additional_features[validation_ids]
            x2_train = additional_features[train_ids]
        else:
            x2_test = None
            x2_val = None
            x2_train = None
        self._labels_test = labels_test
        self._labels_val = labels_val
        self._labels_train = labels_train
        self._x_test = x_test
        self._x_val = x_val
        self._x_train = x_train
        self._x2_test = x2_test
        self._x2_val = x2_val
        self._x2_train = x2_train

    def make_predictions(self, image_array_pred, additional_features_pred,target_states=None,mcdropout_reps=100):
        # select only instances from target areas, if defined
        try:
            total_pred_indices = additional_features_pred.shape[0]
        except:
            total_pred_indices = image_array_pred.shape[0]
        self._n_pred_indices = total_pred_indices
        all_indices = np.arange(self._n_pred_indices)
        if target_states is None:
            target_ids = all_indices
        else:
            target_ids = all_indices[np.in1d(self._pred_data_state_info, target_states)]
        self._pred_target_ids = target_ids
        # select specific data columns, if these were selected for training
        if self._cnn_features_present:
            image_array_pred = image_array_pred[self._pred_target_ids]
            image_array_pred = image_array_pred[:,:,:,self._selected_cnn_channel_ids]
        else:
            image_array_pred = None
        if self._additional_features_present:
            additional_features_pred = additional_features_pred[self._pred_target_ids]
            additional_features_pred = additional_features_pred[:,self._selected_additional_feature_ids]
        else:
            additional_features_pred = None
        if self._model_type == 'cnn':
            features = [image_array_pred,additional_features_pred]
        elif self._model_type == 'nn':
            features = additional_features_pred
        # load model if not already loaded
        if self._model is None:
            self._model = tf.keras.models.load_model(self._model_file)
        if self._mc_dropout:
            print('Predicting labels %i times using MC dropout...'%mcdropout_reps)
            pred = np.array([self._model.predict(features).flatten() for i in np.arange(mcdropout_reps)])
        else:
            pred = self._model.predict(features).flatten()
        scaled_pred = self.rescale_pred_div(pred)
        self._predicted_div = scaled_pred
        if self._mc_dropout:
            df_out = np.hstack([self._pred_data_coords[self._pred_target_ids],self._predicted_div.T])
            np.savetxt(os.path.join(self._model_dir,'predicted_labels_%s_div.txt'%self._mode),df_out,fmt=', '.join(['%i','%i']+['%.3f']*mcdropout_reps),delimiter='\t')
        else:
            df_out = np.hstack([self._pred_data_coords[self._pred_target_ids],self._predicted_div.reshape(len(self._predicted_div),1)])
            np.savetxt(os.path.join(self._model_dir,'predicted_labels_%s_div.txt'%self._mode),df_out,fmt='%i, %i, %.3f',delimiter='\t')
        return(scaled_pred)

    def rescale_pred_div(self,pred):
        if self._label_scale_mode == 'log':
            scaled_pred = np.round(np.exp(pred)).astype(int)
        elif self._label_scale_mode == 'minmax':
            if self._mode == 'alpha':
                scaled_pred = np.round(pred*(self._label_scale_factor_alpha[1]-self._label_scale_factor_alpha[0])+self._label_scale_factor_alpha[0]).astype(int)
            elif self._mode == 'beta':
                scaled_pred = np.round(pred*(self._label_scale_factor_beta[1]-self._label_scale_factor_beta[0])+self._label_scale_factor_beta[0],4).astype(float)
            elif self._mode == 'gamma':
                scaled_pred = np.round(pred*(self._label_scale_factor_gamma[1]-self._label_scale_factor_gamma[0])+self._label_scale_factor_gamma[0]).astype(int)
        elif self._label_scale_mode == 'custom':
            if self._mode == 'alpha':
                scaled_pred = np.round(pred*self._label_scale_factor_alpha).astype(int)
            elif self._mode == 'beta':
                scaled_pred = np.round(pred*self._label_scale_factor_beta,4).astype(float)
            elif self._mode == 'gamma':
                scaled_pred = np.round(pred*self._label_scale_factor_gamma).astype(int)

        else:
            scaled_pred = pred
        return scaled_pred

    def plot_predicted_div(self, div_predictions=None, outdir=None, width=9, height=5.9, pointsize=10, filename_add_string = ''):
        if outdir is None:
            outdir = self._model_dir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if div_predictions is None:
            div_predictions = self._predicted_div
        if self._mc_dropout:
            div_predictions = np.mean(div_predictions,axis=0)
        fig = plt.figure(figsize=(width, height))
        plt.scatter(self._pred_data_coords[self._pred_target_ids, 0],
                    self._pred_data_coords[self._pred_target_ids, 1],
                    c=div_predictions,
                    cmap='inferno_r',#'cmo.speed',
                    marker='s',
                    edgecolors='none',
                    s=pointsize)
        cbar = plt.colorbar()
        plt.axis('off')
        #cbar.set_label("Predicted species diversity", labelpad=+5)
        if len(filename_add_string) > 0:
            filename_add_string = '_'+filename_add_string
        fig.savefig(os.path.join(outdir,'australia_species_div_%s%s.png' %(self._model_type,filename_add_string)),
                    bbox_inches='tight',
                    dpi=500,
                    transparent=False)
        #plt.show()

    def build_cnn_model(self,
                        filters=[5,15],
                        kernels_conv=[5,3],
                        conv_strides=[1,2],
                        pool_size=[2,2],
                        pool_strides=[2,2],
                        nodes=[40,20],
                        dropout=[0.3,0.1],
                        padding='valid',
                        optimizer='adam',
                        pooling_strategy = 'max',
                        actfun = 'relu',
                        output_actfun = 'softplus',
                        use_bias=True):
        self._filters = filters
        self._nodes = nodes
        self._dropout = dropout
        self._kernels_conv = kernels_conv
        self._pool_size = pool_size
        self._conv_strides = conv_strides
        self._pool_strides = pool_strides
        self._padding = padding
        self._optimizer = optimizer
        self._pooling_strategy = pooling_strategy
        self._actfun = actfun
        self._output_actfun = output_actfun
        self._use_bias = use_bias
        architecture_conv = []
        for i,n_filt in enumerate(self._filters):
            architecture_conv.append(tf.keras.layers.Conv2D(filters=n_filt, kernel_size=self._kernels_conv[i], strides=self._conv_strides[i], activation=self._actfun, padding=self._padding,use_bias=self._use_bias))
            if self._pooling_strategy == 'avg':
                architecture_conv.append(tf.keras.layers.AveragePooling2D(pool_size=self._pool_size[i], strides=self._pool_strides[i], padding=self._padding))
            else:
                architecture_conv.append(tf.keras.layers.MaxPooling2D(pool_size=self._pool_size[i], strides=self._pool_strides[i], padding=self._padding))
        architecture_conv.append(tf.keras.layers.Flatten())
        conv_model = tf.keras.Sequential(architecture_conv)
        self._conv_model = conv_model
        # fully connected NN
        architecture_fc = []
        for i,n_nodes in enumerate(self._nodes):
            architecture_fc.append(tf.keras.layers.Dense(n_nodes, activation=self._actfun))
            architecture_fc.append(tf.keras.layers.Dropout(self._dropout[i]))
        architecture_fc.append(tf.keras.layers.Dense(1, activation=self._output_actfun))  # sigmoid or tanh or softplus
        fc_model = tf.keras.Sequential(architecture_fc)
        # define the input layer and apply the convolution part of the NN to it
        input1 = tf.keras.layers.Input(shape=self._x_train.shape[1:])
        cnn_output = conv_model(input1)
        # define the second input that will come in after the convolution
        input2 = tf.keras.layers.Input(shape=(self._n_addfeatures,))
        concatenatedFeatures = tf.keras.layers.Concatenate(axis=1)([cnn_output, input2])
        # output = fc_model(cnn_output)
        output = fc_model(concatenatedFeatures)
        model = tf.keras.models.Model([input1, input2], output)
        model.compile(loss='mean_squared_error', optimizer=self._optimizer, metrics=['mae', 'mse'])
        model.summary()
        self._model = model

    def train_cnn(self, epochs = 200, patience = 0, batch_size = 30, criterion = 'mae'):
        tf.random.set_seed(self._seed)
        np.random.seed(self._seed)
        self._epochs = epochs
        self._patience = patience
        self._batch_size = batch_size
        self._criterion = criterion
        if self._patience > 0:
            early_stop = keras.callbacks.EarlyStopping(monitor='val_%s'%self._criterion,
                                                       patience=self._patience,
                                                       restore_best_weights=True)
            history = self._model.fit( [self._x_train,self._x2_train],
                                        self._labels_train,
                                        epochs=self._epochs,
                                        validation_data=([self._x_val,self._x2_val],self._labels_val),
                                        verbose=1,
                                        callbacks=[early_stop],
                                        batch_size=self._batch_size)
        else:
            history = self._model.fit( [self._x_train,self._x2_train],
                                        self._labels_train,
                                        epochs=self._epochs,
                                        validation_data=([self._x_val,self._x2_val],self._labels_val),
                                        verbose=1,
                                        batch_size=self._batch_size)
        self._train_history = history.history
        self._model_type = 'cnn'
        self.plot_train_history(save_to_file=False)

    def build_nn_model(self,
                       nodes=[40,20],
                       dropout=[0.1],
                       optimizer='adam',
                       actfun = 'relu',
                       output_actfun = 'softplus',
                       loss = 'mae',
                       mc_dropout = False):
        self._nodes = nodes
        self._dropout = dropout
        self._optimizer = optimizer
        self._actfun = actfun
        self._output_actfun = output_actfun
        self._loss = loss
        self._mc_dropout = mc_dropout
        # fully connected NN
        architecture = [tf.keras.layers.Flatten(input_shape=[self._n_addfeatures])]
        for i,nnodes in enumerate(self._nodes):
            architecture.append(tf.keras.layers.Dense(nnodes, activation=self._actfun))
            if self._mc_dropout:
                if len(self._dropout)==1:
                    architecture.append(MCDropout(self._dropout[0]))
                else:
                    architecture.append(MCDropout(self._dropout[i]))
            else:
                if len(self._dropout) == 1:
                    architecture.append(tf.keras.layers.Dropout(self._dropout[0]))
                else:
                    architecture.append(tf.keras.layers.Dropout(self._dropout[i]))
        architecture.append(tf.keras.layers.Dense(1, activation=self._output_actfun))  # sigmoid or tanh or softplus
        model = tf.keras.Sequential(architecture)
        model.compile(loss=self._loss,
                      optimizer=self._optimizer,
                      metrics=['mae',
                               'mape',
                               'mse',
                               'msle'])
                      # metrics=[tf.keras.losses.MeanAbsoluteError(),
                      #          tf.keras.losses.MeanAbsolutePercentageError(),
                      #          tf.keras.losses.MeanSquaredError(),
                      #          tf.keras.losses.MeanSquaredLogarithmicError()])
        model.summary()
        self._model = model

    def train_nn(self,
                 epochs = 200,
                 patience = 0,
                 batch_size = 30,
                 criterion = 'mae',
                 outdir=''):
        tf.random.set_seed(self._seed)
        np.random.seed(self._seed)
        self._epochs = epochs
        if patience == 0:
            patience = epochs
        self._patience = patience
        self._batch_size = batch_size
        self._criterion = criterion
        self._model_type = 'nn'
        if self._n_val_ids == 0:
            early_stop = keras.callbacks.EarlyStopping(monitor='%s' % self._criterion,
                                                       patience=self._patience,
                                                       restore_best_weights=True)
        else:
            early_stop = keras.callbacks.EarlyStopping(monitor='val_%s' % self._criterion,
                                                       patience=self._patience,
                                                       restore_best_weights=True)
        history = self._model.fit(  self._x2_train,
                                    self._labels_train,
                                    epochs=self._epochs,
                                    validation_data=(self._x2_val,self._labels_val),
                                    verbose=1,
                                    callbacks=[early_stop],
                                    batch_size=self._batch_size)
        self._train_history = history.history
        self.plot_train_history(save_to_file=False)
        self.save_model(outdir=outdir)

    def save_model(self, outdir=''):
        self._outdir = outdir
        model_name_string = '%s_%i_%i_states_%i_features_%i_neighbours_%s_nodes_%s_dropout_%s_%i_batchsize_%i_epochs_%s'%(self._model_type,self._seed,len(self._target_states),len(self._selected_additional_feature_names),self._n_neighbours,'_'.join(np.array(self._nodes,dtype=str)),'_'.join(np.array(self._dropout,dtype=str)),self._output_actfun,self._batch_size,self._epochs,self._mode)
        self._model_dir = os.path.join(self._outdir,model_name_string)
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        # save the model separately, since it can't be pickled
        self._model_file = os.path.join(self._model_dir,'tensorflow_model')
        self._model.save(self._model_file)
        self._model = None
        # save the model_obj to pkl file
        self._model_pkl = os.path.join(self._model_dir,'model_obj.pkl')
        save_obj(self, self._model_pkl)

    def summarize_errors(self, error='mape'):
        # train
        pred_error_train = self.get_prediction_error('train', rescale=False, error=error)
        pred_error_train_scaled = self.get_prediction_error('train', rescale=True, error=error)
        print('Mean error on train set:', np.round(np.mean(pred_error_train), 4))
        print('Mean error on train set (scaled):', np.round(np.mean(pred_error_train_scaled), 4))
        # val
        pred_error_val = self.get_prediction_error('val', rescale=False, error=error)
        pred_error_val_scaled = self.get_prediction_error('val', rescale=True, error=error)
        print('Mean error on val set:', np.round(np.mean(pred_error_val), 4))
        print('Mean error on val set (scaled):', np.round(np.mean(pred_error_val_scaled), 4))
        # test
        pred_error_test = self.get_prediction_error('test', rescale=False, error=error)
        pred_error_test_scaled = self.get_prediction_error('test', rescale=True, error=error)
        print('Mean error on test set:', np.round(np.mean(pred_error_test), 4))
        print('Mean error on test set (scaled):', np.round(np.mean(pred_error_test_scaled), 4))
        # save to file
        np.savetxt(os.path.join(self._model_dir, 'train_val_test_%s_unscaled_scaled.txt'%error),
                   np.array([[pred_error_train, pred_error_val, pred_error_test],
                             [pred_error_train_scaled, pred_error_val_scaled, pred_error_test_scaled]]).T,
                   fmt='%.4f')

    def plot_train_history(self,criterion=None,save_to_file=True):
        if criterion is None:
            crit_string = self._criterion
        else:
            crit_string = criterion
        fig = plt.figure(figsize=(8, 5))
        plt.plot(self._train_history['%s'%crit_string],label='Training set')
        try:
            plt.plot(self._train_history['val_%s'%crit_string],label='Validation set')
        except:
            print('No validation set found, only plotting training set history.')
        plt.xlabel('Epochs')
        plt.ylabel(crit_string.upper())
        plt.ylim([self._train_history['%s'%crit_string][-1]-0.02*self._train_history['%s'%crit_string][-1],self._train_history['%s'%crit_string][5]])
        if not self._patience == self._epochs: # if there is an active patience parameter
            best_epoch = np.where(self._train_history['val_%s'%self._criterion] == np.min(self._train_history['val_%s'%self._criterion]))[0][0]
            plt.axvline(best_epoch,c='grey',linestyle='--')
            plt.axhline(self._train_history['val_%s'%self._criterion][best_epoch], c='grey', linestyle='--')
            plt.gca().axvspan(best_epoch,len(self._train_history['%s'%crit_string]),color='grey',alpha=0.3,zorder=3)
        plt.grid()
        plt.legend(loc='upper center')
        if save_to_file:
            fig.savefig(os.path.join(self._model_dir,'train_history.pdf'),
                        bbox_inches = 'tight',
                        dpi = 500,
                        transparent = True)
        if not save_to_file:
            plt.show()

    def get_true_vs_pred_arrays(self, get_this_set='test',rescale=True):
        if get_this_set == 'train':
            x1 = self._x_train
            x2 = self._x2_train
            lab = self._labels_train
        elif get_this_set == 'val':
            x1 = self._x_val
            x2 = self._x2_val
            lab = self._labels_val
        elif get_this_set == 'test':
            x1 = self._x_test
            x2 = self._x2_test
            lab = self._labels_test
        # run the prediction
        # load model if not already loaded
        if self._model is None:
            self._model = tf.keras.models.load_model(self._model_file)
        if self._model_type == 'cnn':
            pred = self._model.predict([x1, x2]).flatten()
        elif self._model_type == 'nn':
            features = x2
            pred = self._model.predict(features).flatten()
        if rescale:
            pred = self.rescale_pred_div(pred)
            lab = self.rescale_pred_div(lab)
        # determine plotsize feature column in selected df
        matches = np.where(np.array(self._selected_additional_feature_names) == 'plotsize')[0]
        if len(matches) == 0:
            plot_size_feature_index = None
            plotsize_array = np.array([0] * len(x2))
        else:
            plot_size_feature_index = matches[0]
            x_prime = x2[:,plot_size_feature_index]
            y = self._addfeat_factor[self._plotsize_feature_index,0]
            z = self._addfeat_factor[self._plotsize_feature_index,1]
            plotsize_array = np.round(x_prime*z-x_prime*y+y).astype(int)
        return(lab.flatten(),pred,plotsize_array)

    def get_prediction_error(self, get_this_set='test', rescale=False, error='mape'):
        true_div, pred_div, __ = self.get_true_vs_pred_arrays(get_this_set,rescale=rescale)
        if error == 'mape':
            error = np.mean(np.abs((pred_div - true_div)/true_div)) #mape
        else:
            error = np.mean(np.abs(pred_div - true_div)) #mae
        return error

    def plot_prediction_accuracy(self, plot_this_set='test'):
        # find the main three categories of plotsize to color separately in scatter plot
        # Set up the axes with gridspec
        fig = plt.figure(figsize=(8, 8))
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
        main_ax = fig.add_subplot(grid[1:, :-1])
        x_hist = fig.add_subplot(grid[0, :-1], xticklabels=[])
        y_hist = fig.add_subplot(grid[1:, -1], yticklabels=[])
        # get the data
        true_div, pred_div, plotsize_array = self.get_true_vs_pred_arrays(plot_this_set)
        plotsize, counts = np.unique(plotsize_array, return_counts=True)
        if len(plotsize[counts>20]) < 3:
            # scatter points on the main axes
            main_ax.plot(true_div, pred_div, 'o', markersize=3, alpha=1, label='Other')
        else:
            top_three_counts = np.sort(counts)[-3:]
            top_three_indices = [np.where(counts == i)[0][0] for i in top_three_counts]
            top_three_plotsizes = plotsize[top_three_indices]
            plotsize_ids = [np.where(plotsize_array == i)[0] for i in top_three_plotsizes]
            # scatter points on the main axes
            main_ax.plot(true_div, pred_div, 'o', markersize=3, color='grey', alpha=0.4, label='Other')
            main_ax.plot(true_div[plotsize_ids[0]], pred_div[plotsize_ids[0]], 'o', markersize=3, alpha=0.4,
                         label='%i m2' % top_three_plotsizes[0])
            main_ax.plot(true_div[plotsize_ids[1]], pred_div[plotsize_ids[1]], 'o', markersize=3, alpha=0.4,
                         label='%i m2' % top_three_plotsizes[1])
            main_ax.plot(true_div[plotsize_ids[2]], pred_div[plotsize_ids[2]], 'o', markersize=3, alpha=0.4,
                         label='%i m2' % top_three_plotsizes[2])
            leg = main_ax.legend(title='Plot size', loc='lower right')
            for lh in leg.legendHandles:
                lh._legmarker.set_alpha(1)
        #xlim = main_ax.get_xlim()
        #ylim = main_ax.get_ylim()
        min_val = np.min(np.concatenate([true_div.flatten(),pred_div]))
        max_val = np.max(np.concatenate([true_div.flatten(),pred_div]))
        buffer = 0.05*(max_val-min_val)
        main_ax.set_xlim(min_val-buffer,max_val+buffer)
        main_ax.set_ylim(min_val-buffer,max_val+buffer)
        main_ax.autoscale(False)
        # main_ax.set_xlim(xlim)
        main_ax.plot([0, main_ax.get_xlim()[-1]], [0, main_ax.get_xlim()[-1]], color='red', linestyle='--')
        main_ax.grid()
        main_ax.set_xlabel('True diversity')
        main_ax.set_ylabel('Predicted diversity')
        # histogram on the attached axes
        x_hist.hist(true_div, 40, orientation='vertical', color='gray')
        #x_hist.set_ylim(min_val-buffer,max_val+buffer)
        x_hist.grid(axis='x')
        y_hist.hist(pred_div, 40, orientation='horizontal', color='gray')
        y_hist.grid(axis='y')
        # scale the axes of the histograms
        max_hist_value = np.max(x_hist.get_ylim()+y_hist.get_xlim())
        hist_buffer = 0.05*max_hist_value
        x_hist.set_xlim(min_val-buffer,max_val+buffer)
        x_hist.set_ylim(0,max_hist_value+hist_buffer)
        y_hist.set_xlim(0,max_hist_value+hist_buffer)
        y_hist.set_ylim(min_val-buffer,max_val+buffer)
        fig.savefig(os.path.join(self._model_dir,'%s_prediction_accuracy.pdf'%plot_this_set),
                    bbox_inches = 'tight',
                    dpi = 500,
                    transparent = True)
        #plt.show()

def get_square_corner_points(center_lon,center_lat,square_side_length):
    # coordinates in clockwise order starting top left
    lon = center_lon
    lat = center_lat
    l = square_side_length
    square_coords = np.array([lon-0.5*l,lat+0.5*l,lon+0.5*l,lat-0.5*l])
    #    square_coords = np.array([(lon-0.5*l,lat+0.5*l),
#                              (lon+0.5*l,lat+0.5*l),
#                              (lon+0.5*l,lat-0.5*l),
#                              (lon-0.5*l,lat-0.5*l)])
    return square_coords

def split_square_in_grid_cells(square_coords,n_cells_per_row):
    lons = np.linspace(square_coords[0],square_coords[2],n_cells_per_row+1)
    lats = np.linspace(square_coords[1],square_coords[3],n_cells_per_row+1)
    #x,y = np.meshgrid(np.arange(len(lons)),np.arange(len(lats)))
    x, y = np.meshgrid(lons, lats)
    coords = np.dstack((x, y)).reshape(len(lons)*len(lats),2)
    indices = np.array([np.array([0, n_cells_per_row+2]) + i for i in np.arange(n_cells_per_row*n_cells_per_row+n_cells_per_row)])
    indices = np.delete(indices, np.arange(n_cells_per_row, indices.shape[0], n_cells_per_row+1),axis=0)
    corner_coords = np.array([coords[i] for i in indices]).reshape(n_cells_per_row*n_cells_per_row,4)
    return(corner_coords)

def find_occs_in_square(input_coords,square_coords):
    x1, y1, x2, y2 = square_coords
    x = input_coords[:,0]
    y = input_coords[:,1]
    valid_x_ids = np.where(np.logical_and(x1 <= x, x < x2))[0]
    valid_ids_rel = np.where(np.logical_and(y1 >= y[valid_x_ids], y[valid_x_ids] > y2))[0]
    valid_ids_abs = valid_x_ids[valid_ids_rel]
    return valid_ids_abs

def save_obj(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def get_list_of_species_and_genera_from_wcvp(wcvp_data,target_family):
    wcvp_data_species = wcvp_data[wcvp_data['rank'] == 'SPECIES']
    wcvp_data_species_accepted = wcvp_data_species[wcvp_data_species.taxonomic_status == 'Accepted']
    target_rows = wcvp_data_species_accepted[wcvp_data_species_accepted.family.values == target_family]
    target_species_list = (target_rows.genus + ' ' + target_rows.species).values.astype(str)
    target_species_list = np.array([i for i in target_species_list if not i.startswith('xx')])
    target_genus_list = np.unique(target_rows.genus).astype(str)
    target_genus_list = np.delete(target_genus_list, np.where(target_genus_list == 'xx'))
    return target_species_list,target_genus_list

def get_distance_between_points(coords_p1,coords_p2):
    if len(coords_p1.shape)==1: # if 1D array
        coords_p1 = coords_p1.reshape(1,2)
    if len(coords_p2.shape)==1: # if 1D array
        coords_p2 = coords_p2.reshape(1,2)
    x1 = coords_p1[:,0]
    y1 = coords_p1[:,1]
    x2 = coords_p2[:,0]
    y2 = coords_p2[:,1]
    distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return distance

class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


'''
def split_square_in_grid_cells_old(square_coords,n_cells_per_row):
    # get evenly spaced lon and lat points
    lons = np.linspace(square_coords[0,0],square_coords[1,0],n_cells_per_row+1)
    lats = np.linspace(square_coords[2,1],square_coords[1,1],n_cells_per_row+1)
    # create indices for meshgrid
    lon_ids,lat_ids = np.meshgrid(np.arange(n_cells_per_row+1),np.arange(n_cells_per_row+1))
    # select so we always get 4 corner coords for each gridcell
    x1 = np.arange(n_cells_per_row)
    x2 = x1+2
    index_list = np.array(list(zip(x1,x2)))
    # iterate through grid cells and store corner points
    grid_cell_squares = []
    for i in np.arange(len(index_list)):
        inverse_pair = index_list[-(i + 1)]
        for index_pair in index_list:
            #print(index_pair,inverse_pair)
            a = lons[lon_ids[index_pair[0]:index_pair[1],index_pair[0]:index_pair[1]]].flatten()[[2,3,1,0]]
            b = lats[lat_ids[inverse_pair[0]:inverse_pair[1],inverse_pair[0]:inverse_pair[1]]].flatten()[[2,3,1,0]]
            gridcell_coords = np.array([a,b]).T
            grid_cell_squares.append(gridcell_coords)
    # plt.plot(lons[lon_ids],lats[lat_ids],'.',color='blue')
    # plt.plot(square_coords[:, 0], square_coords[:, 1], 'r.')
    # grid_cell = grid_cell_squares[1]
    # plt.plot(grid_cell[:,0],grid_cell[:,1], 'g.')
    # plt.plot(grid_cell[:, 0][3], grid_cell[:, 1][3], 'r.')
    return grid_cell_squares

def find_occs_in_square_old(input_coords,square_coords):
    x1, y1 = square_coords[3]
    x2, y2 = square_coords[1]
    x = input_coords[:,0]
    y = input_coords[:,1]
    valid_x_ids = np.where(np.logical_and(x1 < x, x < x2))[0]
    valid_ids_rel = np.where(np.logical_and(y1 < y[valid_x_ids], y[valid_x_ids] < y2))[0]
    valid_ids_abs = valid_x_ids[valid_ids_rel]
    return valid_ids_abs
'''