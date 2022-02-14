import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def get_square_corner_points(center_lon,center_lat,square_side_length):
    # coordinates in clockwise order starting top left
    lon = center_lon
    lat = center_lat
    l = square_side_length
    square_coords = np.array([(lon-0.5*l,lat+0.5*l),
                              (lon+0.5*l,lat+0.5*l),
                              (lon+0.5*l,lat-0.5*l),
                              (lon-0.5*l,lat-0.5*l)])
    return square_coords

def split_square_in_grid_cells(square_coords,n_cells_per_row):
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


def find_occs_in_square(input_coords,square_coords):
    x1, y1 = square_coords[3]
    x2, y2 = square_coords[1]
    x = input_coords[:,0]
    y = input_coords[:,1]
    valid_x_ids = np.where(np.logical_and(x1 < x, x < x2))[0]
    valid_ids_rel = np.where(np.logical_and(y1 < y[valid_x_ids], y[valid_x_ids] < y2))[0]
    valid_ids_abs = valid_x_ids[valid_ids_rel]
    return valid_ids_abs


#def find_occs_in_square(input_coords,square_coords):
#    occs_lon = input_coords[:,0]
#    occs_lat = input_coords[:,1]
#    a = np.where(occs_lon>=square_coords[0,0])[0]
#    select_index = a
#    b = np.where(occs_lon[select_index]<square_coords[1,0])[0]
#    select_index = select_index[b]
#    c = np.where(occs_lat[select_index]>=square_coords[2,1])[0]
#    select_index = select_index[c]
#    d = np.where(occs_lat[select_index]<square_coords[1,1])[0]
#    select_index = select_index[d]
#    return select_index


def build_cnn_model(kernels_conv,pool_size,filters,optimizer,pooling_strategy = 'avg',output_actfun = 'softplus'):
    architecture = []
    # # ADD CONVOLUTION
    architecture.append(tf.keras.layers.Conv2D(filters=filters,
                                               kernel_size=kernels_conv,
                                               activation='relu',
                                               padding='valid',
                                               input_shape=cnn_input_data.shape[1:]))
    if pooling_strategy == 'avg':
        architecture.append(tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                                                             strides=(1, 1),
                                                             padding='same'))
    else:
        architecture.append(tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                                        strides=(1, 1),
                                                        padding='same'))
    # fully connected layers
    architecture.append(tf.keras.layers.Flatten())
    architecture.append(tf.keras.layers.Dense(40, activation='relu'))
    architecture.append(tf.keras.layers.Dense(20, activation='relu'))
    # output layer
    architecture.append(tf.keras.layers.Dense(1, activation=output_actfun))  # sigmoid or tanh or softplus
    # compile model
    model = tf.keras.Sequential(architecture)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


# define input files
true_div_data_file = 'data/true_div_data/true_diversity_data_proj_aus.txt'
gbif_occs_file = 'data/gbif/australia_formatted_cleaned_proj.txt'
land_cells_file = 'data/land_vs_water_data/land_cells_coords.txt'

# read files and transform to right format
# ground truth data (splots)
true_div_points_df = pd.read_csv(true_div_data_file,sep='\t')
true_div_points_coords = true_div_points_df.values[:,:2].astype(int)
true_div_points_plotsize = true_div_points_df.values[:,2].astype(int)
true_div_points_sr = true_div_points_df.values[:,3].astype(int)
# gbif occs for features
gbif_occs_df = pd.read_csv(gbif_occs_file,sep='\t')
gbif_occs_df_complete_rows = gbif_occs_df[~gbif_occs_df['species'].isnull()].copy()
gbif_occs_df_complete_rows[['lon','lat']] = gbif_occs_df_complete_rows[['lon','lat']].astype(int)
gbif_occs_coords = gbif_occs_df_complete_rows.values[:,1:3].astype(int)
gbif_occs_species = gbif_occs_df_complete_rows.values[:,0].astype(str)
# land cell info
land_cell_coords = pd.read_csv(land_cells_file,sep='\t').values

# define side-length of square in m
l = 200000
# define grid-cell size in cells per square (one side)
g = 8

grid_plotting = False

# define output files
outdir = 'data/cnn_input/l_%i_g_%i'%(l,g)
if not os.path.exists(outdir):
    os.makedirs(outdir)
cnn_array_path = os.path.join(outdir,'cnn_data.npy')
features_path = os.path.join(outdir,'extra_features.npy')
labels_path = os.path.join(outdir,'labels.npy')

# for each point with species div data
cnn_input_data = []
for j,div_point in enumerate(true_div_points_coords):
    print('%i/%i'%(j+1,len(true_div_points_coords)),flush=True,end='\r')
    start = time.time()
    lon, lat = div_point # cea coords are in meters
    square_coords = get_square_corner_points(lon,lat,l).astype(int)
    # make a pre-selection for faster processing
    selected_occs_ids = find_occs_in_square(gbif_occs_coords,square_coords)
    selected_landgrid_cell_ids = find_occs_in_square(land_cell_coords, square_coords)
    if grid_plotting == True:
        # plt.plot(gbif_occs_df_complete_rows.lon.values,
        #          gbif_occs_df_complete_rows.lat.values,
        #          '.',
        #          color='black')
        plt.plot(square_coords[:, 0], square_coords[:, 1], '.',color='black')
        plt.plot(np.append(square_coords[:, 0], square_coords[:, 0][0]),
                 np.append(square_coords[:, 1], square_coords[:, 1][0]),
                 '-',
                 color='black')
        plt.scatter(lon, lat,marker='x',color='r',s=100,linewidths=5)
        plt.plot(gbif_occs_coords[selected_occs_ids][:, 0], gbif_occs_coords[selected_occs_ids][:, 1], '.', color='black')

    gridcells = split_square_in_grid_cells(square_coords,g)
    n_occ_layer = np.zeros(g*g).astype(int)
    sr_layer = np.zeros(g*g).astype(int)
    n_landcell_layer = np.zeros(g*g).astype(int)
    for i,gridcell in enumerate(gridcells):
        selected_occs_ids_grid_cell = find_occs_in_square(gbif_occs_coords[selected_occs_ids],gridcell)
        selected_occs_ids_grid_cell_abs = selected_occs_ids[selected_occs_ids_grid_cell]
        land_cells_ids_in_grid_cell = find_occs_in_square(land_cell_coords[selected_landgrid_cell_ids], gridcell)
        land_cells_ids_in_grid_cell_abs = selected_landgrid_cell_ids[land_cells_ids_in_grid_cell]
        n_occs_count = len(selected_occs_ids_grid_cell_abs)
        n_species_count = len(np.unique(gbif_occs_species[selected_occs_ids_grid_cell_abs]))
        n_landcell_count = len(land_cells_ids_in_grid_cell_abs)
        #print(n_occs_count,n_species_count)
        n_occ_layer[i] = n_occs_count
        sr_layer[i] = n_species_count
        n_landcell_layer[i] = n_landcell_count
        if grid_plotting == True:
            plt.fill(gridcell[:, 0],
                     gridcell[:, 1],
                     '.',
                     color='C%i'%(i%10),
                     alpha=0.2)
            # plt.plot(np.append(gridcell[:, 0], gridcell[:, 0][0]),
            #          np.append(gridcell[:, 1], gridcell[:, 1][0]),
            #          '-',
            #          color='C%i'%(i%10))
            plt.plot(gbif_occs_coords[selected_occs_ids_grid_cell_abs][:, 0],
                     gbif_occs_coords[selected_occs_ids_grid_cell_abs][:, 1],
                     '.',
                     color='C%i'%(i%10))
    n_occ_layer = n_occ_layer.reshape([g, g])
    sr_layer = sr_layer.reshape([g,g])
    n_landcell_layer = n_landcell_layer.reshape([g, g])
    cnn_array = np.array([n_occ_layer, sr_layer, n_landcell_layer]).T
    cnn_input_data.append(cnn_array)
    end = time.time()
    #print('Elapsed time: %.4f'%(end-start))

cnn_input_data = np.array(cnn_input_data)
additional_features = true_div_points_plotsize
labels = true_div_points_sr

np.save(cnn_array_path,cnn_input_data)
np.save(features_path,additional_features)
np.save(labels_path,labels)



