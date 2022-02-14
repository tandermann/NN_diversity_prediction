library(sf)
library(raster)
library(rgdal)
setwd('/Users/tobiasandermann/GitHub/plant_div_NN/')
source("src/helper_functions.R")

load("data/splot_open_dataset/sPlotOpen")

#_______________________load splot data_________________________________________
all_splot_data = header.oa
#select only the first resample (resampling was done to remove geographic and resulting biotic biases)
#resampled_records <- all_splot_data[all_splot_data$Resample_1 == TRUE,]
resampled_records <- all_splot_data

# select only plots where all vascular plants have been recorded (true diversity)
plots_where_all_vascular_plants_recorded = resampled_records[resampled_records$Plant_recorded=='All vascular plants',]

sort(table(plots_where_all_vascular_plants_recorded$Releve_area))

# get diversity info for the selected plots
metadata_selected_plots <- CWM_CWV.oa[CWM_CWV.oa$PlotObservationID %in% plots_where_all_vascular_plants_recorded$PlotObservationID,]
div_info_selected_plots = metadata_selected_plots[,c(1,3)]

# get a list of all species names recorded in each plot
species_lists = c()
for (plot_id in plots_where_all_vascular_plants_recorded$PlotObservationID){
  species_list = DT2.oa[DT2.oa$PlotObservationID==plot_id,]$Species
  species_list_string = toString(species_list)
  species_lists = c(species_lists,species_list_string)
  # check if the recorded div and the length of the species list match
  # recorded_div = as.integer(metadata_selected_plots[metadata_selected_plots$PlotObservationID==plot_id,'Species_richness'])
  # if (!recorded_div==length(species_list)){
  #   print(paste(c(recorded_div,length(species_list)),sep=', '))
  # }
}



coordinates = plots_where_all_vascular_plants_recorded[,c('PlotObservationID','Longitude','Latitude','Releve_area')]
combined_df = cbind(coordinates, div_info_selected_plots,species_lists)
# make sure that the plot ids match
combined_df = combined_df[combined_df[,1]==combined_df[,5],]
coordinates_and_div_pre = combined_df[,c(2,3,4,6,7)]
coordinates_and_div_pre = coordinates_and_div_pre[complete.cases(coordinates_and_div_pre),]
# see in which australian state each point falls
spts = SpatialPoints(coordinates_and_div_pre[,c('Longitude','Latitude')])
projection(spts) = "+proj=longlat +datum=WGS84"
aus_state_info = extract_aus_state_ids(spts)
# load data rasters
elevation_raster = raster('data/elevation/wc2.1_30s_elev.tif')
elevation_raster = crop_raster_to_aus(elevation_raster)
hfp_raster = raster('data/human_footprint/HFP2009.tif')
hfp_raster = crop_raster_to_aus(hfp_raster)
hfp_raster_lonlat = projectRaster(hfp_raster,crs=projection(spts)) # takes ~1 min
bioclim_rasters = stack(paste('data/climate/wc0-5/', paste('wc2.1_30s_bio_', 1:19, '.tif', sep=''), sep=''))
bioclim_rasters = crop_raster_to_aus(bioclim_rasters)
# extract the values
elevation_values = extract_value_from_raster(spts,elevation_raster)
hfp_values = extract_value_from_raster(spts,hfp_raster_lonlat)
bioclim_values = extract_value_from_raster(spts,bioclim_rasters)
# tie everything together into one df
coordinates_and_div = cbind(coordinates_and_div_pre[,1:4], aus_state_info,elevation_values,hfp_values,bioclim_values,coordinates_and_div_pre[,5])
col_names = c('lon','lat','plotsize','species_richness','aus_state_info','elevation','hfp',paste('bio_', 1:19, sep=''),'species_list')
names(coordinates_and_div) = col_names
coordinates_and_div[,1:(length(names(coordinates_and_div))-1)]=round(coordinates_and_div[,1:(length(names(coordinates_and_div))-1)],4)
coordinates_and_div[,c('plotsize','species_richness','aus_state_info')]=round(coordinates_and_div[,c('plotsize','species_richness','aus_state_info')])
write.table(coordinates_and_div,file = './data/true_div_data/lonlat/true_diversity_data.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)

# plot on map
library(rworldmap)
world <- getMap(resolution = "less islands")
coords <- cbind(coordinates_and_div$lon, coordinates_and_div$lat)
coords_spatial = SpatialPoints(coords,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
pdf('data/true_div_data/plots/splot_locs_all.pdf')
plot(world)
plot(coords_spatial,add=TRUE,pch = 20, col = "red",cex = 0.1)
dev.off()
#_______________________________________________________________________________

# transform to CEA______________________________________________________________
fileName = 'data/geo_projection.txt'
target_proj = readChar(fileName,file.info(fileName)$size)
coords <- cbind(coordinates_and_div$lon, coordinates_and_div$lat)
coords_spatial = SpatialPoints(coords,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
coords_spatial_projected = spTransform(coords_spatial, CRS(target_proj))
projected_df = cbind(data.frame(coords_spatial_projected),coordinates_and_div[3:length(col_names)])
names(projected_df) = col_names
projected_df[,c('lon','lat','plotsize','species_richness','aus_state_info')]=round(projected_df[,c('lon','lat','plotsize','species_richness','aus_state_info')])
write.table(projected_df,file = 'data/true_div_data/true_diversity_data_proj.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)
#_______________________________________________________________________________



# select only australian splots_________________________________________________
australia_shape = extract_australia_shape()
#australia_raster = raster(australia_shape,res=0.1)
#values(australia_raster) = 1
#australia_raster_cropped = mask(australia_raster,australia_shape)
#plot(australia_raster_cropped)

# save cropped australia shape to file
writeOGR(australia_shape, "data/country_shape_files/australia_lonlat/", "australia", driver = "ESRI Shapefile")
coords <- cbind(coordinates_and_div$lon, coordinates_and_div$lat)
coords_spatial = SpatialPoints(coords,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
a = over(coords_spatial, australia_shape)
splot_data_australia = coordinates_and_div[!is.na(a),]
splot_data_australia = splot_data_australia[complete.cases(splot_data_australia), ]
write.table(splot_data_australia,file = 'data/true_div_data/lonlat/true_diversity_data_aus.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)

# transform to CEA______________________________________________________________
coords <- cbind(splot_data_australia$lon, splot_data_australia$lat)
coords_spatial = SpatialPoints(coords,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
coords_spatial_projected = spTransform(coords_spatial, CRS(target_proj))
projected_df = cbind(data.frame(coords_spatial_projected),splot_data_australia[3:length(col_names)])
names(projected_df) = col_names
projected_df[,c('lon','lat','plotsize','species_richness','aus_state_info')]=round(projected_df[,c('lon','lat','plotsize','species_richness','aus_state_info')])
write.table(projected_df,file = 'data/true_div_data/true_diversity_data_proj_aus.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)
# plot to map
australia_shape = extract_australia_shape()
australia_shape_proj = spTransform(australia_shape, CRS(target_proj))
# save cropped australia shape to file
writeOGR(australia_shape_proj, "data/country_shape_files/australia_cea/", "australia", driver = "ESRI Shapefile")
pdf('data/true_div_data/plots/splot_locs_all_aus.pdf')
plot(australia_shape_proj)
plot(coords_spatial_projected,add=TRUE,pch = 20, col = "red",cex = 0.2)
dev.off()
#_______________________________________________________________________________





#Create a function to generate a continuous color palette
rbPal <- colorRampPalette(c('yellow','red','black'))
color_array <- rbPal(10)[as.numeric(cut(splot_data_australia$species_richness,breaks = 10))]
pdf('data/true_div_data/plots/splot_locs_all_aus_sr.pdf')
plot(australia_shape_proj)
plot(coords_spatial_projected,add=TRUE,pch = 20, col = color_array, cex = 0.2)
legend("topleft",title="Species diversity",legend=levels(cut(splot_data_australia$species_richness,breaks = 10)),col =rbPal(10),pch=20,box.lwd = 0,box.col = "white",bg = "white")
dev.off()




# select only those splots of certain size______________________________________
sort(table(splot_data_australia$plotsize))
plotsizes = c(100, 250, 300, 400, 500, 1000, 1400, 10000)
for (plot_size in plotsizes){
  splots_selected = splot_data_australia[splot_data_australia$plotsize==plot_size,]
  write.table(splots_selected,file = paste0('./data/true_div_data/lonlat/true_diversity_data_aus_',plot_size,'m2.txt'),sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)
  # project coords
  coords_selected <- cbind(splots_selected$lon, splots_selected$lat)
  coords_selected_spatial = SpatialPoints(coords_selected,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
  coords_selected_spatial_proj = spTransform(coords_selected_spatial, target_proj)
  splots_selected_cea = cbind(data.frame(coords_selected_spatial_proj),splots_selected[3:length(col_names)])
  names(splots_selected_cea) = col_names
  splots_selected_cea[,c('lon','lat','plotsize','species_richness','aus_state_info')]=round(splots_selected_cea[,c('lon','lat','plotsize','species_richness','aus_state_info')])
  write.table(splots_selected_cea,file = paste0('./data/true_div_data/true_diversity_data_proj_aus_',plot_size,'m2.txt'),sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)
  # plot to map
  pdf(paste0('data/true_div_data/plots/splot_locs_aus_plot_size_',plot_size,'m2.pdf'))
  plot(australia_shape_proj)
  plot(coords_selected_spatial_proj,add=TRUE,pch = 20, col = "red",cex = 0.2)
  title(paste0(plot_size,' m2'))
  dev.off()
}


# export raster for prediction
australia <- readOGR("data/country_shape_files/australia_cea/australia.shp")
plot(australia)
grid_length = 10000
plotsize = grid_length*grid_length
pred_raster = raster(australia,res=grid_length) #1000 makes the grids 1km x 1km
values(pred_raster) <- 0
pred_raster_cropped = mask(pred_raster,australia)
plot(pred_raster_cropped)
plot(australia,add=T)
pred_coords_df = as.data.frame(pred_raster_cropped,xy=TRUE,na.rm=TRUE)
spts = SpatialPoints(pred_coords_df[,c('x','y')],proj4string = CRS('+proj=cea +datum=WGS84 +lat_ts=-30'))
target_proj = '+proj=longlat +datum=WGS84 +no_defs'
spts = spTransform(spts, CRS(target_proj))
plot_size_col = rep(plotsize,dim(pred_coords_df)[1])
species_richness_col = rep(0,dim(pred_coords_df)[1])
aus_state_col = extract_aus_state_ids(spts)
elevation_col = extract_value_from_raster(spts,elevation_raster)
hfp_col = extract_value_from_raster(spts,hfp_raster_lonlat)
bioclim_col = extract_value_from_raster(spts,bioclim_rasters)
output_data = cbind(pred_coords_df[,c('x','y')],plot_size_col,species_richness_col,aus_state_col,elevation_col,hfp_col,bioclim_col)
output_data = output_data[complete.cases(output_data), ]
names(output_data) = col_names[1:26]
write.table(output_data,file = paste0('data/prediction_data/prediction_cells_coords_res_',grid_length,'.txt'),sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)





