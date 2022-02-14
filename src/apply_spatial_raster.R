library(raster)
library(maptools)
library(rgdal)
setwd('/Users/tobiasandermann/GitHub/plant_div_NN/')
source("src/helper_functions.R")
library(rasterpdf)
library(RColorBrewer)

# get shape of australia________________________________________________________
australia <- readOGR("data/country_shape_files/australia_cea/australia.shp")
plot(australia)

# define raster and get cell coords
australia_raster = raster(australia,res=1000) #1000 makes the grids 1km x 1km
values(australia_raster) <- 0
australia_raster_cropped = australia_raster#mask(australia_raster,australia)
plot(australia_raster_cropped)
raster_df = as.data.frame(australia_raster_cropped,xy=TRUE,na.rm=TRUE)
lon = raster_df$x
lat = raster_df$y
points = cbind(lon,lat)
spatial_points = SpatialPoints(points)



# export the raster with 0 if sea and 1 if land________________________________
# define raster over australia of fixed cell size
australia_raster = raster(australia,res=1000) #1000 makes the grids 1km x 1km
values(australia_raster) <- 1
australia_raster_cropped = mask(australia_raster,australia)
# turn nans to 0
australia_raster_cropped[is.na(australia_raster_cropped[])] <- 0
plot(australia_raster_cropped)
raster_df = as.data.frame(australia_raster_cropped,xy=TRUE,na.rm=TRUE)
write.table(round(raster_df),file = 'data/land_vs_water_data/land_water_coords.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)


# export a raster with climatic values_________________________________________
# get climate data from worldclim
download_folder = 'data/climate/wc2/'
#bilfiles <- paste('bio', 1:19, '.bil', sep='')
#hdrfiles <- paste('bio', 1:19, '.hdr', sep='')	
tiffiles <- paste('wc2.1_30s_bio_', 1:19, '.tif', sep='')
climate_input <- stack(paste(download_folder, tiffiles, sep=''))
projection(climate_input) <- "+proj=longlat +datum=WGS84"
australia_lonlat = spTransform(australia, projection(climate_input))
#climate_input_cea = projectRaster(climate_input,crs=projection(australia)) # takes ~1 min
for (i in seq(1:dim(climate_input)[3])){
  print(i)
  clim_rast = climate_input[[i]]
  clim_rast_cropped = crop(clim_rast,australia_lonlat)
  clim_rast_cropped_cea = projectRaster(clim_rast_cropped,crs=projection(australia)) # takes ~1 min
  pdf(paste0('plots/worldclim_layers/bio',i,'_australia.pdf'))
  plot(clim_rast_cropped_cea,main=paste0('bio',i))
  plot(australia,add=T)
  dev.off()
  # get climate values for coords
  bio_ex <- raster::extract(x = clim_rast_cropped_cea,
                            y = spatial_points)
  points = cbind(points,bio_ex)
}
climate_data_points = as.data.frame(points)
climate_data_points_complete = climate_data_points[complete.cases(climate_data_points), ]
bio_colnames = paste('bio', seq(1:dim(climate_input)[3]), sep='')
names(climate_data_points_complete) = c('lon','lat',bio_colnames)
climate_data_points_complete[,'lon']=round(climate_data_points_complete[,'lon'])
climate_data_points_complete[,'lat']=round(climate_data_points_complete[,'lat'])
climate_data_points_complete[,bio_colnames]=round(climate_data_points_complete[,bio_colnames],3)
write.table(climate_data_points_complete,file = 'data/climate/land_cells_coords_climate.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)



# export a raster with human footprint values__________________________________
hfp_file = 'data/human_footprint/HFP2009.tif'
hfp_raster = raster(hfp_file)
# reduce raster only to australia to reduce size and speed up projection conversion
australia_moll = spTransform(australia, CRS(projection(hfp_raster)))
hfp_raster_cropped = crop(hfp_raster,australia_moll)
hfp_cea = projectRaster(hfp_raster_cropped,crs=projection(australia)) # takes 5 min
hfp_cea_cropped = crop(hfp_cea,australia)
plot(hfp_cea_cropped)
pdf('plots/human_footprint_australia.pdf')
plot(hfp_cea_cropped,main='hfp')
plot(australia,add=T)
dev.off()
# get hfp values for coords
hfp_index <- raster::extract(x = hfp_cea_cropped,
                             y = spatial_points)
points = cbind(lon,lat)
points = cbind(points,hfp_index)
hfp_data_points = as.data.frame(points)
hfp_data_points_complete = hfp_data_points[complete.cases(hfp_data_points), ]
hfp_data_points_complete[,'lon']=round(hfp_data_points_complete[,'lon'])
hfp_data_points_complete[,'lat']=round(hfp_data_points_complete[,'lat'])
hfp_data_points_complete[,'hfp_index']=round(hfp_data_points_complete[,'hfp_index'],6)
write.table(hfp_data_points_complete,file = 'data/human_footprint/land_cells_coords_hfp.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)



# export a raster for elevation data__________________________________
elev_file = 'data/elevation/wc2.1_30s_elev.tif'
elev_raster = raster(elev_file)
# reduce raster only to australia to reduce size and speed up projection conversion
australia_transformed = spTransform(australia, CRS(projection(elev_raster)))
elev_raster_cropped = crop(elev_raster,australia_transformed)
elev_cea = projectRaster(elev_raster_cropped,crs=projection(australia)) # takes 5 min
elev_cea_cropped = crop(elev_cea,australia)
plot(elev_cea_cropped)
pdf('plots/elevation_australia.pdf')
plot(elev_cea_cropped,main='elevation')
plot(australia,add=T)
dev.off()
# get hfp values for coords
elevation <- raster::extract(x = elev_cea_cropped,
                              y = spatial_points)
points = cbind(lon,lat)
points = cbind(points,elevation)
elev_data_points = as.data.frame(points)
elev_data_points_complete = elev_data_points[complete.cases(elev_data_points), ]
elev_data_points_complete[,'lon']=round(elev_data_points_complete[,'lon'])
elev_data_points_complete[,'lat']=round(elev_data_points_complete[,'lat'])
elev_data_points_complete[,'elevation']=round(elev_data_points_complete[,'elevation'],6)
write.table(elev_data_points_complete,file = 'data/elevation/land_cells_coords_elev.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)




#plot(australia_raster_cropped)
#plot(australia_shape_CEA,add=T)
# 


# read the gbif occurrence data
australia_raster = raster(australia,res=10000) #1000 makes the grids 1km x 1km
#values(australia_raster) <- 0
australia_raster_cropped = mask(australia_raster,australia)


gbif_occs_file = 'data/gbif/australia_formatted.txt'
gbif_occs = read.table(gbif_occs_file,sep='\t',header = TRUE)
gbif_occ_coords <- cbind(gbif_occs$decimalLongitude, gbif_occs$decimalLatitude)
gbif_occ_coords_spatial = SpatialPoints(gbif_occ_coords,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
gbif_occ_coords_spatial_cea = spTransform(gbif_occ_coords_spatial, CRS(projection(australia_raster_cropped)))


# count gbif occs per raster cell
counts_per_cell = rasterize(gbif_occ_coords_spatial_cea,australia_raster_cropped,fun='count')
counts_per_cell = log10(counts_per_cell)
counts_per_cell_cropped = mask(counts_per_cell,australia)
plot(counts_per_cell_cropped)

log10(1)

pal <- colorRampPalette(c("white","black"))

#plot(counts_per_cell, breaks=cuts, col = pal(length(cuts)-1)) #plot with defined breaks
raster_pdf('plots/gbif_occs_10km_10km_grid.pdf',res = 1000)
plot(counts_per_cell_cropped,axes=FALSE, box=FALSE, col = pal(100))
plot(australia,add=T)
dev.off()


