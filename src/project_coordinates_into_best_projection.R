library(raster)
library(maptools)
setwd('/Users/tobiasandermann/GitHub/plant_div_NN/')

# get shape of australia________________________________________________________
# download and unzip the map data
#download.file("http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip", "data/countries.zip")
#unzip("countries.zip")
# Read in the shapefile
world <- st_read("data/country_shape_files/ne_10m_admin_0_countries.shp")
#world <- readShapeSpatial("ne_10m_admin_0_countries.shp")
# Plot Australia
australia_shape = world[world$ADMIN=="Australia",1]
# crop to only mainland
australia_spatial = as(australia_shape, 'Spatial')
e <- extent(113, 154.2, -39.2, -10.5)
australia_spatial_cropped <- crop(australia_spatial, e)
# define the correct projection
cea_proj = "+proj=cea +datum=WGS84 +lat_ts=-30"
australia_shape_CEA = spTransform(australia_spatial_cropped, CRS(cea_proj))

# read the gbif occurrence data
gbif_occs_file = 'data/gbif/australia_formatted.txt'
gbif_occs = read.table(gbif_occs_file,sep='\t',header = TRUE)
gbif_occ_coords <- cbind(gbif_occs$decimalLongitude, gbif_occs$decimalLatitude)
gbif_occ_coords_spatial = SpatialPoints(gbif_occ_coords,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
gbif_occ_coords_spatial_cea = spTransform(gbif_occ_coords_spatial, CRS(cea_proj))
cea_df = cbind(gbif_occs$species,data.frame(gbif_occ_coords_spatial_cea))
write.table(cea_df,file = 'data/gbif/australia_formatted_cea_proj.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)

true_div_file = 'data/true_diversity_data_400m2.txt'
true_div = read.table(true_div_file,sep='\t',header = TRUE)
true_div_coords <- cbind(true_div$Longitude, true_div$Latitude)
true_div_coords_spatial = SpatialPoints(true_div_coords,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
a = over(true_div_coords_spatial, australia_spatial_cropped)
true_div_coords_spatial_australia = true_div_coords_spatial[!is.na(a)]
species_richness_australia = true_div$Species_richness[!is.na(a)]
true_div_coords_spatial_cea = spTransform(true_div_coords_spatial_australia, CRS(cea_proj))
true_div_cea_df = cbind(data.frame(true_div_coords_spatial_cea),species_richness_australia)
write.table(true_div_cea_df,file = 'data/true_diversity_data_400m2_cea_proj.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)




aus_raster = raster(australia_shape_CEA,resolution=400)
counts_per_cell = rasterize(true_div_coords_spatial_cea,aus_raster,fun='count')
values(a) = runif(ncell(a))
plot(australia_shape_CEA)
plot(counts_per_cell,add=TRUE)
plot(australia_shape_CEA,add=TRUE)
plot(true_div_coords_spatial_cea,add=TRUE)
