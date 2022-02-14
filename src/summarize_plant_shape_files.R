library(rgdal)
library(raster)
library(sf)
setwd('/Users/tobiasandermann/GitHub/plant_div_NN/')



aus_sr_raster_file = 'data/australia_sr_grid/GRID_NVIS6_0_AUST_EXT_MVG/aus6_0e_mvg/dblbnd.adf'
aus_sr_raster = raster(aus_sr_raster_file)
plot(aus_sr_raster)



# load shape file
australia_land_plants_file = 'data/iucn_range_data/redlist_species_data_9c3e648c-4c71-4afb-808f-9db60cb26dbd/data_0.shp'
#australia_land_plant_ranges = readOGR(australia_land_plants_file)
australia_land_plant_ranges_st = st_read(australia_land_plants_file)

australia_land_plant_ranges_st[2,]$geometry
plot(australia_land_plant_ranges_st[5,]$geometry)

length(australia_land_plant_ranges_st)
names(australia_land_plant_ranges_st)


australia_land_plant_ranges_st$BINOMIAL
