library(raster)
library(rgdal)
library(stringi)
setwd('/Users/tobiasandermann/GitHub/plant_div_NN/')
source("src/helper_functions.R")

rfile = '/Users/tobiasandermann/GitHub/plant_div_NN/data/australia_sr_grid/GRID_NVIS6_0_AUST_EXT_MVG/aus6_0e_mvg/dblbnd.adf'
ra = raster(rfile)
plot(ra)

cropping_window = c(113,  154.2, -39.2, -10.5)
e=extent(cropping_window)

soil_data = 'data/soil/Atlas_of_Australian_Soils_(digital)/soilAtlas2M/soilAtlas2M.shp'
soil_shp = readOGR(soil_data)
pdf('plots/aus_soil.pdf')
plot(soil_shp,col=soil_shp$MAP_CODE,lwd=1, border=NaN)
dev.off()

sort(table(soil_shp$MAP_UNIT),decreasing=TRUE)

ecoregions_file = 'data/ecoregions/wwf_ecoregions/wwf_terr_ecos.shp'
#ecoregions = st_read(ecoregions_file)
#ecoregions_spatial = as(ecoregions, 'Spatial')
ecoregions_spatial = readOGR(ecoregions_file)
ecoregions_shp_cropped <- crop(ecoregions_spatial, e)
table(ecoregions_shp_cropped$G200_BIOME)
table(ecoregions_shp_cropped$BIOME)
table(ecoregions_shp_cropped$ECO_NAME)

# remove islands and stuff
remove_these = grepl('Archipelago', ecoregions_shp_cropped$ECO_NAME,fixed=TRUE)
remove_these[grepl('Guinea', ecoregions_shp_cropped$ECO_NAME,fixed=TRUE)] = TRUE
remove_these[grepl('Timor', ecoregions_shp_cropped$ECO_NAME,fixed=TRUE)] = TRUE
remove_these[grepl('Papuan', ecoregions_shp_cropped$ECO_NAME,fixed=TRUE)] = TRUE
ecoregions_shp_cropped_selected = ecoregions_shp_cropped[!remove_these,]


biome_ids = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14)
biome_names = c('Tropical & Subtropical Moist Broadleaf Forests',
  'Tropical & Subtropical Dry Broadleaf Forests',
  'Tropical & Subtropical Coniferous Forests',
  'Temperate Broadleaf & Mixed Forests',
  'Temperate Conifer Forests',
  'Boreal Forests/Taiga',
  'Tropical & Subtropical Grasslands, Savannas & Shrublands',
  'Temperate Grasslands, Savannas & Shrublands',
  'Flooded Grasslands & Savannas ',
  'Montane Grasslands & Shrublands',
  'Tundra',
  'Mediterranean Forests, Woodlands & Scrub',
  'Deserts & Xeric Shrublands',
  'Mangroves')
color_list = c('#31a354',
               '#BDB336',
               '#9EC748',
               '#addd8e',
               '#006878',
               '#83B797',
               '#EFA32B',
               '#FAD71D',
               '#6BC0B4',
               '#C4A172',
               '#BCD88F',
               '#D9232A',
               '#E0785B',
               '#D81774')

# aggregate by biome
biomes_shp = aggregate(ecoregions_shp_cropped_selected, by='BIOME')

plot_unit = biomes_shp$BIOME
color_pool = color_list#sample(rainbow(length(table(plot_unit))))
color_vector = rep('grey',length(plot_unit))
biome_names_selected = c()
color_list_selected = c()
for (i in 1:length(table(plot_unit))){
  id = as.integer(names(table(plot_unit))[i])
  name = biome_names[id]
  #name = paste0('(',id,') ',biome_names[id])
  biome_names_selected=c(biome_names_selected,name)
  color_vector[plot_unit==id]=color_pool[id]
  color_list_selected = c(color_list_selected,color_pool[id])
}



pdf('plots/aus_ecoregions.pdf',height=9,width=11)
plot(biomes_shp,xlim=c(110,155),ylim=c(-43,-11.5),col=color_vector, lwd=1, border='black',axes = T)
legend('bottom', legend=biome_names_selected,col=color_list_selected, pch=15,pt.cex=2,bg = 'grey',ncol = 2)
dev.off()


rr <- raster(biomes_shp, res=0.1)
rr <- rasterize(biomes_shp, rr, field='BIOME')
plot(rr)


###_______FOR DANIELE_____________
# extract bioclim data for 100 randomely selected points from each biome
# load bioclim rasters
bioclim_rasters = stack(paste('data/climate/wc2/', paste('wc2.1_30s_bio_', 1:19, '.tif', sep=''), sep=''))
bioclim_rasters = crop_raster_to_aus(bioclim_rasters)
#bioclim_rasters = stack(paste('data/climate/wc10/', paste('bio', 1:19, '.bil', sep=''), sep=''))
# coordinates in right resolution
bioclim_raster = raster(bioclim_rasters$wc2.1_30s_bio_1)
values(bioclim_raster) = 1
# select 100 random points for each biome
n_points = 100
biome_id_col = c()
for (biome_id in biomes_shp$BIOME){
  biome_shape = biomes_shp[biomes_shp$BIOME == biome_id,]
  cells_per_biome = mask(bioclim_raster,biome_shape)
  selected_points = sampleRandom(cells_per_biome, size=n_points,cells=TRUE, sp=TRUE)
  biome_id_col = c(biome_id_col,rep(biome_id,n_points))
  if (biome_id==1){
    selected_points_all = selected_points
  }else{
    selected_points_all = rbind(selected_points_all,selected_points) 
  }
}

# ectract bioclim values for these points
pdf('plots/aus_biomes_randomly_selected_points.pdf',height=9,width=11)
plot(biomes_shp,xlim=c(110,155),ylim=c(-43,-11.5),col=color_vector, lwd=1, border='black',axes = T)
plot(selected_points_all,pch=20,cex=.5,add=TRUE)
legend('bottomleft', legend=biome_names_selected,col=color_list_selected, pch=15,pt.cex=2,bg = 'grey',ncol = 2)
dev.off()

bioclim_cols = extract_value_from_raster(selected_points_all,bioclim_rasters)

# export to csv
coords = as.data.frame(selected_points_all)[c('x','y')]
output_data = cbind(coords,biome_id_col,bioclim_cols)
names(output_data) = c('lon','lat','biome',paste0('bio', 1:19))
write.table(output_data,file = 'data/other/bioclim_data_100_random_points_per_biome.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)




