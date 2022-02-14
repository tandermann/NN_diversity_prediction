library(sf)
library(raster)
library(rgdal)

extract_australia_shape <- function(){
  world <- st_read("data/country_shape_files/ne_10m_admin_0_countries.shp")
  #world <- readShapeSpatial("ne_10m_admin_0_countries.shp")
  # Plot Australia
  australia_shape = world[world$ADMIN=="Australia",1]
  # crop to only mainland
  australia_spatial = as(australia_shape, 'Spatial')
  e <- extent(113, 154.2, -44.2, -10.5)
  australia_spatial_cropped <- crop(australia_spatial, e)
  return(australia_spatial_cropped)
}

extract_aus_state_ids <- function(spts){
  australia_states = readOGR("data/country_shape_files/AUS_adm/AUS_adm1.shp")
  crs(spts) = crs(australia_states)
  state_out = over(spts,australia_states)
  
  #australia_states$ID_1
  #australia_states$NAME_1
  aus_state_info = as.integer(state_out$ID_1)
  aus_state_info[is.na(aus_state_info)] = 999
  aus_state_info[aus_state_info==1] = 999 # these are all non-continental territories
  aus_state_info[aus_state_info==3] = 999
  aus_state_info[aus_state_info==4] = 999
#  aus_state_info[aus_state_info==9] = 999 # Tasmania
  return(aus_state_info)
}

extract_biome_ids <- function(spts){
  ecoregions_file = 'data/ecoregions/wwf_ecoregions/wwf_terr_ecos.shp'
  ecoregions = readOGR(ecoregions_file)
  spts_projected = spTransform(spts, CRS(projection(ecoregions)))
  ecoregion_out = over(spts_projected,ecoregions)
  #australia_states$ID_1
  #australia_states$NAME_1
  biome_info = as.integer(ecoregion_out$BIOME)
  ecoregion_info = as.integer(ecoregion_out$ECO_NUM)
  return(list(biome_info,ecoregion_info))
}


extract_value_from_raster <- function(spts,raster){
  ex_values <- raster::extract(x = raster,
                               y = spts)
  return(ex_values)
}

crop_raster_to_aus <- function(raster){
  e <- extent(113, 154.2, -44.2, -10.5)
  p <- as(e, 'SpatialPolygons')
  projection(p) = "+proj=longlat +datum=WGS84"
  p_transformed = spTransform(p, projection(raster))
  rast_cropped = crop(raster,p_transformed)
}
