library(ggmap)
library(rgdal)
library(sf)
library(raster)
library(rgeos)
setwd('/Users/tobiasandermann/GitHub/plant_div_NN/')

googl_api_key = 'AIzaSyCBD3r1kfrdYL-kFW4r0BGSuPog6K29jzY'
register_google(googl_api_key)


xlim = c(150.475,150.545)
ylim = c(-30.525,-30.475)
center_point = c(xlim[1]+(xlim[2]-xlim[1])/2,ylim[2]-(ylim[2]-ylim[1])/2)
australia_google <- get_map(location=center_point,zoom=13,maptype = "satellite")
ggmap(australia_google)

#n_points = 20
#lons = runif(n_points,xlim[1],xlim[2])
#lats = runif(n_points,ylim[1],ylim[2])
#points = data.frame(cbind(lons,lats))
#save(points,file='doc/figures/plotting_data/point_coords.RData')
load('doc/figures/plotting_data/point_coords.RData')
points2 = data.frame(cbind(center_point[1],center_point[2]))
n_gbif_points = 100
lons2 = runif(n_gbif_points,xlim[1],xlim[2])
lats2 = runif(n_gbif_points,ylim[1],ylim[2])
points3 = data.frame(cbind(lons2,lats2))

png('plots/aus_map_small_site.png',height=500,width=600)
qmap(location=center_point,zoom=13,maptype = "satellite")+
#ggmap(australia_google) +
  xlim(xlim) +
  ylim(ylim) +
  geom_point(data = points2, aes(x = X1, y = X2),color='white',shape=0,size=200,stroke = 1) +
  geom_point(data = points3, aes(x = lons2, y = lats2),color='white',shape=4,size=2,stroke = 1) +
  geom_point(data = points, aes(x = lons, y = lats),color='red',shape=0,size=8,stroke = 2) +
  geom_point(data = points2, aes(x = X1, y = X2),color='red',shape=15,size=8,stroke = 2)
dev.off()  
