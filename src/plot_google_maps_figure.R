library(ggmap)
library(rgdal)
library(sf)
library(raster)
library(rgeos)
setwd('/Users/tobiasandermann/GitHub/plant_div_NN/')

googl_api_key = 'AIzaSyCBD3r1kfrdYL-kFW4r0BGSuPog6K29jzY'
register_google(googl_api_key)

# get shape file of australia
australia <- readOGR("data/country_shape_files/australia_lonlat/australia.shp")
#australia <- readOGR('data/country_shape_files/AUS_adm/AUS_adm0.shp')
# coastline <- readOGR('data/country_shape_files/natural-earth-vector-master/10m_physical/ne_10m_land.shp')
# # define cropping window
# cropping_window = c(100,  160, -42, -8)
# e <- extent(cropping_window)
# australia <- crop(coastline, e)
plot(australia)

box = raster(australia,nrow=1, ncol=1)
box_spatial = as(box, 'SpatialPolygonsDataFrame')
inverse_australia <- gDifference(box_spatial,geometry(australia), drop_lower_td = TRUE)
plot(inverse_australia,col='green')
inverse_australia <- fortify(inverse_australia)

# get google maps image of australia
#australia_google <- get_map(location=center_point,zoom=4,maptype = "satellite")

# get the state shapes of australia
australia_states = readOGR("data/country_shape_files/AUS_adm/AUS_adm1.shp")
aus_states_fort = fortify(australia_states)
states <- fortify(australia_states) # highlighted
#data1 <- fortify(australia_states[c(5,7,8,10),]) # highlighted
data2 <- fortify(australia_states[c(2,6,11),]) # dark

# read splot locations
splot_data_aus = 'data/true_div_data/lonlat/true_diversity_data_aus.txt'
splot_data = read.csv(splot_data_aus,sep='\t')
splot_coords = cbind(splot_data$lon,splot_data$lat)
splot_coords = data.frame(splot_coords)
#splot_spts = SpatialPoints(splot_coords, proj4string = crs(australia_states))

# make plot
cropping_window = c(113, 154.2, -44.2, -10.5)
delta_x = cropping_window[2]-cropping_window[1]
delta_y = abs(cropping_window[3])-abs(cropping_window[4])
#center_point = c(cropping_window[1]+delta_x/2,cropping_window[3]+delta_y/2)
height_width_ratio = delta_y/delta_x
height = 8
width = (height/height_width_ratio)
width = width-0.11*width # correct by 10% margin
pdf('plots/aus_map.pdf',height=height,width = width)
  qmap('Australia',zoom=4,maptype = 'satellite',crs=crs(australia_states)) +
    xlim(c(cropping_window[1],cropping_window[2])) +
    ylim(c(cropping_window[3],cropping_window[4])) +
    geom_polygon(aes(x = long, y = lat, group = group), data = states,
                colour = 'black', fill = NaN, alpha = .4, size = .3)+
    #  geom_polygon(aes(x = long, y = lat, group = group),data = inverse_australia,col='white',fill='white',size=0.001)+
    #  geom_polygon(aes(x = long, y = lat, group = group), data = data1,
    #               colour = 'white', fill = NaN, alpha = .4, size = .3)+
    #  geom_polygon(aes(x = long, y = lat, group = group), data = data2,
    #               colour = 'white', fill = 'black', alpha = .4, size = .3)
    geom_point(data = splot_coords, aes(x = X1, y = X2), color="white", size=0.1, alpha=1)
dev.off()
