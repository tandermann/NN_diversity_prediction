library(raster)

setwd('/Users/tobiasandermann/GitHub/plant_div_NN/')
download_folder = 'data/climate/'
res = 2.5
climate_input <- raster::getData('worldclim',
                                var = 'bio',
                                res = res,
                                path = download_folder)
download_folder = 'data/elevation/'
elevation_data <- raster::getData('alt',
                                  country='Australia',
                                 res = res,
                                 path = download_folder)

res=10
#zip <- paste('bio_', res, 'm_bil.zip', sep='')
#zipfile <- paste(download_folder, zip, sep='')

bilfiles <- paste('bio', 1:19, '.bil', sep='')
hdrfiles <- paste('bio', 1:19, '.hdr', sep='')	

climate_input <- stack(paste(download_folder, bilfiles, sep=''))
projection(climate_input) <- "+proj=longlat +datum=WGS84"

bio_ex <- raster::extract(x = climate_input,
                          y = x[,c(lon, lat)])

