library(rFIA)
library(sf)
library(raster)


outdir = '/Users/tobiasandermann/GitHub/plant_div_NN/data/FIA_vegetation_data/'
getFIA('NE',dir=outdir,load=FALSE)
NE_FIA_data = readFIA(outdir)

## Make a most recent subset
NE_FIA_data_mr <- clipFIA(NE_FIA_data)

# create a raster spanning across North America
north_america_raster = raster(resolution=1, xmn=-180, xmx=-52, ymn=25, ymx=80)
north_america_raster_spatial <- as(north_america_raster, 'SpatialPolygonsDataFrame')


## Most recent estimates for all stems on forest land grouped by user-defined areal units
ctSF <- diversity(clipFIA(NE_FIA_data_mr, mostRecent = TRUE),
                  polys = north_america_raster_spatial,
                  returnSpatial = TRUE)


spatial_ctSF = as(ctSF, 'Spatial')


library(sp)
plot(north_america_raster_spatial)
spplot(spatial_ctSF, zcol="S_g")


plot(spatial_ctSF['S_g'],axes = T)

plot(spatial_ctSF$S_g)

typeof(ctSF)




plotFIA(ctSF, S_g)






