library(raster)
setwd('/Users/tobiasandermann/GitHub/plant_div_NN/')

occurrence_records_df = read.table('data/gbif/australia_formatted.txt',sep = '\t',fill=TRUE,header = TRUE)
#head(occurrence_records_df)
dim(occurrence_records_df)
occurrence_records_df = transform(occurrence_records_df, decimalLongitude = as.numeric(decimalLongitude))
occurrence_records_df = transform(occurrence_records_df, decimalLatitude = as.numeric(decimalLatitude))
#head(occurrence_records_df)

# only keep rows that have complete longitude and latitude
occurrence_records_df = occurrence_records_df[complete.cases(occurrence_records_df[ , c('decimalLongitude','decimalLatitude')]),]

# check species names against list of valid species names from WCVP
wcvp_species_list = c(read.csv('data/wcvp/wcvp_species_list.txt', header=FALSE))$V1
occurrence_records_df <- occurrence_records_df[occurrence_records_df$species %in% wcvp_species_list,]
names(occurrence_records_df) = c('species','lon','lat')

# write to table
write.table(occurrence_records_df,file = 'data/gbif/australia_formatted_cleaned.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)

# convert into right projection
fileName = 'data/geo_projection.txt'
target_proj = readChar(fileName,file.info(fileName)$size)
gbif_occ_coords <- cbind(occurrence_records_df$lon, occurrence_records_df$lat)
gbif_occ_coords_spatial = SpatialPoints(gbif_occ_coords,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
gbif_occ_coords_spatial_projected = spTransform(gbif_occ_coords_spatial, CRS(target_proj))
projected_df = cbind(occurrence_records_df$species,data.frame(gbif_occ_coords_spatial_projected))
names(projected_df) = c('species','lon','lat')
write.table(projected_df,file = 'data/gbif/australia_formatted_cleaned_proj.txt',sep = '\t',row.names = FALSE,col.names = TRUE,quote = FALSE)



# _________________________species geo coder cleaning___________________________
# library(CoordinateCleaner)
# # Run record-level tests
# rl <- clean_coordinates(x = occurrence_records_df,
#                         lon = "decimalLongitude",
#                         lat = "decimalLatitude",
#                         tests = c("capitals", 
#                                   "centroids", 
#                                   "equal", 
#                                   "gbif", 
#                                   "institutions", 
#                                   "seas", 
#                                   "zeros"))
# #summary(rl)
# #plot(rl,lon='decimalLongitude',lat='decimalLatitude')
# 
# # remove all flagged rows
# rl = rl[rl[,'.summary'],]
# 
# # check how many records were removed
# dim(occurrence_records_df)
# dim(rl)
# 
# names(rl)
# 
# rl_reduced = rl[,c('species','decimalLongitude','decimalLatitude')]
# head(rl_reduced)
# 
# rl_reduced = rl_reduced[order(rl_reduced$species),]
# head(rl_reduced)










