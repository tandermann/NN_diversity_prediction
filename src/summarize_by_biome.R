setwd('/Users/tobiasandermann/GitHub/plant_div_NN/')
source("src/helper_functions.R")

predictions_file = 'results/production_models/production_model_gamma_ensemble/gamma_uncertainty_predictions.txt'
predictions_data = read.csv(predictions_file,header = FALSE,sep=' ')
pred_coords = predictions_data[,c(1,2)]
pred_preds = predictions_data[,3]
uncertainty_values = predictions_data[,4]

spts = SpatialPoints(pred_coords)
fileName = 'data/geo_projection.txt'
target_proj = readChar(fileName,file.info(fileName)$size)
projection(spts) = target_proj
output = extract_biome_ids(spts)
biome_info = output[[1]]
ecoregion_info = output[[2]]

output_data = cbind(pred_coords,biome_info,ecoregion_info,uncertainty_values,pred_preds)
output_data = output_data[complete.cases(output_data), ]
write.table(output_data,file = paste0(dirname(predictions_file),'/pred_with_biome_info.txt'),sep = '\t',row.names = FALSE,col.names = FALSE,quote = FALSE)



#plot(spts,col=ecoregion_info)
# 
# biome_ids = c()
# div_estimates = c()
# for (biome_id in names(table(biome_info))){
#   biome_ids = c(biome_ids,biome_id)
#   biome_div_estimates = pred_preds_mean[biome_info==as.integer(biome_id)]
#   biome_div_mean = mean(biome_div_estimates,na.rm=TRUE)
#   biome_div_std = sd(biome_div_estimates,na.rm=TRUE)
#   div_estimates = c(div_estimates,biome_div_mean)
# }

