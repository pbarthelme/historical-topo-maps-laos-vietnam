library(mapaccuracy)
library(dplyr)

pixel_area = (4^2) / 1e6 # based on 4m resolution in km²
  
##### Laos #####
# Load data
data <- read.csv("../data/processed/acc_eval/test_labels_pred_lao.csv")
strata_counts <- read.csv("../data/processed/acc_eval/strata_counts_lao.csv") 
class_counts <- read.csv("../data/processed/acc_eval/class_counts_lao.csv")

# Perform accuracy assessment
stratum <- as.factor(data$stratum)
reference <- as.factor(data$label)   
pred <- as.factor(data$pred)        
Nh <- setNames(strata_counts$pixels, as.factor(strata_counts$class)) 
total_area <- sum(Nh) * pixel_area  

accuracy_results <- stehman2014(s = stratum, r = reference, m = pred, Nh = Nh)

# Calculate mapped area in km²
mapped_area <- setNames(class_counts$pixels, as.factor(class_counts$class)) * pixel_area 

# Table
# Extract metrics
oa <- accuracy_results$OA
ua <- accuracy_results$UA
pa <- accuracy_results$PA
se_oa <- accuracy_results$SEoa
se_ua <- accuracy_results$SEua
se_pa <- accuracy_results$SEpa
adjusted_area <- accuracy_results$area * total_area 
se_adjusted_area <- accuracy_results$SEa * total_area  

# Create the results table
results_table <- data.frame(
  `Class` = names(accuracy_results$area),
  `Estimated Area (km²)` = paste0(round(adjusted_area, 1), " ± ", round(1.96 * se_adjusted_area, 1)),
  `Estimated Area Proportion` = paste0(round(adjusted_area / sum(adjusted_area) * 100, 3), "% ± ", round(1.96 * se_adjusted_area / sum(adjusted_area) * 100, 3), "%"),
  `Margin of Error` = paste0(round(1.96 * se_adjusted_area / adjusted_area * 100, 1), "%"),
  `User Accuracy` = paste0(round(ua * 100, 1), "% ± ", round(1.96 * se_ua * 100, 1), "%"),
  `Producer Accuracy` = paste0(round(pa * 100, 1), "% ± ", round(1.96 * se_pa * 100, 1), "%"),
  check.names = FALSE
)

results_table <- results_table %>%
  arrange(as.numeric(Class))
results_table$`Class Name` <- class_counts$class_name
results_table$`Mapped Area (km²)` <- round(mapped_area, 1)
results_table$`Mapped Area Proportion` <- paste0("", round(mapped_area / sum(mapped_area) * 100, 3), "%")

print(oa)
print(1.96*se_oa)
print(results_table)

write.csv(
  results_table[, c(
    "Class Name", "Mapped Area (km²)", "Estimated Area (km²)",
    "Mapped Area Proportion", "Estimated Area Proportion",
    "Margin of Error", "User Accuracy", "Producer Accuracy"
    )],
  "../outputs/accuracy_and_area_estimates_lao.csv",
  fileEncoding="UTF-16",
  row.names = FALSE
  )

##### Vietnam ######
##### Laos #####
# Load data
data <- read.csv("../data/processed/acc_eval/test_labels_pred_vnm.csv")
strata_counts <- read.csv("../data/processed/acc_eval/strata_counts_vnm.csv") 
class_counts <- read.csv("../data/processed/acc_eval/class_counts_vnm.csv")

# Perform accuracy assessment
stratum <- as.factor(data$stratum)
reference <- as.factor(data$label)   
pred <- as.factor(data$pred)        
Nh <- setNames(strata_counts$pixels, as.factor(strata_counts$class)) 
total_area <- sum(Nh) * pixel_area  

accuracy_results <- stehman2014(s = stratum, r = reference, m = pred, Nh = Nh)

# Calculate mapped area in km²
mapped_area <- setNames(class_counts$pixels, as.factor(class_counts$class)) * pixel_area 

# Table
# Extract metrics
oa <- accuracy_results$OA
ua <- accuracy_results$UA
pa <- accuracy_results$PA
se_oa <- accuracy_results$SEoa
se_ua <- accuracy_results$SEua
se_pa <- accuracy_results$SEpa
adjusted_area <- accuracy_results$area * total_area 
se_adjusted_area <- accuracy_results$SEa * total_area  

# Create the results table
results_table <- data.frame(
  `Class` = names(accuracy_results$area),
  `Estimated Area (km²)` = paste0(round(adjusted_area, 1), " ± ", round(1.96 * se_adjusted_area, 1)),
  `Estimated Area Proportion` = paste0(round(adjusted_area / sum(adjusted_area) * 100, 3), "% ± ", round(1.96 * se_adjusted_area / sum(adjusted_area) * 100, 3), "%"),
  `Margin of Error` = paste0(round(1.96 * se_adjusted_area / adjusted_area * 100, 1), "%"),
  `User Accuracy` = paste0(round(ua * 100, 1), "% ± ", round(1.96 * se_ua * 100, 1), "%"),
  `Producer Accuracy` = paste0(round(pa * 100, 1), "% ± ", round(1.96 * se_pa * 100, 1), "%"),
  check.names = FALSE
)

results_table <- results_table %>%
  arrange(as.numeric(Class))
results_table$`Class Name` <- class_counts$class_name
results_table$`Mapped Area (km²)` <- round(mapped_area, 1)
results_table$`Mapped Area Proportion` <- paste0(round(mapped_area / sum(mapped_area) * 100, 3), "%")

print(oa)
print(1.96*se_oa)
print(results_table)

write.csv(
  results_table[, c(
    "Class Name", "Mapped Area (km²)", "Estimated Area (km²)",
    "Mapped Area Proportion", "Estimated Area Proportion",
    "Margin of Error", "User Accuracy", "Producer Accuracy"
  )],
  "../outputs/accuracy_and_area_estimates_vnm.csv",
  fileEncoding="UTF-16",
  row.names = FALSE
)
