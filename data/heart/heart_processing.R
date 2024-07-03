#Directories
# source("/Users/berk/Desktop/Dropbox/Research/SLIM/R/DataProcessingFunctions.R")

### start-up and script variables ###
data_name = "heart"
raw_data_file = "processed.cleveland.data"
outcome_name = "Disease"
sampling_weight_name = "SamplingWeight"

#set directories
#To-Do: update this
data_dir = "/Users/berk/Dropbox (Harvard University)/repos/lda/data/"
raw_data_dir = paste0(data_dir, data_name, "/")

#output file names
file_header = paste0(raw_data_dir, data_name)
data_file = paste0(file_header, "_data.csv")
helper_file = paste0(file_header, "_helper.csv")
weights_file = paste0(file_header, "_weights.csv");
library(dplyr)
#Description of Raw Data Variables Available at
#https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names


# Attribute Information:
# 	-- Only 14 used
# -- 1. #3  (age)
# -- 2. #4  (sex)
# -- 3. #9  (cp)
# -- 4. #10 (trestbps)
# -- 5. #12 (chol)
# -- 6. #16 (fbs)
# -- 7. #19 (restecg)
# -- 8. #32 (thalach)
# -- 9. #38 (exang)
# -- 10. #40 (oldpeak)
# -- 11. #41 (slope)
# -- 12. #44 (ca)
# -- 13. #51 (thal)
# -- 14. #58 (num)       (the predicted attribute)
source(paste0(data_dir,"processing_functions.R"));

#Load Data (https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)
data = read.csv(file = paste0(raw_data_dir, raw_data_file), stringsAsFactors = TRUE, header=FALSE)

#Renaming Columns so that we know what the variables actually are
colnames(data) = c("Age",
								"Sex",
								"ChestPain",
								"RestBP",
								"Cholesterol",
								"FastingBloodSugar_ge_120",
								"RestECG",
								"ExercisingMaxHR",
								"ExerciseAngina",
								"ExerciseSTDepression",
								"ExerciseSTPeakSlope",
								"NumVesselsByFlourosopy",
								"Thal",
								"DiseaseLVL")

group_names = c("Sex", "Age")

#Deal with Missing Data in Florusopy
a = data$NumVesselsByFlourosopy
a[a=="?"]<-as.factor("0.0")
a = as.numeric(a)
a = a - 2
data$NumVesselsByFlourosopy = a

#Binarize Chest Pain (categorical variable)
data$ChestPain = as.factor(data$ChestPain)
levels(data$ChestPain) = c("typicalangina","atypicalangina","nonanginal","asymptomatic")
data = binarize.categorical.variables(df=data,varlist="ChestPain",remove_categorical_variable = TRUE)

#Binarize Exercise ST Slope (categorical variable)
data$ExerciseSTPeakSlope = as.factor(data$ExerciseSTPeakSlope)
levels(data$ExerciseSTPeakSlope) = c("Up","Flat","Down")
data = binarize.categorical.variables(df=data,varlist="ExerciseSTPeakSlope",remove_categorical_variable = TRUE)

#Binarize Thal(categorical variable)
data$Thal = as.factor(data$Thal)
levels(data$Thal) = c("Missing","Normal","FixedDefect","ReversableDefect")
data = binarize.categorical.variables(df=data,varlist="Thal",remove_categorical_variable = TRUE)

#Binarize Rest ECG (categorical variable)
data$RestECG = as.factor(data$RestECG)
levels(data$RestECG) = c("normal","st_abnormality","likely_v_hypertrophy")
data = binarize.categorical.variables(df=data,varlist="RestECG",remove_categorical_variable = TRUE)

#Cholesterol Levels
# binarize according to medical threshholds
# http://www.nhlbi.nih.gov/health/public/heart/chol/wyntk.htm
data$Cholesterol_lt_200 			= data$Cholesterol<200
data$Cholesterol_200_to_240 	= (data$Cholesterol>=200 & data$Cholesterol<=240)
data$Cholesterol_gt_240 			= data$Cholesterol>240
data$Cholesterol 							= NULL

#RestBP
data$RestBP_Normal 								= data$RestBP<120
data$RestBP_PreHypertension 			= (data$RestBP>=120 & data$RestBP<139)
data$RestBP_Hypertension_Stage_1  = (data$RestBP>=140 & data$RestBP<159)
data$RestBP_Hypertension_Stage_2  = (data$RestBP>=160)
data$RestBP_HypertensiveCrisis 		= (data$RestBP>=180)
data$RestBP = NULL

#Scale Exercise Max Heart Rate
#http://www.heart.org/HEARTORG/GettingHealthy/PhysicalActivity/Target-Heart-Rates_UCM_434341_Article.jsp
TargetMaxHR =  220 - data$Age
data$MaxExerciseHRTargetRatio = (TargetMaxHR/data$ExercisingMaxHR)-1
data$ExercisingMaxHR = NULL

#Age Groups
data$Age = as.factor(ifelse(data$Age>=60, "Old", "Young"))
data$Sex = as.factor(ifelse(data$Sex == 0, 'Female', 'Male'))


#Change DiseaseLVL (ordinal) to Disease (binary)
data$Disease = data$DiseaseLVL>0
data$DiseaseLVL = NULL

##### Additional Processing

# data = convert.logical.to.binary(data)
# data = remove.variables.without.variance(data)
data$Thal_eq_Missing = NULL

data = data %>%
  select(outcome_name, group_names, everything())
# write.csv(x=data,file=data_file,row.names=FALSE,quote=FALSE)
#
# #helper file
# helper_df = get.header.descriptions(data, outcome_name, sampling_weight_name, "", group_names);
# write.csv(x = helper_df, file = helper_file, row.names = FALSE, quote = FALSE);

##### Write Version for Rule Based Models

bdata = data;

bdata$ExerciseSTDepression_geq_1 = bdata$ExerciseSTDepression>1;
bdata$ExerciseSTDepression_geq_2 = bdata$ExerciseSTDepression>2;
bdata$VesselsColored_gt_2 = bdata$NumVesselsByFlourosopy >0;
bdata$Within10PercentageOfTargetMaxHR = abs(bdata$MaxExerciseHRTargetRatio)<=0.05

# Drop Numeric Variables
bdata = bdata %>%
    select(-NumVesselsByFlourosopy, -ExerciseSTDepression, -MaxExerciseHRTargetRatio)

# Binary Cleanup
bdata = remove.complements.of.variables(bdata)
bdata = convert.logical.to.binary(bdata)
for (vname in group_names){
    bdata[vname] = data[vname]
}

# Reorder
bdata = bdata %>%
    select(outcome_name, group_names, everything())

# data file
write.csv(x=bdata, file=data_file,row.names=FALSE,quote=FALSE)

#helper file
helper_df = get.header.descriptions(bdata, outcome_name, sampling_weight_name, "", group_names);
write.csv(x = helper_df, file = helper_file, row.names = FALSE, quote = FALSE)
