#Dataset Processing Script for Adult
#
#-------------------------------------------------------------------------------
#This file loads the raw adult dataset, does some cleanup, and CSV files
#suitable for rule generation using the Python code. These files include:
#
#data_file: contains training data
#
#helper_file: contains information on the variables in data_file
#             note this is needed if the data contains ordinal variables.
#
#weights_file: contains sampling weights for training data, if the raw data
#              contains sampling weights.

#### start-up and script variables ####
data_name = "adult"
raw_data_file = "adult.data";
outcome_name = "IncomeOver50K"
sampling_weight_name = "SamplingWeight"
ordinal_names = c("EducationLevel");

#formats
raw_missing_label = "?";

#set directories
#To-Do: update this
data_dir = "/Users/berk/Dropbox (Harvard University)/repos/lda/data/"
raw_data_dir = paste0(data_dir, data_name, "/")

#output file names
file_header = paste0(raw_data_dir, data_name)
data_file = paste0(file_header, "_data.csv")
helper_file = paste0(file_header, "_helper.csv")
weights_file = paste0(file_header, "_weights.csv");

#load libraries
source(paste0(data_dir,"processing_functions.R"));
required_packages = c('dplyr', 'tidyr', 'forcats', 'here', 'stringr')
for (pkg in required_packages){
    suppressPackageStartupMessages(library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE));
}

#### dataset notes ####

# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

#### load raw data ####

data = read.csv(file = paste0(raw_data_dir, raw_data_file), stringsAsFactors = TRUE, header=FALSE)
colnames(data) = c("Age",
                   "Industry",
                   "SamplingWeight",
                   "EducationLevel",
                   "EducationYrs",
                   "MaritalStatus",
                   "Job",
                   "RelationToHeadOfHousehold",
                   "Race",
                   "Sex",
                   "CapitalGain",
                   "CapitalLoss",
                   "WorkHrsPerWeek",
                   "NativeCountry",
                   "IncomeOver50K");

group_names = c("Race", "Sex", "Immigrant", "MaritalStatus")

#### fix outcome variable ####
outcome_df = data[[outcome_name]];
outcome_df = as.numeric(outcome_df);
neg_ind = outcome_df==1;
pos_ind = outcome_df==2;
outcome_df[neg_ind]=-1;
outcome_df[pos_ind]=1;
data[[outcome_name]] = outcome_df;

#process sampling weights
data = remove.sampling.weights.and.save.to.disk(df = data, sampling_weight_name = sampling_weight_name, sampling_weight_file = weights_file)

# clean up levels for categorical variables
rename_fun = function(s){
    s = trimws(s); #remove whitespace
    s = gsub("\\(.*?\\)","", s); #remove contents of parentheses
    s = gsub("-", " ", s); #replace "-" with " "
    s = str_to_title(s);
    s = gsub(" ","", s); #replace "-" with " "
    return(s)
}

for (var_name in names(data)) {
    v = data[[var_name]];
    if (class(v) == "factor"){
        v = fct_relabel(v, .fun = rename_fun);
        if (any(v=="(Missing)")){
            v = fct_recode(v, "(Missing)" = "?");
            v = fct_explicit_na(v, na_level = "(Missing)");
        }
    }
    data[[var_name]] = v;
}


# Dataset Specific Stuff
data$EducationLevel = fct_relabel(data$EducationLevel, .fun = function(s){
                                      s = gsub("Th", "th", s);
                                      s = gsub("St", "st", s);
                                      s = gsub("Nd", "nd", s);
                                      return(s)}
);

data$EducationLevel = fct_relevel(data$EducationLevel,
                                  c("Preschool","1st4th","5th6th",
                                    "7th8th","9th","10th","11th","12th",
                                    "HsGrad","SomeCollege",
                                    "AssocAcdm","AssocVoc","ProfSchool",
                                    "Bachelors", "Masters","Doctorate"));

# Binarization

#Capital
capital = data$CapitalGain - data$CapitalLoss
data$NoCapital = (capital == 0)*1
data$AnyCapitalGain = (capital > 0)*1
data$AnyCapitalLoss = (capital < 0)*1

#Age
data = data %>%
    mutate(Age_leq_20 = Age <=20,
           Age_geq_21 = Age >=21,
           Age_geq_30 = Age >=30,
           Age_geq_45 = Age >=45,
           Age_geq_60 = Age >=60) %>%
    select(-Age)



#Work Hours
data = data %>%
    mutate(WorkHrs_lt_40 = WorkHrsPerWeek < 40,
           WorkHrs_geq_40 = WorkHrsPerWeek >= 40,
           WorkHrs_geq_50 = WorkHrsPerWeek >= 50,
           WorkHrs_geq_60 = WorkHrsPerWeek >= 60) %>%
    select(-WorkHrsPerWeek)

# Race
data$Race = fct_collapse(data$Race,
                         White=c("White"),
                         NonWhite=c("Black", "Other","AmerIndianEskimo", "AsianPacIslander"))

# Immigrant
data = data %>%
    mutate(Immigrant = !(NativeCountry=='UnitedStates')) %>%
    select(-NativeCountry)


#Martial Status
data$MaritalStatus = fct_collapse(data$MaritalStatus,
                                  Married=c('MarriedAFSpouse', 'MarriedAfSpouse', 'MarriedCivSpouse', 'MarriedSpouseAbsent'),
                                  Single=c("NeverMarried", 'Separated', 'Divorced', 'Widowed')
)

# To Drop
data = data %>%
    select(-RelationToHeadOfHousehold, # drop because it may explicitly indicates protected variables in some cases
           -EducationYrs, #dropped because this is already in EducationLevel
           -CapitalGain,#binarized
           -CapitalLoss,#binarized
    )


# Industry Collapse then Convert into Rules
data$Industry = fct_collapse(data$Industry,
                 Private=c('Private'),
                 Government=c('FederalGov', 'LocalGov', 'StateGov'),
                 SelfEmployed=c('SelfEmpInc', 'SelfEmpNotInc')
                 # Not Included: ['(Missing)', NeverWorked, WithoutPay
                 )

data = data %>%
    mutate(Industry_is_Private = Industry=="Private",
           Industry_is_Government = Industry=="Government",
           Industry_is_SelfEmployed = Industry=="SelfEmployed"
    ) %>%
    select(-Industry)

# Job Collapse then Convert into Rules
data$Job = fct_collapse(data$Job,
    WhiteCollar=c('ExecManagerial', 'AdmClerical'),
    PinkCollar=c('Sales', 'OtherService', 'PrivHouseServ'),
    BlueCollar=c('HandlersCleaners',  'CraftRepair', 'FarmingFishing', 'TransportMoving'),
    Specialized=c('ProfSpecialty', 'MachineOpInspct', 'TechSupport'),
    ArmedOrProtective = c('ArmedForces', 'ProtectiveServ')
)

data = data %>%
    mutate(Job_is_WhiteCollar = Job=="WhiteCollar",
           Job_is_PinkCollar = Job=="PinkCollar",
           Job_is_BlueCollar = Job=="BlueCollar",
           Job_is_Specialized = Job=="Specialized",
           Job_is_ArmedOrProtective = Job=="ArmedOrProtective"
    ) %>%
    select(-Job)

# Education Level
data = data %>%
    mutate(Education_is_HsGrad = EducationLevel=="HsGrad",
           Education_is_Associates = EducationLevel %in% c('AssocAcdm','AssocVoc'),
           Education_is_ProfDegree = EducationLevel %in% c('ProfSchool'),
           Education_is_MastersOrPhD = EducationLevel %in% c("Masters","Doctorate"),
    ) %>%
    select(-EducationLevel)


#### save outputs ####
data = data %>% select(outcome_name, group_names, everything())

#raw data
write.csv(x = data, file = data_file, row.names = FALSE, quote = FALSE);

#helper file
helper_df = get.header.descriptions(data, outcome_name, sampling_weight_name, ordinal_names, group_names);
write.csv(x = helper_df, file = helper_file, row.names = FALSE, quote = FALSE);
