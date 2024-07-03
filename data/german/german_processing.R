pacman::p_load(
  tidyverse,
  here,
  glue,
  fs
)

path <- glue("{here()}/data/german")

df_raw <- read_csv(glue("{path}/german_processed.csv"))


qplot(df_raw$MissedPayments)

df <- df_raw %>% 
  mutate(
    ForeignWorker = if_else(ForeignWorker == 0, "No", "Yes"),
    Single = if_else(Single == 0, "No", "Yes")
  ) %>% 
  mutate(
    Age_leq_20 = Age <=20,
    Age_geq_21 = Age >=21,
    Age_geq_30 = Age >=30,
    Age_geq_45 = Age >=45,
    Age_geq_60 = Age >=60,
  ) %>% 
  select(-Age) %>% 
  mutate(
    PurposeOfLoan = if_else(PurposeOfLoan %in% c("Business", "Electronics", "Furniture", "NewCar", "UsedCar"), PurposeOfLoan, "Other"),
    PurposeOfLoan_is_Business = PurposeOfLoan == "Business",
    PurposeOfLoan_is_Electronics = PurposeOfLoan == "Electronics",
    PurposeOfLoan_is_Furniture = PurposeOfLoan == "Furniture",
    PurposeOfLoan_is_NewCar = PurposeOfLoan == "NewCar",
    PurposeOfLoan_is_Used = PurposeOfLoan == "UsedCar",
    PurposeOfLoan_is_Other = PurposeOfLoan == "Other",
  ) %>% 
  select(-PurposeOfLoan) %>% 
  mutate(
    LoanDuration_leq_10 = LoanDuration <= 10,
    LoanDuration_geq_11 = LoanDuration >= 11,
    LoanDuration_geq_21 = LoanDuration >= 21,
    LoanDuration_geq_31 = LoanDuration >= 31,
    LoanDuration_geq_41 = LoanDuration >= 41,
  ) %>% 
  select(-LoanDuration) %>% 
  mutate(
    LoanAmount_leq_1000 = LoanAmount <= 1000,
    LoanAmount_geq_1001 = LoanAmount >= 1001,
    LoanAmount_geq_2001 = LoanAmount >= 2001,
    LoanAmount_geq_3001 = LoanAmount >= 3001,
    LoanAmount_geq_5001 = LoanAmount >= 5001,
    LoanAmount_geq_10001 = LoanAmount >= 10001,
  ) %>% 
  select(-LoanAmount) %>% 
  mutate(
    LoanRateAsPercentOfIncome_is_1 = LoanRateAsPercentOfIncome == 1,
    LoanRateAsPercentOfIncome_is_2 = LoanRateAsPercentOfIncome == 2,
    LoanRateAsPercentOfIncome_is_3 = LoanRateAsPercentOfIncome == 3,
    LoanRateAsPercentOfIncome_is_4 = LoanRateAsPercentOfIncome == 4,
  ) %>% 
  select(-LoanRateAsPercentOfIncome) %>% 
  mutate(
    YearsAtCurrentHome_is_1 = YearsAtCurrentHome == 1,
    YearsAtCurrentHome_is_2 = YearsAtCurrentHome == 2,
    YearsAtCurrentHome_is_3 = YearsAtCurrentHome == 3,
    YearsAtCurrentHome_is_4 = YearsAtCurrentHome == 4,
  ) %>% 
  select(-YearsAtCurrentHome) %>% 
  mutate(
    NumberOfOtherLoansAtBank_is_1 = NumberOfOtherLoansAtBank == 1,
    NumberOfOtherLoansAtBank_geq_2 = NumberOfOtherLoansAtBank >= 2,
  ) %>% 
  select(-NumberOfOtherLoansAtBank) %>% 
  mutate(
    NumberOfLiableIndividuals_is_1 = NumberOfLiableIndividuals == 1,
    NumberOfLiableIndividuals_is_2 = NumberOfLiableIndividuals == 2,
  ) %>% 
  select(-NumberOfLiableIndividuals) %>% 
  select(-OtherLoansAtStore) %>% 
  glimpse()


df_check_if_all_binary_or_cat <- df %>% 
  summarize_all(
    ~n_distinct(.)
  ) %>% 
  pivot_longer(cols = everything()) %>% 
  filter(value != 2) %>% 
  glimpse()

df_helper <- tibble(
    header = names(df)
  ) %>% 
  glimpse()


df_helper <- df %>% 
  summarize_all(
    ~class(.)
  ) %>% 
  pivot_longer(cols = everything(), names_to = "header", values_to = "r_class") %>% 
  mutate(
    is_outcome = if_else(header == "GoodCustomer", TRUE, FALSE),
    is_group_attribute = if_else(header %in% c("Gender", "ForeignWorker", "Single"), TRUE, FALSE),
    type = case_when(
      r_class == "numeric" ~ "numeric",
      r_class == "character" ~ "categorical",
      r_class == "logical" ~ "boolean",
      TRUE ~ NA
    ),
    ordering = NA
  ) %>% 
  select(-r_class) %>% 
  print()

write_csv(df, glue("{path}/german_data.csv"))
write_csv(df_helper, glue("{path}/german_helper.csv"))
