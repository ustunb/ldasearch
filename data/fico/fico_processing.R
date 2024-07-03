pacman::p_load(
  tidyverse,
  recipes,
  fixest,
  skimr,
  here,
  glue,
  fs
)

path <- glue("{here()}/data/fico")

df_groups <- read_csv(glue("{path}/fico_raw.csv")) %>% 
  transmute(
    RiskPerformance = if_else(RiskPerformance == 0, -1, RiskPerformance),
    GroupRisk = if_else(ExternalRiskEstimate > median(ExternalRiskEstimate), "Low", "High"),
    GroupTrades = if_else(NumTotalTrades > median(NumTotalTrades), "Thick", "Thin")
  ) %>% 
  glimpse()

df_raw <- read_csv(glue("{path}/fico_binned.csv"))

v_cat_to_dummy <- c("MonthsWithZeroBalanceOverLast6Months", "MonthsWithLowSpendingOverLast6Months", "MonthsWithHighSpendingOverLast6Months", "TotalOverdueCounts", "TotalMonthsOverdue")
v_num_to_discretize <- c("MaxBillAmountOverLast6Months", "MaxPaymentAmountOverLast6Months", "MostRecentBillAmount", "MostRecentPaymentAmount")


df <- df_raw %>% 
  select(-RiskPerformance) %>% 
  bind_cols(df_groups) %>% 
  relocate(c(RiskPerformance, GroupRisk, GroupTrades), .before = everything()) %>% 
  select(-GroupRisk, -NumTotalTrades) %>%
  recipe(RiskPerformance ~ .) %>% 
  step_dummy(c(-RiskPerformance, -GroupTrades)) %>%
  prep() %>% 
  bake(new_data = NULL) %>% 
  rename_with(~str_replace_all(., "\\.", "_")) %>% 
  relocate(RiskPerformance, .before = everything()) %>% 
  glimpse()



df_helper <- df %>% 
  summarize_all(
    ~class(.)
  ) %>% 
  pivot_longer(cols = everything(), names_to = "header", values_to = "r_class") %>% 
  mutate(
    is_outcome = if_else(header == "RiskPerformance", TRUE, FALSE),
    is_group_attribute = if_else(header %in% c("GroupTrades"), TRUE, FALSE),
    type = case_when(
      r_class == "numeric" ~ "numeric",
      r_class %in% c("character", "factor") ~ "categorical",
      r_class == "logical" ~ "boolean",
      TRUE ~ NA
    ),
    ordering = NA
  ) %>%
  select(-r_class) %>% 
  print()



write_csv(df, glue("{path}/fico_data.csv"))
write_csv(df_helper, glue("{path}/fico_helper.csv"))





