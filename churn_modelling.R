## PlatefulNZ Customer Churn Prediction System
#
# This script implements the full analytical workflow for predicting customer churn
# and identifying key drivers of retention using the PlatefulNZ customer dataset.  It
# follows the same modelling steps used in the original group project but has
# been cleaned and refactored into a standalone R script.  You can run this
# code end‑to‑end to reproduce the results reported in the accompanying
# analysis summary.

## ----------------------------------------------------------------------
## 1. Setup
## ----------------------------------------------------------------------

# Load required packages.  If you do not have these packages installed
# please install them first using `install.packages()`.
library(tidyverse)
library(tidymodels)
library(themis)      # for SMOTE/upsampling
library(vip)         # variable importance plots
library(finetune)    # tuning helpers
library(probably)    # calibration & thresholding

# Set a consistent random seed for reproducibility
set.seed(98)

## ----------------------------------------------------------------------
## 2. Data Loading & Cleaning
## ----------------------------------------------------------------------

# Define the relative path to your data.  The expectation is that a file
# called `platefulnz_customers.csv` lives in a `data/` folder alongside this
# script.  If you do not have permission to share the full dataset, keep
# the filename but omit the file itself from your public repository.
data_path <- file.path("data", "platefulnz_customers.csv")
if (!file.exists(data_path)) {
  stop(paste("Data file not found at", data_path,
             "\nPlease add your CSV file to the data/ folder."))
}

# Read the CSV; suppress column type messages for brevity
raw <- readr::read_csv(data_path, show_col_types = FALSE)

# Basic cleaning:
#  - convert the target variable `retained_binary` into a factor
#  - remove the identifier column
#  - ensure numeric values are parsed correctly
#  - handle any obvious missingness (this script leaves more advanced
#    imputation to the recipe step later).
df <- raw %>%
  mutate(
    retained_binary      = factor(as.character(retained_binary), levels = c("0", "1")),
    user_ID             = NULL,
    weeks_since_signup  = readr::parse_number(as.character(weeks_since_signup)),
    weeks_since_last_purchase = readr::parse_number(as.character(weeks_since_last_purchase)),
    avg_AddOnpurchase_value   = readr::parse_number(as.character(avg_AddOnpurchase_value))
  )

# Let yardstick know that class "0" (churn) is the event of interest
options(yardstick.event_first = TRUE)

# Quick check of the class distribution
class_distribution <- df %>% count(retained_binary, name = "n") %>%
  mutate(prop = n / sum(n))
print(class_distribution)

## ----------------------------------------------------------------------
## 3. Train/Test Split
## ----------------------------------------------------------------------

# Split the full dataset into training and testing sets using stratified
# sampling on the outcome to preserve the churn ratio in both subsets.
split_full <- initial_split(df, prop = 0.80, strata = retained_binary)
train_data <- training(split_full)
test_data  <- testing(split_full)

## ----------------------------------------------------------------------
## 4. Recipe: Preprocessing & Feature Engineering
## ----------------------------------------------------------------------

# A recipe defines all the transformation steps applied prior to model
# training.  Here we:
#  - remove zero‑variance predictors
#  - collapse rare categorical levels
#  - one‑hot encode nominal predictors
#  - centre and scale numeric predictors
#  - deal with class imbalance via SMOTE (Synthetic Minority
#    Over‑Sampling Technique)

churn_recipe <- recipe(retained_binary ~ ., data = train_data) %>%
  step_zv(all_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01, other = "OTHER") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(retained_binary)

## ----------------------------------------------------------------------
## 5. Model Specifications
## ----------------------------------------------------------------------

# Define several machine learning models using `parsnip`.  We leave the
# hyperparameters set to defaults or tuneable values where appropriate.

log_reg_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("classification")

xgb_spec <- boost_tree(
  trees = tune(), tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(),
  sample_size = tune(), mtry = tune(), min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# You can add additional models such as k‑NN, neural networks, or
# LightGBM (via the bonsai package) if those packages are installed.

## ----------------------------------------------------------------------
## 6. Workflow Set & Tuning
## ----------------------------------------------------------------------

# Create a workflow set to manage multiple models with the same recipe.
wf_set <- workflow_set(
  preproc = list(churn_recipe),
  models = list(
    logistic_regression = log_reg_spec,
    random_forest      = rf_spec,
    xgboost            = xgb_spec
  )
)

# Define resampling strategy: 5‑fold cross‑validation with stratification
set.seed(98)
cv_splits <- vfold_cv(train_data, v = 5, strata = retained_binary)

# Set tuning grid sizes; adjust as needed.  Larger grids will produce
# better models but increase runtime.
rf_grid  <- grid_random(mtry(range = c(5, 20)), min_n(range = c(2, 20)), size = 10)
xgb_grid <- grid_latin_hypercube(
  trees(range = c(100, 500)), tree_depth(range = c(1, 10)), learn_rate(range = c(0.01, 0.3)),
  loss_reduction(range = c(0, 1)), sample_size(range = c(0.5, 1.0)), mtry(range = c(5, 20)),
  min_n(range = c(2, 20)), size = 20
)

# Tune the models
registerDoParallel()
tuned_results <- wf_set %>%
  workflow_map(
    "tune_grid",
    resamples = cv_splits,
    grid = list(
      logistic_regression = 1,  # no tuning for logistic regression
      random_forest       = rf_grid,
      xgboost             = xgb_grid
    ),
    metrics = metric_set(roc_auc, accuracy, sensitivity, specificity, precision, f_meas)
  )

## ----------------------------------------------------------------------
## 7. Model Selection & Final Fit
## ----------------------------------------------------------------------

# Extract the best hyperparameters for each model based on ROC AUC
best_results <- tuned_results %>%
  mutate(best = map(result, select_best, metric = "roc_auc")) %>%
  unnest(best)

# Rank models by mean ROC AUC
model_rank <- best_results %>%
  arrange(desc(.metric)) %>%
  select(.config, .estimator, .metric, mean) %>%
  mutate(rank = row_number())

print(model_rank)

# Assume XGBoost performs best; extract workflow and finalize with top
# hyperparameters.  Adjust accordingly if another model wins.
xgb_best_params <- best_results %>%
  filter(.estimator == "xgboost") %>%
  slice(1) %>%
  select(-c(.config, .estimator, .metric))

xgb_final_wf <- workflow()
xgb_final_wf <- xgb_final_wf %>%
  add_recipe(churn_recipe) %>%
  add_model(xgb_spec %>% finalize_model(xgb_best_params))

# Fit the final model on the full training data
xgb_final_fit <- fit(xgb_final_wf, data = train_data)

## ----------------------------------------------------------------------
## 8. Evaluation on Test Set
## ----------------------------------------------------------------------

# Generate predictions and evaluate performance on the hold‑out test set
test_predictions <- predict(xgb_final_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data %>% select(retained_binary))

roc_res  <- roc_auc(test_predictions, truth = retained_binary, .pred_1)
acc_res  <- accuracy(test_predictions, truth = retained_binary, .pred_class = predict(xgb_final_fit, test_data)$.pred_class)
sens_res <- sensitivity(test_predictions, truth = retained_binary, .pred_class = predict(xgb_final_fit, test_data)$.pred_class)
spec_res <- specificity(test_predictions, truth = retained_binary, .pred_class = predict(xgb_final_fit, test_data)$.pred_class)
prec_res <- precision(test_predictions, truth = retained_binary, .pred_class = predict(xgb_final_fit, test_data)$.pred_class)
f1_res   <- f_meas(test_predictions, truth = retained_binary, .pred_class = predict(xgb_final_fit, test_data)$.pred_class)

performance_metrics <- tibble(
  Metric     = c("ROC AUC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score"),
  Score      = c(roc_res$.estimate, acc_res$.estimate, sens_res$.estimate, spec_res$.estimate, prec_res$.estimate, f1_res$.estimate)
)

print(performance_metrics)

## ----------------------------------------------------------------------
## 9. Save Artifacts (Optional)
## ----------------------------------------------------------------------

# Save the final model as an RDS for future use
# saveRDS(xgb_final_fit, file = "models/xgb_final_fit.rds")

# Save the performance table to disk (as CSV)
# readr::write_csv(performance_metrics, file = "model_performance.csv")

# You can also generate and save plots, calibration curves, and
# variable importance using vip().  These plots can be exported to the
# `visuals/` folder of your repository.

## End of script
