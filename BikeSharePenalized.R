## ----------------------- Penalized Regression (glmnet) -----------------------
## Requirements: tidymodels, vroom, dplyr, lubridate, readr, purrr, stringr, tibble
library(tidymodels)
library(dplyr)
library(lubridate)
library(readr)
library(vroom)
library(purrr)
library(stringr)
library(tibble)

## Assumes you already loaded: train, test, and ss (sampleSubmission)
##    from your earlier code

# 1) Prep training data (remove leakage cols; log1p the outcome for linear_reg)
train_glmnet <- train %>%
  select(-casual, -registered) %>%
  mutate(count = log1p(count))   # log1p keeps things stable near zero

# 2) Recipe: encode categoricals, engineer hour, normalize numerics
rec_glmnet <- recipe(count ~ ., data = train_glmnet) %>%
  step_mutate(hour = lubridate::hour(lubridate::ymd_hms(datetime, quiet = TRUE))) %>%
  step_mutate(
    season      = factor(season),
    holiday     = factor(holiday),
    workingday  = factor(workingday),
    weather     = factor(weather)
  ) %>%
  step_rm(datetime) %>%                              # remove raw timestamp
  step_dummy(all_nominal_predictors()) %>%           # no categoricals left
  step_impute_median(all_numeric_predictors()) %>%   # just in case
  step_normalize(all_numeric_predictors())           # put on same scale

# 3) Model spec: elastic net via glmnet, with tunable penalty & mixture
glmnet_spec <- linear_reg(
  penalty = tune(),   # lambda
  mixture = tune()    # alpha: 0=ridge, 1=lasso, (0,1)=elastic-net
) %>%
  set_engine("glmnet")

# 4) Workflow
glmnet_wf <- workflow() %>%
  add_recipe(rec_glmnet) %>%
  add_model(glmnet_spec)

# 5) Resampling & grid (≥5 combos; here we try 25 combos)
set.seed(42)
folds <- vfold_cv(train_glmnet, v = 5)
grid_glmnet <- tidyr::crossing(
  penalty = c(1e-4, 1e-3, 1e-2, 1e-1, 1),      # > 0
  mixture = c(0, 0.25, 0.5, 0.75, 1)           # in [0,1]
)

# 6) Tune with RMSE on the log1p(count) scale
glmnet_tuned <- tune_grid(
  glmnet_wf,
  resamples = folds,
  grid      = grid_glmnet,
  metrics   = metric_set(rmse),
  control   = control_grid(save_pred = FALSE)
)

# Save CV metrics so you can see which (penalty, mixture) did best locally
cv_metrics <- glmnet_tuned %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  arrange(mean)
readr::write_csv(cv_metrics, "glmnet_cv_metrics.csv")

# 7) Fit full-data model with the CV-best params; write a "best" submission
best_params <- cv_metrics %>% slice(1) %>% dplyr::select(penalty, mixture)
final_wf <- finalize_workflow(glmnet_wf, best_params)
final_fit <- final_wf %>% fit(train_glmnet)

# Predict on test; invert log1p; clip to nonnegative; Kaggle wants ints
pred_log_best <- predict(final_fit, new_data = test) %>% dplyr::pull(.pred)
pred_cnt_best <- pmax(0, exp(pred_log_best) - 1)

submission_best <- tibble(
  datetime = ss$datetime,
  count    = as.integer(round(pred_cnt_best))
)
readr::write_csv(submission_best, "submission_glmnet_best.csv")

# 8) ALSO: make one CSV per (penalty, mixture) combo so you can submit all
dir.create("submissions_glmnet", showWarnings = FALSE)

param_list <- cv_metrics %>%
  dplyr::select(penalty, mixture) %>%
  distinct()

pwalk(param_list, function(penalty, mixture) {
  wf_i  <- finalize_workflow(glmnet_wf, list(penalty = penalty, mixture = mixture))
  fit_i <- wf_i %>% fit(train_glmnet)
  
  pred_i_log <- predict(fit_i, new_data = test) %>% dplyr::pull(.pred)
  pred_i_cnt <- pmax(0, exp(pred_i_log) - 1)
  
  out <- tibble(
    datetime = ss$datetime,
    count    = as.integer(round(pred_i_cnt))
  )
  
  # tidy filename like: sub_pen0.001_mix0.50.csv
  pen_str <- gsub("[^0-9eE\\.-]", "", format(penalty, scientific = TRUE))
  mix_str <- sprintf("%.2f", mixture)
  fname <- file.path("submissions_glmnet",
                     paste0("sub_pen", pen_str, "_mix", mix_str, ".csv"))
  readr::write_csv(out, fname)
})

# 9) After submitting to Kaggle:
#    - Record the (penalty, mixture) that yields the lowest leaderboard score (LS).
#    - You can paste that into a small tibble for your writeup if you want:
# best_from_kaggle <- tribble(
#   ~penalty, ~mixture, ~leaderboard_score,
#   0.001,     0.50,     0.42   # <--- replace with your actual best
# )


## ====================== HOMEWORK: Penalized Regression =======================
## Requirements: tidymodels, dplyr, readr, vroom, lubridate, purrr, tibble, stringr
library(tidymodels); library(dplyr); library(readr); library(vroom)
library(lubridate); library(purrr); library(tibble); library(stringr)

# Assumes train.csv, test.csv, and sampleSubmission.csv already read as `train`, `test`, `ss`.
# If not, uncomment:
# train <- vroom::vroom("train.csv")
# test  <- vroom::vroom("test.csv")
# ss    <- vroom::vroom("sampleSubmission.csv",
#                       col_types = vroom::cols(datetime=vroom::col_character(),
#                                               count   =vroom::col_double()))

# 1) Prep training data (remove leakage; transform target for linear_reg)
train_glmnet <- train %>%
  select(-casual, -registered) %>%
  mutate(count = log1p(count))   # log1p instead of log to be stable near zero

# 2) Recipe: encode categoricals, engineer hour, normalize numerics
rec_glmnet <- recipe(count ~ ., data = train_glmnet) %>%
  step_mutate(hour = lubridate::hour(lubridate::ymd_hms(datetime, quiet = TRUE))) %>%
  step_mutate(
    season     = factor(season),
    holiday    = factor(holiday),
    workingday = factor(workingday),
    weather    = factor(weather)
  ) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

# 3) Model: elastic net via glmnet (tune penalty λ and mixture α)
glmnet_spec <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

wf <- workflow() %>% add_recipe(rec_glmnet) %>% add_model(glmnet_spec)

# 4) Resamples + grid (≥5 combos; we use 5×5 = 25)
set.seed(42)
folds <- vfold_cv(train_glmnet, v = 5)
grid  <- tidyr::crossing(
  penalty = c(1e-4, 1e-3, 1e-2, 1e-1, 1),
  mixture = c(0, 0.25, 0.50, 0.75, 1)
)

# 5) Tune on RMSE of log1p(count)
tuned <- tune_grid(
  wf, resamples = folds, grid = grid,
  metrics = metric_set(rmse),
  control = control_grid(save_pred = FALSE)
)

# Save full CV table for the write-up
cv_metrics <- tuned %>% collect_metrics() %>% filter(.metric == "rmse") %>% arrange(mean)
readr::write_csv(cv_metrics, "glmnet_cv_metrics.csv")

# 6) Pick best, refit on all training data
best_params <- select_best(tuned, metric = "rmse")
final_wf    <- finalize_workflow(wf, best_params)
final_fit   <- fit(final_wf, data = train_glmnet)

# 7) Predict on test, invert log1p, clip to nonnegative, write Kaggle CSV
pred_log <- predict(final_fit, new_data = test) %>% pull(.pred)
pred_cnt <- pmax(0, exp(pred_log) - 1)

stopifnot(nrow(ss) == length(pred_cnt))
submission <- tibble(datetime = ss$datetime,
                     count    = as.integer(round(pred_cnt)))
readr::write_csv(submission, "submission_glmnet_best.csv")

# 8) Console summary to paste in your LS report
message(sprintf(
  "Best (by CV RMSE): penalty = %s, mixture = %.2f, CV RMSE = %.5f\nSaved: submission_glmnet_best.csv",
  format(best_params$penalty, scientific = TRUE), best_params$mixture,
  cv_metrics %>% semi_join(as_tibble(best_params), by = c("penalty","mixture")) %>% pull(mean)
))

