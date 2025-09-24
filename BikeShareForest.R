library(rpart)
library(tidymodels)
library(dplyr)
library(tidyr)

train_clean <- train %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>% #Type of model6
  set_engine("ranger") %>% # What R function to use7
  set_mode("regression")

## Create a workflow with model & recipe10

rec_forest <- recipe(count ~ ., data = train_clean) %>%
  step_date(datetime, features="dow") %>%
  step_time(datetime, features=c("hour", "minute")) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

rec_prep <- prep(rec_forest)
train_juiced <- juice(rec_prep)
  

tree_wf <- workflow() %>%
  add_recipe(rec_forest) %>%
  add_model(my_mod)

## Set up grid of tuning values12 

maxNumXs <- train_juiced %>% select(-count) %>% ncol()

mygrid <- grid_regular(
  mtry(  range = c(1L, maxNumXs) ),
  min_n( range = c(2L, 50L) ),
  levels = 5
)

## Set up K-fold CV14

folds <- vfold_cv(train_clean, v = 5, repeats = 1)

rf_metrics <- metric_set(rmse, rsq, mae)

rf_tuned <- tune_grid(
  tree_wf,
  resamples = folds,
  grid      = mygrid,
  metrics   = rf_metrics
)

best_params <- select_best(rf_tuned, metric = "rmse")

final_rf_wf <- finalize_workflow(tree_wf, best_params)

final_rf_fit <- fit(final_rf_wf, data = train_clean)

pred_df <- predict(final_rf_fit, new_data = test) %>%
  rename(.pred_log = .pred) %>%
  mutate(count = pmax(0, exp(.pred_log)))

submission <- tibble(datetime = ss$datetime,
                     count    = pred_df$count)
readr::write_csv(submission, "submission_forest_best.csv")
