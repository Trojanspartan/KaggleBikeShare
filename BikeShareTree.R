library(rpart)
library(tidymodels)
library(dplyr)
library(tidyr)

my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model6
  set_engine("rpart") %>% # What R function to use7
  set_mode("regression")

rec_tree <- recipe(count ~ ., data = train_glmnet) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors())

tree_wf <- workflow() %>%
  add_recipe(rec_tree) %>%
  add_model(my_mod)

set.seed(42)
folds <- vfold_cv(train_glmnet, v = 5)
grid_tree <- tidyr::crossing(
  cost_complexity = c(0.00001, 0.0001, 0.001, 0.01, 0.1),
  tree_depth      = c(2, 4, 6, 8, 12),
  min_n           = c(2, 5, 10, 20, 40)
)

tree_tuned <- tune_grid(
  tree_wf,
  resamples = folds,
  grid      = grid_tree,
  metrics   = metric_set(rmse),
  control   = control_grid(save_pred = FALSE)
)

tree_metrics <- tree_tuned %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  arrange(mean)

readr::write_csv(tree_metrics, "tree_cv_metrics.csv")

best_params <- tree_metrics %>% slice(1) %>% dplyr::select(cost_complexity, tree_depth, min_n)

final_wf <- finalize_workflow(tree_wf, best_params)
final_fit <- final_wf %>% fit(train_glmnet)

pred_log <- predict(final_fit, new_data = test) %>% pull(.pred)
pred_cnt <- pmax(0, exp(pred_log) - 1)

stopifnot(nrow(ss) == length(pred_cnt))
submission <- tibble(datetime = ss$datetime,
                     count    = as.integer(round(pred_cnt)))
readr::write_csv(submission, "submission_tree_best.csv")