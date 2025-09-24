library(tidymodels)

library(dplyr)
library(lubridate)
library(readr)

train = vroom::vroom("train.csv")

test = vroom::vroom("test.csv")

ss <- vroom::vroom(
  "sampleSubmission.csv",
  col_types = vroom::cols(
    datetime = vroom::col_character(),
    count    = vroom::col_double()
  )
)

train = train %>%
  select(-casual) %>%
  select(-registered) %>%
  mutate(count = log(count))

bike.lm = linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula = count ~ season + holiday + workingday + weather + temp + atemp + humidity + windspeed, data=train)


predictions = predict(bike.lm, new_data = test)$.pred

predictions <- pmax(0, predictions)

submission <- tibble(
  datetime = ss$datetime,
  count    = as.integer(round(predictions))
)

## ------------------------------------------- Workflows HW----------------------

train_clean <- train %>%
  select(-casual, -registered) %>%
  mutate(count = log1p(count))

rec <- recipe(count ~ ., data = train_clean) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_mutate(hour = hour(ymd_hms(datetime))) %>%
  step_mutate(season = factor(season)) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_rm(datetime)

baked_train <- prep(rec) %>% bake(new_data = NULL)
print(head(baked_train, 5))

lm_spec <- linear_reg() %>% set_engine("lm")
wf <- workflow() %>% add_recipe(rec) %>% add_model(lm_spec)
fit_wf <- wf %>% fit(train_clean)

pred_log  <- predict(fit_wf, new_data = test) %>% dplyr::pull(.pred)
pred_cnt  <- pmax(0, exp(pred_log) - 1) 

ss <- vroom::vroom("sampleSubmission.csv",
                   col_types = vroom::cols(
                     datetime = vroom::col_character(),
                     count    = vroom::col_double()))
stopifnot(nrow(ss) == length(pred_cnt))
ss$count <- pmax(0L, as.integer(round(pred_cnt)))

readr::write_csv(ss, "submissionWorkflow.csv")

##-----------------------Penalized Regression----------------------------

