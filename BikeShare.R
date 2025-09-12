library(tidymodels)

train = vroom::vroom("train.csv")

test = vroom::vroom("test.csv")

ss <- vroom::vroom(
  "sampleSubmission.csv",
  col_types = vroom::cols(
    datetime = vroom::col_character(),
    count    = vroom::col_double()
  )
)

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
