## Initial Setup

# Set Working Directory
setwd("C:/Users/colby/OneDrive/Desktop/STAT 348/RestaurantRevenue")

# Call Libraries
library(tidyverse)
library(vroom)
library(tidymodels)
library(dplyr)
library(poissonreg)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)
library(janitor)
library(embed)
library(bonsai)
library(lightgbm)

# Read in data
train_data <- vroom("train.csv")

train_data <- train_data %>%
  clean_names()

test_data <- vroom("test.csv")

test_data <- test_data %>%
  clean_names()

n_observed <- nrow(test_data)

for (i in 1:n_observed){
  if(test_data$type[i] == "MB"){
    test_data$type[i] <- "DT"
  }
}

# Imported Turkey Data Set
turkey <- vroom("Turkey.csv") %>%
  clean_names()

turkey <- turkey %>%
  mutate(income = gsub(" TL", "", turkey$per_capita_annual_income)) %>%
  mutate(education = (as.numeric(gsub("%", "", turkey$number_of_people_with_higher_education_and_above)) / 100)) %>%
  mutate(population = format(as.numeric(gsub("\\.", "", turkey$population)), big.mark = ",")) %>%
  select(c(city, population, income, education))

enhanced_train <- train_data %>%
  left_join(turkey, by = "city")

enhanced_test <- test_data %>%
  left_join(turkey, by = "city")

n_observed <- nrow(enhanced_test)

for (i in 1:n_observed){
  if (is.na(enhanced_test$population[i])){
    enhanced_test$population[i] <- enhanced_test$population[i-1]
  }
  if (is.na(enhanced_test$income[i])){
    enhanced_test$income[i] <- enhanced_test$income[i-1]
  }
  if (is.na(enhanced_test$education[i])){
    enhanced_test$education[i] <- enhanced_test$education[i-1]
  }
}

## Recipe Setup

regression_recipe <- recipe(revenue~., data = train_data) %>%
  step_rm(open_date, id) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(c("city", "city_group"))
  

ridge_model <- linear_reg(penalty = 0, mixture = 1) %>%
  set_engine("lm")

lasso_model <- linear_reg(penalty = 1, mixture = 0) %>%
  set_engine("lm")

elastic_model <- linear_reg(penalty = tune(),
                            mixture = tune()) %>%
  set_engine("lm")

ridge_wf <- workflow() %>%
  add_recipe(regression_recipe) %>%
  add_model(ridge_model) %>%
  fit(data = train_data)

ridge_preds <- predict(ridge_wf, new_data = test_data) %>%
  bind_cols(., test_data) %>%
  rename(revenue = .pred) %>%
  select(id, revenue) %>%
  mutate(revenue = pmax(0, revenue))

vroom_write(ridge_preds, file = "ridge_preds.csv", delim = ",")


lasso_wf <- workflow() %>%
  add_recipe(regression_recipe) %>%
  add_model(lasso_model) %>%
  fit(data = train_data)

lasso_preds <- predict(lasso_wf, new_data = test_data) %>%
  bind_cols(., test_data) %>%
  rename(revenue = .pred) %>%
  select(id, revenue) %>%
  mutate(revenue = pmax(0, revenue))

vroom_write(lasso_preds, file = "lasso_preds.csv", delim = ",")


elastic_wf <- workflow() %>%
  add_recipe(regression_recipe) %>%
  add_model(elastic_model) %>%
  fit(data = train_data)

elastic_preds <- predict(elastic_wf, new_data = test_data) %>%
  bind_cols(., test_data) %>%
  rename(revenue = .pred) %>%
  select(id, revenue) %>%
  mutate(revenue = pmax(0, revenue))

vroom_write(elastic_preds, file = "elastic_preds.csv", delim = ",")

# Random Forest
forest_recipe <- recipe(revenue~., train_data) %>%
  step_mutate(open_date = as.Date(open_date, format  = "%m/%d/%Y"),
              city = as.factor(city),
              city_group = as.factor(city_group),
              type = as.factor(type)) %>%
  update_role(id, new_role = "ID") %>%
  step_date(open_date, features = c("year", "quarter")) %>%
  step_rm(open_date) %>%
  step_other(all_nominal_predictors(), other = 0.05) %>%
  step_zv(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

forest_model <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 5000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

forest_wf <- workflow() %>%
  add_recipe(forest_recipe) %>%
  add_model(forest_model)

tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

cv_results <- forest_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq))

best_tune <- cv_results %>%
  select_best("rmse")

final_wf <- forest_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train_data)

forest_preds <- predict(final_wf, 
                 new_data = test_data) %>%
  bind_cols(., test_data) %>%
  select(id, .pred) %>%
  rename(Id = id) %>%
  rename(Prediction = .pred) %>%
  mutate(Prediction = pmax(0, Prediction))

vroom_write(forest_preds, file = "rand_forest_preds2.csv", delim = ",")

# Enhanced Data Set
forest_recipe <- recipe(revenue~., enhanced_train) %>%
  step_mutate(open_date = as.Date(open_date, format  = "%m/%d/%Y"),
              city = as.factor(city),
              city_group = as.factor(city_group)) %>%
  update_role(id, new_role = "ID") %>%
  step_date(open_date, features = c("year", "quarter")) %>%
  step_rm(open_date) %>%
  step_other(all_nominal_predictors(), other = 0.05) %>%
  step_zv(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

forest_model <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 5000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

forest_wf <- workflow() %>%
  add_recipe(forest_recipe) %>%
  add_model(forest_model)

tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(enhanced_train, v = 5, repeats = 1)

cv_results <- forest_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq))

best_tune <- cv_results %>%
  select_best("rmse")

final_wf <- forest_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = enhanced_train)

forest_preds <- predict(final_wf, 
                        new_data = enhanced_test) %>%
  bind_cols(., enhanced_test) %>%
  select(id, .pred) %>%
  rename(Id = id) %>%
  rename(Prediction = .pred) %>%
  mutate(Prediction = pmax(0, Prediction))

vroom_write(forest_preds, file = "rand_forest_preds_enhanced.csv", delim = ",")

# Boosted Tree Model
boost_recipe <- recipe(revenue~., train_data) %>%
  step_mutate(open_date = as.Date(open_date, format  = "%m/%d/%Y"),
              city = as.factor(city),
              city_group = as.factor(city_group),
              type = as.factor(type)) %>%
  update_role(id, new_role = "ID") %>%
  step_date(open_date, features = c("year", "quarter")) %>%
  step_rm(open_date) %>%
  step_other(all_nominal_predictors(), other = 0.05) %>%
  step_zv(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

boost_wf <- workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_model)

boost_tune_grid <- grid_regular(tree_depth(),
                                trees(),
                                learn_rate(),
                                levels = 5)

folds <- vfold_cv(data = train_data, v = 5, repeats = 1)

tuned_boost <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = boost_tune_grid,
            metrics = metric_set(rmse, mae, rsq))

best_tune_boost <- tuned_boost %>%
  select_best("rmse")

final_boost_wf <- boost_wf %>%
  finalize_workflow(best_tune_boost) %>%
  fit(data = train_data)

boost_preds <- predict(final_boost_wf,
                       new_data = test_data) %>%
  mutate(Prediction = .pred) %>%
  bind_cols(., test_data) %>%
  select(id, Prediction) %>%
  rename(Id = id)

vroom_write(boost_preds, file = "BoostPreds.csv", delim = ",")

