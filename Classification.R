# Chapter 4 - Classification #

library(tidyverse)
library(class) # for KNN
library(MASS) # for LDA and QDA

if(!file.exists("./adult.data")){
  
  fileUrl <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"  
  
  download.file(fileUrl, destfile = "./adult.data")
  
}

if(!file.exists("./adult.test")){
  
  fileUrl <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"  
  
  download.file(fileUrl, destfile = "./adult.test")
  
}

db.adult <- read_table("adult.data")

income_data <- read_csv("data/income_evaluation.csv", col_names = TRUE, col_types = cols(
  # define the character columns as factors
    age = col_double(),
    workclass = col_factor(),
    fnlwgt = col_double(),
    education = col_factor(),
    `education-num` = col_double(),
    `marital-status` = col_factor(),
    occupation = col_factor(),
    relationship = col_factor(),
    race = col_factor(),
    sex = col_factor(),
    `capital-gain` = col_double(),
    `capital-loss` = col_double(),
    `hours-per-week` = col_double(),
    `native-country` = col_factor(),
    income = col_factor()
  )) %>% 
  mutate(
    
    immigrant = ifelse(`native-country` != "United-States", 1, 0)
    
  )

attach(income_data)

# creating test set
income_test <- sample_n(income_data, )

# LOGISTIC REGRESSION


logistic1 <- glm(income ~ race + education + sex + `hours-per-week` + occupation + immigrant, data = income_data, family = binomial)

summary(logistic1)


contrasts(income) # shows us break down of y variable

logistic_probs <- predict(logistic1, type = "response")

logistic_predictions <- rep("<=50K", nrow(income_data))
logistic_predictions[logistic_probs > 0.5] <- ">50K"

# confusion matrix - main diagonal is correct predictions
table(logistic_predictions, income)

mean(logistic_predictions == income)

summary(income)


# LINEAR DISCRIMINANT ANALYSIS

lda1 <- lda(income ~ race + education + sex + `hours-per-week` + occupation + immigrant, data = income_data)

lda_predictions <- predict(lda1, income)

lda_class <- lda_predictions$class

table(lda_class, income)

mean(lda_class == income)



# QUADRATIC DISCRIMINANT ANALYSIS

qda1 <- qda(income ~ race + `hours-per-week` + occupation + sex + immigrant, data = income_data) # removed education because it was too correlated with race

qda_predictions <- predict(qda1, income)

qda_class <- qda_predictions$class

table(qda_class, income)

mean(qda_class == income)


# K NEAREST NEIGHBOURS
set.seed(1) # need to run this so that it can break ties when points are equidistant
predictors <- cbind(race, education, `hours-per-week`, occupation, sex, immigrant)


knn1 <- knn(predictors, predictors, income, k = 10)

table(knn1, income)

mean(knn1 == income)


