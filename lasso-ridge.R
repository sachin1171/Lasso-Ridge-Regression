####################### problem 1###########################
library(glmnet)
startups=read.csv(file.choose())
attach(startups)
x <- model.matrix(Profit ~ ., data = startups)[,-1]
y <- startups$Profit

grid <- 10^seq(10, -2, length = 100)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)
# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)
sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)
####################### problem 2 #######################
# Read data from file
Computer_Data <- read.csv(file.choose())
colnames(Computer_Data)
# Reorder the variables and removing columns "x" 
Computer_Data <- Computer_Data[ ,c(2,3,4,5,6,7,8,9,10)]
str(Computer_Data)

library(glmnet)
x <- model.matrix(price ~ ., data = Computer_Data )[ ,-1]
y <- Computer_Data $price

grid <- 10^seq(12, -2, length = 100)
grid
# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)

optimumlambda <- cv_fit$lambda.min
optimumlambda

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

pred <- predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)
predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)
#root mean squared error
error <- y - y_a
rmse <- sqrt(mean(error**2))
rmse
# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)

optimumlambda_1 <- cv_fit_1$lambda.min
optimumlambda_1

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

pred <- predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)
predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

#root mean squared error
error <- y - y_a

rmse <- sqrt(mean(error**2))
rmse
############################# problem 3 ###############################
library(glmnet)
library(readr)
corolla=read.csv(file.choose())
attach(corolla)
x <- model.matrix(Price ~ ., data = corolla)[,-3]
y <- corolla$Price
grid <- 10^seq(10, -2, length = 100)
grid
# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min
y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared
predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)
# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min
y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared
predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)
###################### problem 4 ################################
#load the data
life <- read.csv(file.choose())

View(life)

summary(life)
str(life)
table(life$Status)

# Reorder the variables
life <- life[,-c(1)]

# remove na in r - remove rows - na.omit function / option
life <- na.omit(life)

class(life)
attach(life)
library(glmnet)
x <- model.matrix(Life_expectancy ~ ., data = life)[ ,-1]
y <- life$Life_expectancy
grid <- 10^seq(12, -2, length = 100)
grid
# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min
optimumlambda

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared
predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)
pred <- predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)
#root mean squared error
error <- y - y_a
rmse <- sqrt(mean(error**2))
rmse
# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)
cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min
optimumlambda_1 
y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared
predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)
pred <- predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)
#root mena squared error
error <- y - y_a
rmse <- sqrt(mean(error**2))
rmse
