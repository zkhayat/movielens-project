# This code will be creating a movie recommendation system including the best possible RMSE value, using
# the MovieLens dataset.The recommendation system was designed using the 10M version (a small set) of the Movielens set. 

library(tidyverse)
library(caret)
library(data.table)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Split the edx set (training set) into training and validation sets 
edx_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-edx_index,]
edx_test <- edx[edx_index,]

edx_validation <- edx_test %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")
removed <- anti_join(edx_test, edx_validation)
edx_train <- rbind(edx_train, removed)

# y(u,i)= mu+e(u,i)
mu_hat<- mean(edx_train$rating)
mu_hat
naive_rmse <- RMSE(edx_test$rating, mu_hat)
naive_rmse
RMSE_results <- data_frame(method = "Just the average", RMSE = naive_rmse)
RMSE_results %>% knitr::kable() 

# y(u,i)=mu+b_i+e(u,i)
mu<- mean(edx_train$rating)
movie_avgs<-edx_train %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))

predicted_ratings <- mu+edx_validation %>% left_join(movie_avgs, by='movieId') %>% .$b_i

model_bi_rmse <- RMSE(predicted_ratings, edx_validation$rating)
RMSE_results <- bind_rows(RMSE_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_bi_rmse ))
RMSE_results %>% knitr::kable()

# adding the user avg(b_u) factor to the model and calculate the RMSE 
#  y(u,i) = mu + b_i + b_u + e(u,i)


user_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- edx_validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% .$pred


model_bu_rmse <- RMSE(predicted_ratings, edx_validation$rating)
RMSE_results <- bind_rows(RMSE_results,data_frame(method="Movie + User Effects Model",  
                                                  RMSE = model_bu_rmse ))


RMSE_results %>% knitr::kable()

# using Regularization technique to optimize the movie and user effects
lambda <- 4
mu <- mean(edx_train$rating)
movie_reg_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

predicted_ratings <- edx_validation %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_lambda_rmse <- RMSE(predicted_ratings, edx_validation$rating)
RMSE_results <- bind_rows(RMSE_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_lambda_rmse ))
RMSE_results %>% knitr::kable()

# using cross validation to select the best lambda value for best RMSE value

lambdas <- seq(0, 6, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_train$rating)
  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    edx_validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, edx_validation$rating))})


qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

RMSE_results <- bind_rows(RMSE_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
RMSE_results %>% knitr::kable()

# calculating the best value of RMSE for the final Model with Lambda=4.5, edx training set and validation set. 
lambdas <- 4.5
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

RMSE_results<-bind_rows(RMSE_results,data_frame(method="Final Model",RMSE=rmses))
RMSE_results%>%knitr::kable()


