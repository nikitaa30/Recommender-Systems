#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
data_path = 'ml-20m/'
movies_filename = 'movies.csv'
ratings_filename = 'ratings.csv'

df_movies = pd.read_csv(
    os.path.join(data_path, movies_filename),
#     movies_filename,
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})

df_ratings = pd.read_csv(
    os.path.join(data_path, ratings_filename),
#     ratings_filename,
    usecols=['userId', 'movieId', 'rating'],
    
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})


# In[2]:


# ratings = np.array(ratings_list)
# # users = np.array(users_list)
# movies = np.array(movies_list)


# In[3]:


df_movies.head()


# In[4]:


df_ratings.head()


# In[5]:


df_ratings=df_ratings[:2000000]
df_movie_features = df_ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)


# In[6]:


df_movie_features.head()


# In[7]:


R = df_movie_features.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)


# In[8]:


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)


# In[9]:


# Done. 
# that the Sigma$ returned is just the values instead of a diagonal matrix. 
# This is useful, but since I'm going to leverage matrix multiplication to get predictions 
# I'll convert it to the diagonal matrix form.

sigma = np.diag(sigma)


# In[10]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)


# In[11]:


preds_df = pd.DataFrame(all_user_predicted_ratings, columns = df_movie_features.columns)
preds_df.head()


# In[20]:


# def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=10):
    
#     # Get and sort the user's predictions
#     user_row_number = userID - 1 # UserID starts at 1, not 0
#     sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) # UserID starts at 1
#     print(sorted_user_predictions)
#     # Get the user's data and merge in the movie information.
#     user_data = original_ratings_df[original_ratings_df.userId == (userID)]
#     user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
#                      sort_values(['rating'], ascending=False)
#                  )

# #     print ('User {0} has already rated {1} movies').format(userID, user_full.shape[0])
# #     print( 'Recommending highest {0} predicted ratings movies not already rated.').format(num_recommendations)
    
#     # Recommend the highest predicted rating movies that the user hasn't seen yet.
#     recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
#          merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left').rename(columns = {user_row_number: 'Predictions'}).
#          sort_values('Predictions', ascending = False).
#                        iloc[:num_recommendations, :-1]
#                       )

#     return user_full, recommendations


def recommend_movies(preds_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) # UserID starts at 1
#     print(preds_df.iloc[user_row_number])
#     print(sorted_user_predictions)
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )
#     print(user_full)
#     print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
#     print 'Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    #                left_on = 'movieId',
#                right_on = 'movieId').
# merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left').rename(columns = {user_row_number: 'Predictions'}).
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])]).merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', left_on = 'movieId',
               right_on = 'movieId').rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1]
                      

    return user_full, recommendations


# In[21]:


already_rated, predictions = recommend_movies(preds_df, 330, df_movies, df_ratings, 10)


# In[22]:


already_rated.head(10)


# In[23]:


predictions


# In[ ]:




