import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot
import pandas as pd

# Load processed data
train_data = pd.read_parquet("data/processed/processed_ratings.parquet")
num_users = train_data["userId"].nunique()
num_movies = train_data["movieId"].nunique()
embedding_size = 50

# Define model
user_input = Input(shape=(1,), name="user")
movie_input = Input(shape=(1,), name="movie")
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size)(movie_input)
user_vec = Flatten()(user_embedding)
movie_vec = Flatten()(movie_embedding)
dot_product = Dot(axes=1)([user_vec, movie_vec])
model = Model(inputs=[user_input, movie_input], outputs=dot_product)

# Compile and train
model.compile(optimizer="adam", loss="mse")
model.fit([train_data["userId"], train_data["movieId"]], train_data["rating"], epochs=10, batch_size=64)

# Save model
model.save("models/trained_models/recommendation_model.h5")