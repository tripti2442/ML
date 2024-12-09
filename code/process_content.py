import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

def process_user_features(user_file, output_file):
    users = {}
    with open(user_file, 'r') as f:
        for line in f:
            fields = line.strip().split('::')  # Use '::' as the delimiter
            if len(fields) != 5:  # Skip lines with incorrect number of fields
                print("Skipping malformed line: {}".format(line.strip()))
                continue
            user_id, gender, age, occupation, _ = fields
            try:
                users[int(user_id)] = [gender, int(age), occupation]
            except ValueError as e:
                print("Skipping line due to error: {} - {}".format(line.strip(), e))
                continue

    # Encode categorical features
    gender_encoder = LabelEncoder()
    occupation_encoder = LabelEncoder()

    genders = [v[0] for v in users.values()]
    occupations = [v[2] for v in users.values()]

    gender_encoded = gender_encoder.fit_transform(genders)
    occupation_encoded = occupation_encoder.fit_transform(occupations)

    # Create final feature vectors
    user_features = {}
    for idx, (gender, age, occupation) in enumerate(zip(gender_encoded, [v[1] for v in users.values()], occupation_encoded)):
        user_features[idx + 1] = np.array([gender, age, occupation], dtype=np.float32)

    with open(output_file, 'wb') as f:
        pickle.dump(user_features, f, protocol=2)  # Use protocol=2 for Python 2 compatibility

def process_movie_features(movie_file, output_file):
    movies = {}
    with open(movie_file, 'r') as f:
        for line in f:
            fields = line.strip().split('::')  # Use '::' as the delimiter
            if len(fields) < 3:  # Ensure sufficient fields
                print("Skipping malformed line: {}".format(line.strip()))
                continue
            movie_id, title, genres = fields[0], fields[1], fields[2]
            genre_list = genres.split('|')  # Genres are separated by '|'
            movies[int(movie_id)] = genre_list

    # Encode genres as a multi-hot vector
    all_genres = set(g for movie_genres in movies.values() for g in movie_genres)
    genre_encoder = {genre: idx for idx, genre in enumerate(all_genres)}

    movie_features = {}
    for movie_id, genres in movies.items():
        genre_vector = np.zeros(len(all_genres), dtype=np.float32)
        for g in genres:
            genre_vector[genre_encoder[g]] = 1.0
        movie_features[movie_id] = genre_vector

    with open(output_file, 'wb') as f:
        pickle.dump(movie_features, f, protocol=2)  # Use protocol=2 for Python 2 compatibility

if _name_ == "_main_":
    process_user_features('../data/ml-1m/users.dat', '../data/ml-1m/user_profile.pk')
    process_movie_features('../data/ml-1m/movies.dat', '../data/ml-1m/movie_profile.pk')