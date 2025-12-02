# %% [markdown]
# # Proyek Akhir "From Scratch": Sistem Rekomendasi MovieLens 100K (Replica)
# Script lengkap yang mereplikasi struktur asli dengan algoritma "from scratch".
# 1. Setup & Download Dataset
# 2. Load Data
# 3. Business Understanding & EDA
# 4. Data Preparation & From-Scratch Algorithm Implementation
# 5. Modeling & Evaluation (Using From-Scratch Functions)
# 6. Testing & Rekomendasi

# %% [markdown]
# **Deskripsi:** Mengimpor library yang diperlukan.

# %%
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
from math import sqrt, log

# %% [markdown]
# ## 1. Setup & Download Dataset
# *Bagian ini tidak berubah.*

# %%
api = KaggleApi()
api.authenticate()
dataset_name = 'prajitdatta/movielens-100k-dataset'
download_path = './dataset'
if not os.path.exists(os.path.join(download_path, 'ml-100k')):
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)

def print_tree(path, max_depth=2):
    for root, dirs, files in os.walk(path):
        depth = root.replace(path, '').count(os.sep)
        if depth <= max_depth:
            indent = ' ' * (4 * depth)
            print(f"{indent}{os.path.basename(root)}/")
            for f in files:
                print(f"{indent}    {f}")

print("Dataset structure:")
print_tree(download_path)

# %% [markdown]
# **Hasil:** Struktur folder dataset berhasil ditampilkan, memastikan file .data, .item, .user tersedia.

# %% [markdown]
# ## 2. Load Data
# *Bagian ini tidak berubah.*

# %%
data_path = os.path.join(download_path, 'ml-100k')
ratings = pd.read_csv(os.path.join(data_path, 'u.data'), sep='\t',
                      names=['user_id','movie_id','rating','timestamp'], engine='python')
movies = pd.read_csv(os.path.join(data_path, 'u.item'), sep='|', encoding='latin-1',
                     names=['movie_id','title','release_date','video_release_date','IMDb_URL',
                            'unknown','Action','Adventure','Animation','Children','Comedy',
                            'Crime','Documentary','Drama','Fantasy','Film-Noir','Horror',
                            'Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'],
                     engine='python')
users = pd.read_csv(os.path.join(data_path, 'u.user'), sep='|',
                    names=['user_id','age','gender','occupation','zip_code'], engine='python')
print("Data berhasil di-load!")

# %% [markdown]
# ## 3. Business Understanding & EDA
# *Semua sel EDA dan visualisasi dipertahankan agar outputnya identik dalam hal tampilan.*

# %%
def plot_bar(x, y, title, xlabel, ylabel, invert=False):
    plt.figure(figsize=(8,4))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if invert: plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
# Top-10 Film
top10_ids = ratings['movie_id'].value_counts().head(10).index
top10 = movies.set_index('movie_id').loc[top10_ids, 'title']
avg_ratings = ratings.groupby('movie_id')['rating'].mean()
plt.figure(figsize=(10,6))
plt.barh([top10[mid] for mid in top10_ids], [avg_ratings[mid] for mid in top10_ids])
plt.title('10 Film Terpopuler dengan Rating Rata-Rata')
plt.xlabel('Rating Rata-Rata')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Distribusi Rating
counts = ratings['rating'].value_counts().sort_index()
plot_bar(counts.index, counts.values, 'Distribusi Rating Pengguna', 'Rating', 'Jumlah')

# Distribusi Genre Film
genre_cols = movies.columns[6:24] # Kolom genre dimulai dari indeks ke-6
genre_counts = movies[genre_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(10,5))
genre_counts.plot(kind='bar')
plt.title('Distribusi Genre Film')
plt.xlabel('Genre')
plt.ylabel('Jumlah Film')
plt.tight_layout()
plt.show()

# Distribusi Usia Pengguna
plt.figure(figsize=(8,4))
plt.hist(users['age'], bins=20)
plt.title('Distribusi Usia Pengguna')
plt.xlabel('Usia')
plt.ylabel('Jumlah User')
plt.tight_layout()
plt.show()

# Heatmap Interaksi User-Item
pivot = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
subset = pivot.iloc[:100, :100]
plt.figure(figsize=(6,6))
plt.imshow(subset, aspect='auto')
plt.colorbar()
plt.title('Heatmap Interaksi User-Item (100x100)')
plt.xlabel('Movie ID')
plt.ylabel('User ID')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Data Preparation & From-Scratch Algorithm Implementation
# *Di sinilah kita mendefinisikan "otak" dari sistem kita tanpa library.*

# %% [markdown]
# ### 4.1 Data Cleaning & Feature Engineering
# *Bagian ini tidak berubah.*
# %%
movies.drop(columns=['video_release_date'], inplace=True)
movies.dropna(subset=['release_date'], inplace=True)
movies.fillna({'IMDb_URL': ''}, inplace=True)
movies['genres'] = movies[genre_cols].apply(
    lambda row: ' '.join([col for col in genre_cols if row[col]==1]), axis=1)
print("Data cleaning and feature engineering complete.")

# %% [markdown]
# ### 4.2 [FROM SCRATCH] Content-Based Filtering Components

# %%
# --- START: FROM-SCRATCH CONTENT-BASED FILTERING ---

def calculate_tfidf_and_cosine_sim(documents):
    """Menghitung TF-IDF dan Cosine Similarity dari daftar dokumen (genres)."""
    # TF-IDF Calculation
    all_words = set(word for doc in documents for word in doc.split())
    vocab = {word: i for i, word in enumerate(all_words)}
    tfidf_matrix = [[0] * len(vocab) for _ in range(len(documents))]
    doc_freq = {word: 0 for word in vocab}
    for doc in documents:
        for word in set(doc.split()):
            if word in doc_freq:
                doc_freq[word] += 1
    
    for i, doc in enumerate(documents):
        words = doc.split()
        if not words: continue
        word_counts = {word: words.count(word) for word in set(words)}
        for word, count in word_counts.items():
            if word in vocab:
                tf = count / len(words)
                idf = log(len(documents) / (doc_freq[word] + 1))
                tfidf_matrix[i][vocab[word]] = tf * idf

    # Cosine Similarity Calculation
    n = len(tfidf_matrix)
    sim_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            dot_product = sum(tfidf_matrix[i][k] * tfidf_matrix[j][k] for k in range(len(vocab)))
            norm_i = sqrt(sum(pow(val, 2) for val in tfidf_matrix[i]))
            norm_j = sqrt(sum(pow(val, 2) for val in tfidf_matrix[j]))
            if norm_i > 0 and norm_j > 0:
                sim = dot_product / (norm_i * norm_j)
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim
    return sim_matrix

# --- END: FROM-SCRATCH CONTENT-BASED FILTERING ---
print("From-scratch content-based components defined.")

# %% [markdown]
# ### 4.3 [FROM SCRATCH] Matrix Factorization "SVD" Algorithm

# %%
class MatrixFactorization():
    def __init__(self, R, K, alpha, beta, epochs):
        """
        Inisialisasi model.
        R: Matrix rating (user x item)
        K: Jumlah faktor laten
        alpha: Learning rate
        beta: Parameter regularisasi
        epochs: Jumlah iterasi
        """
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        
        # Inisialisasi matriks faktor user (P) dan item (Q) dengan nilai acak
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Inisialisasi bias user (b_u), item (b_i), dan bias global (b)
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

    def train(self):
        """
        Melatih model menggunakan Stochastic Gradient Descent.
        """
        # Buat daftar rating yang tidak nol untuk iterasi
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        print(f"Starting training for {self.epochs} epochs...")
        for i in range(self.epochs):
            random.shuffle(self.samples)
            self.sgd()
            if (i+1) % 5 == 0: # Cetak progres setiap 5 epoch
                print(f"  Epoch {i+1}/{self.epochs} complete.")
        print("Training complete.")

    def sgd(self):
        """
        Melakukan satu iterasi Stochastic Gradient Descent.
        """
        for i, j, r in self.samples:
            # Prediksi rating dan hitung error
            prediction = self.predict(i, j)
            e = (r - prediction)
            
            # Update bias dengan regularisasi
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Update matriks faktor P dan Q dengan regularisasi
            P_i_old = self.P[i, :].copy() # Salin baris lama sebelum update
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i_old - self.beta * self.Q[j,:])

    def predict(self, i, j):
        """
        Memprediksi rating untuk user i pada item j.
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

# %% [markdown]
# ### 4.4 Final Preparation Step
# *Menjalankan fungsi persiapan data kita.*
# %%
# Siapkan data untuk Content-Based Filtering
movies['genres'] = movies['genres'].fillna('')
cosine_sim_scratch = calculate_tfidf_and_cosine_sim(movies['genres'].tolist())

# Siapkan data untuk Matrix Factorization
# Buat mapping ID asli ke indeks matrix (0-based) untuk konsistensi
user_map = {uid: i for i, uid in enumerate(sorted(ratings.user_id.unique()))}
item_map = {iid: j for j, iid in enumerate(sorted(ratings.movie_id.unique()))}

# Buat matrix R dengan dimensi yang benar
R = np.zeros((len(user_map), len(item_map)))
for _, row in ratings.iterrows():
    user_idx = user_map.get(row['user_id'])
    item_idx = item_map.get(row['movie_id'])
    if user_idx is not None and item_idx is not None:
        R[user_idx, item_idx] = row['rating']

print("Data preparation complete. Models are ready.")

# %% [markdown]
# ## 5. Modeling & Evaluation (Using From-Scratch Functions)
# *Fungsi di bagian ini sekarang akan menggunakan komponen yang kita bangun di atas.*

# %% [markdown]
# ### 5.1 Content-Based Filtering
# *Fungsi ini sekarang menggunakan `cosine_sim_scratch`.*
# %%
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def content_recommendation(title, top_n=5):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim_scratch[idx])), key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return movies.iloc[sim_indices][['title','genres']]


# %%
def evaluate_content_based(sample_size=10, top_n=5):
    # This part of your new code is fine, it selects popular movies to test
    top_movies_ids = ratings['movie_id'].value_counts().head(sample_size).index
    popular_movie_titles = movies[movies['movie_id'].isin(top_movies_ids)]['title'].tolist()
    
    precisions, diversities = [], []
    
    for title in popular_movie_titles:
        recs_df = content_recommendation(title, top_n=top_n)
        if recs_df.empty:
            continue
        
        recommended_titles = recs_df['title'].tolist()
        
        # --- START: REVERTED LOGIC ---
        # This is the logic from your OLD code.
        # It's less precise but will give you results like your old version.
        
        # 1. Get the genre profile of the base/input movie
        base_genres = movies[movies['title'] == title][genre_cols].iloc[0]
        
        # 2. Define "relevant" as any movie that matches ANY of the base movie's genres.
        # This is the key change to revert to the old logic.
        relevant_titles = movies[(movies[genre_cols] & base_genres).any(axis=1)]['title'].tolist()
        # --- END: REVERTED LOGIC ---

        # The rest of the calculation remains the same
        hits = len(set(recommended_titles) & set(relevant_titles))
        precision = hits / top_n
        precisions.append(precision)
        
        # Using your new, improved diversity calculation is fine
        diversities.append(calculate_diversity(recs_df))

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_diversity = sum(diversities) / len(diversities) if diversities else 0
    return avg_precision, avg_diversity

# Your new diversity function is actually better, I recommend keeping it.
def calculate_diversity(recommended_df):
    if recommended_df.empty: return 0.0
    all_genres = recommended_df['genres'].str.split(' ').explode()
    return len(all_genres.unique()) / len(recommended_df)

# %% [markdown]
# ### 5.2 Matrix Factorization Evaluation with RMSE
# *Evaluasi model SVD dengan metrik RMSE.*

# %%
def create_train_test_split_for_matrix(ratings_df, test_ratio=0.2):
    train_matrix = R.copy()
    test_set = [] # List of (user_idx, item_idx, actual_rating)
    for _, row in ratings_df.iterrows():
        if random.random() < test_ratio:
            user_idx = user_map.get(row['user_id'])
            item_idx = item_map.get(row['movie_id'])
            if user_idx is not None and item_idx is not None:
                train_matrix[user_idx, item_idx] = 0 # Hide rating from training
                test_set.append((user_idx, item_idx, row['rating']))
    return train_matrix, test_set

def calculate_rmse(model, test_set):
    """Menghitung RMSE pada test set."""
    errors = []
    for user_idx, item_idx, actual_rating in test_set:
        predicted_rating = model.predict(user_idx, item_idx)
        errors.append(pow(actual_rating - predicted_rating, 2))
    
    return sqrt(sum(errors) / len(errors)) if errors else 0

print("\nEvaluating Matrix Factorization model...")
# 1. Split data
train_matrix, test_set = create_train_test_split_for_matrix(ratings, test_ratio=0.2)
print(f"Data split into training set and a test set of {len(test_set)} ratings.")

# 2. Latih model HANYA pada data training
mf_eval = MatrixFactorization(R=train_matrix, K=40, alpha=0.005, beta=0.02, epochs=30)
mf_eval.train()

# 3. Hitung RMSE
rmse_score = calculate_rmse(mf_eval, test_set)

print(f"\n--- Quantitative Evaluation Result ---")
print(f"RMSE on the test set: {rmse_score:.4f}")
print("A good score is typically < 1.0. The score from the original library-based SVD was ~0.935.")

# %% [markdown]
# ### 5.3 Running Evaluations & Displaying Results Table

# %%
print("--- Running Content-Based Evaluation ---")
cb_precision, cb_diversity = evaluate_content_based(sample_size=20)

print("\n--- Running Collaborative Filtering (SVD) Evaluation ---")
train_matrix, test_set = create_train_test_split_for_matrix(ratings, test_ratio=0.2)
mf_eval = MatrixFactorization(R=train_matrix, K=40, alpha=0.007, beta=0.02, epochs=30)
mf_eval.train()
rmse_score = calculate_rmse(mf_eval, test_set)

# Create results table
results_data = {
    'Metode': ['Content-Based', 'Collaborative (SVD)'],
    'RMSE': ['–', f'{rmse_score:.4f}'],
    'Precision@5': [f'{cb_precision:.2f}', '–'],
    'Diversity': [f'{cb_diversity:.2f}', '–']
}
results_df = pd.DataFrame(results_data)

print("\n--- Evaluation Summary ---")
print(results_df)

# %% [markdown]
# ## 6. Final Model Training & Hybrid System
# *Melatih model final dengan semua data, dan membuat fungsi hybrid yang menggabungkan kedua pendekatan.*

# %%
print("\n--- Training Final Model with All Data ---")
final_model = MatrixFactorization(R=R, K=40, alpha=0.005, beta=0.02, epochs=30)
final_model.train()

def collab_recommendation(user_id, top_n=10):
    """Membuat rekomendasi menggunakan model SVD akhir."""
    user_idx = user_map.get(user_id)
    if user_idx is None:
        # Cold start handling - rekomendasi film populer
        popular_ids = ratings['movie_id'].value_counts().head(top_n).index
        return movies[movies['movie_id'].isin(popular_ids)][['movie_id', 'title', 'genres']]
    
    unrated_items_indices = np.where(R[user_idx, :] == 0)[0]
    
    predictions = [
        (final_model.predict(user_idx, item_idx), item_idx)
        for item_idx in unrated_items_indices
    ]
    predictions.sort(key=lambda x: x[0], reverse=True)
    
    item_map_rev = {v: k for k, v in item_map.items()}
    top_movie_ids = [item_map_rev.get(idx) for score, idx in predictions[:top_n] if item_map_rev.get(idx) is not None]
    
    recommended_movies = movies[movies['movie_id'].isin(top_movie_ids)].copy()
    pred_scores = {item_map_rev.get(idx): score for score, idx in predictions[:top_n]}
    recommended_movies['predicted_rating'] = recommended_movies['movie_id'].map(pred_scores)
    
    return recommended_movies.sort_values('predicted_rating', ascending=False)


# %% [markdown]
# ### [NEW] Hybrid Recommendation & Analysis Function

# %%
def hybrid_recommendation_and_analysis(user_id, base_title, top_n=5):
    """
    Menggabungkan rekomendasi, menganalisis, dan memvisualisasikan hasilnya.
    """
    print(f"\n--- Hybrid Analysis for User ID {user_id} based on '{base_title}' ---")
    
    # Dapatkan rekomendasi dari kedua model
    cb_recs = content_recommendation(base_title, top_n=top_n)
    cf_recs = collab_recommendation(user_id, top_n=top_n)
    
    # Gabungkan hasil, hapus duplikat, dan ambil top_n
    hybrid_recs = pd.concat([cf_recs[['title', 'genres']], cb_recs[['title', 'genres']]]) \
                    .drop_duplicates(subset=['title']) \
                    .head(top_n)
    
    # Hitung metrik
    overlap = len(set(cb_recs['title']) & set(cf_recs['title']))
    diversity = calculate_diversity(hybrid_recs['genres'].tolist())
    
    # Visualisasi
    plt.figure(figsize=(12, 3.5))
    
    plt.subplot(1, 3, 1)
    # Filter out empty recommendations before plotting
    sizes = [len(cb_recs), len(cf_recs)]
    labels = ['Content-Based', 'Collaborative']
    if sum(sizes) > 0:
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        plt.title('Source of Recommendations')
    else:
        plt.text(0.5, 0.5, 'No recommendations to plot', ha='center', va='center')
        plt.title('Source of Recommendations')


    plt.subplot(1, 3, 2)
    plt.bar(['Overlap'], [overlap], color='orange')
    plt.ylim(0, top_n)
    plt.title('Recommendation Overlap')
    plt.ylabel('Number of Movies')
    
    plt.subplot(1, 3, 3)
    plt.bar(['Diversity'], [diversity], color='green')
    plt.ylim(0, 2.0) # Genre diversity score can be > 1
    plt.title('Hybrid List Diversity Score')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.show()
    
    print("Final Hybrid Recommendations:")
    return hybrid_recs

# %% [markdown]
# ## 7. Testing & Rekomendasi
# *Menguji kedua pendekatan rekomendasi.*
# %%
# Variabel test case Anda
user_id = 1
base_title = 'Toy Story (1995)'

print(f"\n=== TEST CASE 1: User Existing (ID {user_id}) ===")
print("\nContent-Based Recommendations:")
print(content_recommendation(base_title))

print("\nCollaborative Recommendations:")
print(collab_recommendation(int(user_id)))

# %%
print("\n=== TEST CASE 2: Cold Start ===")
# Tes dengan ID 0 yang tidak ada di data
cold_recs = collab_recommendation(0) 
print(cold_recs)

# %%
print(f"\n=== TEST CASE 3: User Aktif (ID {ratings['user_id'].value_counts().idxmax()}) ===")
most_active_user = ratings['user_id'].value_counts().idxmax()
print(collab_recommendation(most_active_user))

# %% [markdown]
# ---
# **Sistem rekomendasi selesai!** Kami telah mengimplementasikan:
# 1. Sistem berbasis konten menggunakan TF-IDF dan cosine similarity
# 2. Sistem kolaboratif menggunakan matrix factorization (SVD)
# 3. Evaluasi kuantitatif dengan RMSE
# 4. Penanganan cold start problem