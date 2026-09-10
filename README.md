# Run this in Google Colab.
# Step 1: Upload your kaggle.json (Kaggle account -> Settings -> Create New API Token)

# --- Cell 1: setup ---
# !pip install kaggle -q
# from google.colab import files
# files.upload()  # upload kaggle.json
# !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# --- Cell 2: download dataset ---
# !kaggle datasets download -d thoughtvector/customer-support-on-twitter
# !unzip -o customer-support-on-twitter.zip -d data/raw

import pandas as pd

def load_and_filter(brand="SpotifyCares", raw_path="data/raw/twcs/twcs.csv"):
    df = pd.read_csv(raw_path)

    # Identify tweets authored by the brand
    brand_tweets = df[df["author_id"] == brand]

    # Reconstruct customer -> brand reply pairs using in_response_to_tweet_id
    pairs = brand_tweets.merge(
        df,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_brand", "_customer"),
    )

    pairs = pairs[pairs["author_id_customer"] != brand]  # ensure customer-originated

    cols = [
        "tweet_id_customer", "text_customer", "created_at_customer",
        "tweet_id_brand", "text_brand", "created_at_brand",
    ]
    pairs = pairs[cols].rename(columns={
        "text_customer": "customer_msg",
        "text_brand": "brand_reply",
    })

    print(f"Total {brand} reply pairs: {len(pairs)}")
    return pairs


def quick_cluster_preview(pairs, n_clusters=10, sample_size=5000):
    """
    Cheap first pass to see what real intents look like before hand-labeling.
    Uses TF-IDF + KMeans purely for exploration -- NOT the final classifier.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    sample = pairs.sample(min(sample_size, len(pairs)), random_state=42)
    vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=5)
    X = vec.fit_transform(sample["customer_msg"])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sample = sample.copy()
    sample["cluster"] = labels

    for c in range(n_clusters):
        print(f"\n=== Cluster {c} (n={sum(labels == c)}) ===")
        for msg in sample[sample["cluster"] == c]["customer_msg"].head(5):
            print(" -", msg[:120].replace("\n", " "))

    return sample


if __name__ == "__main__":
    pairs = load_and_filter()
    pairs.to_csv("data/processed/spotify_pairs.csv", index=False)
    quick_cluster_preview(pairs)
