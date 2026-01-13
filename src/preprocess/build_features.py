import pandas as pd
import re
import json
from pathlib import Path
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder

RAW_FILE = Path("./data/raw/new.csv")
OUTPUT_FILE = Path("./data/processed/pheme_features.csv")
MAPPING_FILE = Path("./outputs/label_mappings.json")


def load_data(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def classify_sentiment(text: str) -> str:
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "POSITIVE"
    elif polarity < -0.1:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def sentiment_counts(sentiments):
    return {
        "pos": sum(1 for s in sentiments if s == "POSITIVE"),
        "neg": sum(1 for s in sentiments if s == "NEGATIVE"),
        "neu": sum(1 for s in sentiments if s == "NEUTRAL"),
    }


def process_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data[
        ["text", "in_reply_to_status_id", "id", "favorite_count", "retweeted", "entities_hashtags","entities_urls",
            "retweet_count", "in_reply_to_user_id", "user_id", "user_verified", "user_followers_count", "user_statuses_count", "user_friends_count",
            "user_favourites_count", "user_created_at", "lang", "created_at", "event", "category", "thread_id", "subfolder", "thread_veracity"]].copy()

    
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce")
    data["user_created_at"] = pd.to_datetime(data["user_created_at"], errors="coerce")
    data["account_age_days"] = (data["created_at"] - data["user_created_at"]).dt.days
    data["text_length_words"] = data["text"].fillna("").str.split().str.len()
    data["has_exclamation"] = data["text"].fillna("").str.contains("!").astype(int)
    data["has_question"] = data["text"].fillna("").str.contains(r"\?").astype(int)
    emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    data["has_emoji"] = data["text"].fillna("").apply(lambda x: int(len(emoji_pattern.findall(x)) > 0))
    data["mentions_flag"] = data["text"].fillna("").str.contains(r"@\w+").astype(int)
    data["hashtags_count"] = ( pd.to_numeric(data["entities_hashtags"], errors="coerce").fillna(0).astype(int))
    data["urls_count"] = (pd.to_numeric(data["entities_urls"], errors="coerce").fillna(0).astype(int))
    data["retweets"] = data["retweet_count"]
    data["favorites"] = data["favorite_count"]
    data["is_verified"] = data["user_verified"].astype(int)
    data["followers_count"] = data["user_followers_count"]
    data["friends_count"] = data["user_friends_count"]
    data["favorites_statuses_ratio"] = (data["user_favourites_count"] / (data["user_statuses_count"] + 1))
    data["tweet_event"] = data["event"]
    data["tweet_type"] = data["subfolder"]
    data["classification"] = data["category"]
    #data["language"] = data["lang"]
    data["diffusion_speed"] = data.groupby("thread_id")["created_at"].transform(lambda x: (x.max() - x.min()).total_seconds())
    data["sentiment"] = data["text"].astype(str).apply(classify_sentiment)
    reply_sentiments = (data[data["in_reply_to_status_id"].notnull()].groupby("thread_id")["sentiment"].apply(list))

    reply_counts_map = {
        tid: sentiment_counts(sents) for tid, sents in reply_sentiments.items()
    }

    data["users_pos_count"] = data["thread_id"].map(
        lambda tid: reply_counts_map.get(tid, {}).get("pos", 0)
    )
    data["users_neg_count"] = data["thread_id"].map(
        lambda tid: reply_counts_map.get(tid, {}).get("neg", 0)
    )
    data["users_neu_count"] = data["thread_id"].map(
        lambda tid: reply_counts_map.get(tid, {}).get("neu", 0)
    )
    data = data[data["subfolder"] == "source-tweets"]

    
    data = data.drop(columns=["in_reply_to_status_id","id","favorite_count","retweeted","entities_hashtags", "entities_urls","retweet_count","in_reply_to_user_id",
                              "user_id","user_verified", "user_followers_count","user_statuses_count","user_friends_count","user_favourites_count", "user_created_at",
                              "lang","created_at","event","category","thread_id","subfolder","thread_veracity"])

    mappings = {}
    for col in ["tweet_event", "sentiment", "classification"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        mappings[col] = {
            str(cls): int(code)
            for cls, code in zip(le.classes_, le.transform(le.classes_))
        }
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)

    return data


def main():
    print("Cargando datos raw...")
    data = load_data(RAW_FILE)

    print("Procesando features...")
    data = process_features(data)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)

    print(f"Features guardadas en {OUTPUT_FILE}")
    print(f"Shape final: {data.shape}")
    print(f"Mapeos guardados en {MAPPING_FILE}")


if __name__ == "__main__":
    main()
