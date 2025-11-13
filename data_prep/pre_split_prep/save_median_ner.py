import csv
import re
import sys
from collections import Counter
from pathlib import Path
import string
import numpy as np
import random
# -------------------------------
# Stopwords and utilities
# -------------------------------
STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "at", "by", "to", "for", "and",
    "is", "it", "he", "she", "we", "they", "be", "or", "as", "la", "le",
    "el", "un", "una", "no", "not"
}

# -------------------------------
    # Regex for @mentions (stop at comma or line end)
    # -------------------------------
pattern_at = re.compile(r'@\s*([A-Za-z][^,\n]*)')
pattern_hash = re.compile(r'#\w+')  
radio_pattern = re.compile(
    r'\b[KW][A-Z]{2,3}(?:[-\s]?(?:FM|AM|HD\d)?)?\s?\d{2,3}(?:\.\d+)?\b',
    re.IGNORECASE
)
traffic_keywords = ["accident", "slow", "block", "clear", "crash", "jam", "road", "delay", "construction", "collision", "obstruction", "stall"]
traffic_patterns = [(a, "traffic") for a in traffic_keywords] + [("traffic", b) for b in traffic_keywords]
phone_pattern = re.compile(r'\b(?:\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{3,4})\b')


def has_traffic_pattern(text):
    """Check if tweet contains any traffic-related co-occurrence."""
    text_norm = normalize_text(text)
    for w1, w2 in traffic_patterns:
        if w1 in text_norm and w2 in text_norm:
            return True
    return False




def has_radio_station(text):
    return bool(radio_pattern.search(text))

def has_phone_pattern(text):
    return bool(phone_pattern.search(text))

def filter_prefix_keywords(keywords):
    """Keep only the largest keyword if smaller one is prefix."""
    sorted_kws = sorted(keywords, key=len, reverse=True)
    final = []
    for kw in sorted_kws:
        #if not any(k.startswith(kw) and k != kw for k in final):
        final.append(kw)
    return set(final)

def normalize_text(text):
    """Lowercase, collapse spaces, remove surrounding punctuation."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip(string.punctuation + " ")

def match_keyword_in_text(keyword, text):
    """Return True if keyword (with spaces normalized) exists in text, including hashtags."""
    kw_norm = " ".join(keyword.lower().split())
    text_norm = normalize_text(text)
    if kw_norm in text_norm:
        return True
    if f"#{kw_norm.replace(' ', '')}" in text_norm.replace(' ', ''):
        return True
    return False

def adaptive_rate(base_small, base_large, total):
    """Interpolate sampling rate between small and large datasets."""
    if total < 1000:
        return base_small
    elif total < 5000:
        return (base_small + base_large) / 2
    elif total < 20000:
        return base_large
    else:
        return base_large / 2
    
def dynamic_threshold(freq, kw):
    """Compute dynamic multiplier threshold based on keyword and frequency characteristics."""
    base = 1.5  # base multiplier
    length_factor = 1.0 + max(0, 5 - len(kw)) * 0.15  # shorter keywords → higher tolerance
    freq_factor = 1.0 + (1 / (freq + 1)) * 3.0        # low frequency → higher tolerance
    # Cap to avoid extreme leniency
    threshold = min(6.0, base * length_factor * freq_factor)
    return threshold

# def compute_weight(row):
#         length_factor = min(len(row["cleaned"]) / 100, 2.0)
#         return 0.5 + 0.5 * length_factor

# -------------------------------
# CLI / setup
# -------------------------------
def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    input_path = Path(input_file)
    if not input_path.is_file():
        print(f"Error: file '{input_file}' does not exist.")
        sys.exit(1)

    output_mentions = input_path.with_name(input_path.stem + "_mentions.csv")
    # output_no_mentions = input_path.with_name(input_path.stem + "_no_mentions.csv")
    hashtag_file = input_path.with_name(input_path.stem + "_hashtags.txt") 
    tweet_usage_counter = Counter()



    # -------------------------------
    # First pass: extract @mentions
    # -------------------------------
    keyword_counter = Counter()
    all_rows = []


    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if "cleaned" not in reader.fieldnames:
            print("Error: CSV missing 'cleaned' column")
            sys.exit(1)

        fieldnames = reader.fieldnames + ["keywords", "freq_sum"]

        for row in reader:
            text = row["cleaned"]
            matches = pattern_at.findall(text)
            kws_in_tweet = []
            for m in matches:
                # Clean and normalize
                cleaned = " ".join(m.strip().split())  # collapse multiple spaces
                cleaned = cleaned.strip("@").strip()
                cleaned = cleaned.rstrip("\u2026")  # remove trailing ellipsis
                if (
                    cleaned
                    and not cleaned[0].isdigit()
                    and len(cleaned) >= 3
                    and cleaned.lower() not in STOPWORDS
                ):
                    keyword_counter[cleaned] += 1
                    kws_in_tweet.append(cleaned)

            
            row["keywords"] = ", ".join(kws_in_tweet)
            row["freq_sum"] = ""
            all_rows.append(row)
    


    traffic_rows = [r for r in all_rows if has_traffic_pattern(r["cleaned"])]
    radio_rows = [r for r in all_rows if has_radio_station(r["cleaned"])]
    # Preserve top phone-number-like area codes (never dropped)
    
    phone_rows = [r for r in all_rows if has_phone_pattern(r["cleaned"])]
    # Combine and deduplicate
    preserved_rows = {id(r): r for r in (traffic_rows + radio_rows+phone_rows)}.values()

    all_radios = []
    all_phones = []

    for r in radio_rows:
        radios = radio_pattern.findall(r["cleaned"])
        if radios:
            all_radios.extend(radios)

    for r in phone_rows:
        phones = phone_pattern.findall(r["cleaned"])
        if phones:
            all_phones.extend(phones)

    # Deduplicate while preserving order
    all_radios = list(dict.fromkeys(all_radios))
    all_phones = list(dict.fromkeys(all_phones))

    



    # -------------------------------
    # Remove truncated/prefix duplicates
    # -------------------------------
    filtered_keywords = filter_prefix_keywords(keyword_counter.keys())
    filtered_keywords = {kw for kw in filtered_keywords if len(kw) >= 3 and kw.lower() not in STOPWORDS}


    # Remove trailing ellipsis from keywords before building expanded map
    #filtered_keywords = {kw.rstrip("\u2026") for kw in filtered_keywords}

    keyword_counter = Counter({k: keyword_counter[k] for k in filtered_keywords})
    hashtag_counter = Counter()

    # -------------------------------
    # Map smaller → larger keywords
    # -------------------------------


    expanded_map = {}
    for kw in keyword_counter.keys():
        candidates = [other for other in keyword_counter.keys() if other.lower().startswith(kw.lower()) and other != kw]
        if candidates:
            largest_kw = max(candidates, key=len)
            expanded_map[kw] = largest_kw


    # -------------------------------
    # Classify tweets
    # -------------------------------
    mentions_rows = []
    no_mentions_rows = []

    for row in all_rows:
        text = row["cleaned"]

        # Tweets with @ mentions
        if "@" in text:
            kws_in_tweet = row["keywords"].split(", ") if row["keywords"] else []
            final_kws = []
            for kw in kws_in_tweet:
                kw_norm = kw.strip()
                # pick largest keyword starting with kw_norm
                candidates = [k for k in keyword_counter.keys() if k.lower().startswith(kw_norm.lower())]
                if candidates:
                    largest_kw = max(candidates, key=len)
                    final_kws.append(largest_kw)
            final_kws = sorted(set(final_kws))
            row["keywords"] = ", ".join(f"{kw} ({keyword_counter[kw]})" for kw in final_kws)
            row["freq_sum"] = sum(keyword_counter[kw] for kw in final_kws)
            mentions_rows.append(row)
            continue

        # Non-mention tweets
        matched = set()
        for kw in keyword_counter.keys():
            # Map to largest if exists
            full_kw = expanded_map.get(kw, kw)
            if match_keyword_in_text(kw, text) or match_keyword_in_text(full_kw, text):
                matched.add(full_kw)

        if matched:
            matched = sorted(matched)
            row["keywords"] = ", ".join(f"{kw} ({keyword_counter[kw]})" for kw in matched)
            row["freq_sum"] = sum(keyword_counter[kw] for kw in matched)
            mentions_rows.append(row)
        else:
            hashtags = pattern_hash.findall(text) 
            hashtag_counter.update([h.lower() for h in hashtags]) 
            row["keywords"] = ""
            row["freq_sum"] = ""
            no_mentions_rows.append(row)




    for row in mentions_rows:
        if row.get("keywords"):
            # extract keywords before parentheses (e.g., "Little Rock (5)" -> "Little Rock")
            for part in row["keywords"].split(","):
                kw = part.strip().split(" (")[0]
                if kw:
                    tweet_usage_counter[kw] += 1


    # Remove duplicates from mentions/no-mentions before merging
    mentions_rows = [r for r in mentions_rows if id(r) not in preserved_rows]
    no_mentions_rows = [r for r in no_mentions_rows if id(r) not in preserved_rows]


    total_tweets = len(all_rows)
    non_mention_rows_len=len(no_mentions_rows)
    # Define base sampling percentages (smaller for large datasets)
    hash_sample_pct = adaptive_rate(0.15, 0.1, non_mention_rows_len)
    nohash_sample_pct = adaptive_rate(0.1, 0.05, non_mention_rows_len)

    # Split non-mention tweets by presence of hashtags
    no_mentions_with_hash = [r for r in no_mentions_rows if pattern_hash.search(r["cleaned"])]
    no_mentions_no_hash = [r for r in no_mentions_rows if not pattern_hash.search(r["cleaned"])]



    hash_counter = Counter()
    for r in no_mentions_with_hash:
        hashtags = pattern_hash.findall(r["cleaned"])
        hash_counter.update(hashtags)


    if no_mentions_with_hash:
        hashtag_freqs = np.array([
            sum(hash_counter[h] for h in pattern_hash.findall(r["cleaned"]))
            for r in no_mentions_with_hash
        ])
        probs = hashtag_freqs / hashtag_freqs.sum() if hashtag_freqs.sum() > 0 else None

        sample_size = max(1, int(len(no_mentions_with_hash) * hash_sample_pct))
        if probs is not None and len(no_mentions_with_hash) > sample_size:
            sampled_hash = list(np.random.choice(
                no_mentions_with_hash,
                size=sample_size,
                replace=False,
                p=probs
            ))
        else:
            sampled_hash = no_mentions_with_hash
    else:
        sampled_hash = []

    sampled_nohash = random.sample(
        no_mentions_no_hash,
        max(1, int(len(no_mentions_no_hash) * nohash_sample_pct))
    ) if no_mentions_no_hash else []

    


    # Combine sampled and preserved rows
    no_mentions_rows = list({id(r): r for r in sampled_hash + sampled_nohash}.values())


    freq_values = list(keyword_counter.values())
    median_kw_freq = np.median(freq_values) if freq_values else 0.0


    


    high_rows, low_rows = [], []



    for row in mentions_rows:
        text = row.get("cleaned", "")
        handles = [m.strip("@") for m in re.findall(r'@\s*([A-Za-z][^,\n\s]*)', text)]
        # if len(handles)>0:
        #     print("handles",handles)

        # Check if any handle frequency > median
        high_freq = any(
            keyword_counter.get(h, 0) > median_kw_freq
            for h in handles
        )

        if high_freq:
            high_rows.append(row)
        else:
            low_rows.append(row)






   

    # if high_rows:
    #     weights = np.array([compute_weight(r) for r in high_rows])
    #     weights = weights / weights.sum()
    #     sample_size = max(1, int(len(high_rows) * sample_pct))
    #     sample_size = min(sample_size, len(high_rows))  # avoid oversampling

    #     # weighted sample *without replacement*
    #     sampled_high = list(np.random.choice(high_rows, size=sample_size, replace=False, p=weights))
    # else:
    #    sampled_high = []


    if high_rows:
    # Compute how far each tweet's handle frequency is above the median
        above_median_scores = []
        for r in high_rows:
            text = r.get("cleaned", "")
            handles = [m.strip("@") for m in re.findall(r'@\s*([A-Za-z][^,\n\s]*)', text)]
            # Take max frequency among handles for scaling
            if handles:
                max_freq = max(keyword_counter.get(h, 0) for h in handles)
            else:
                max_freq = 0
            # how far above median (normalized)
            diff = max(0, max_freq - median_kw_freq)
            above_median_scores.append(diff)

        # Avoid all-zero weights
        if sum(above_median_scores) == 0:
            weights = np.ones(len(high_rows)) / len(high_rows)
        else:
            weights = np.array(above_median_scores) / sum(above_median_scores)

        # Adjust sampling percentage depending on total tweet count
        sample_pct = adaptive_rate(0.9, 0.6, total_tweets)

        # Fewer samples for much higher frequencies
        # Scale down based on how large the average "above median" gap is
        avg_gap_factor = 1.0 - min(0.7, np.mean(above_median_scores) / (median_kw_freq + 1))
        sample_size = max(1, int(len(high_rows) * sample_pct * avg_gap_factor))
        sample_size = min(sample_size, len(high_rows))

        # Weighted sampling without replacement
        sampled_high = list(np.random.choice(high_rows, size=sample_size, replace=False, p=weights))
    else:
        sampled_high = []


    mentions_rows = low_rows + sampled_high + list(preserved_rows)


    with open(output_mentions, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mentions_rows + no_mentions_rows)

    with open(hashtag_file, "w", encoding="utf-8") as f:

            total = len(mentions_rows) + len(no_mentions_rows)
            f.write(f"Mentions tweets: {len(mentions_rows)}")
            f.write("\n")
            f.write(f"No-mentions tweets: {len(no_mentions_rows)}")
            f.write("\n")
            f.write(f"Original total: {len(all_rows)} | Combined: {total}")

            f.write("\n")
        
            f.write("\n")


             # --- Write most common keywords ---
            f.write("## Most common keywords:\n")
            for kw, count in keyword_counter.most_common(30):
                f.write(f"{kw}: {count}\n")
            f.write("\n")
            f.write("\n")

            f.write("## Radio stations found:\n")
            if all_radios:
                for r in all_radios:
                    f.write(f"{r}\n")
            else:
                f.write("(none found)\n")
            f.write("\n")
            f.write("\n")

            f.write("## Phone numbers found:\n")
            if all_phones:
                for p in all_phones:
                    f.write(f"{p}\n")
            else:
                f.write("(none found)\n")
            f.write("\n")
            f.write("\n")


            f.write("# Keywords with unusually high tweet usage compared to their mention frequency\n")
            f.write("# Format: keyword: freq=..., used_in_tweets=..., threshold=..., ratio=...\n\n")

            results = []
            for kw, freq in keyword_counter.items():
                used_in = tweet_usage_counter.get(kw, 0)
                threshold = dynamic_threshold(freq, kw)
                ratio = used_in / freq if freq else float('inf')

                if ratio > threshold:
                    results.append((kw, freq, used_in, threshold, ratio))

            for kw, freq, used_in, threshold, ratio in sorted(results, key=lambda x: -x[4]):
                f.write(f"{kw}: freq={freq}, used_in_tweets={used_in}, threshold≈{threshold:.2f}, ratio={ratio:.2f}\n")

            f.write("\n")
            # --- Write hashtags ---
            f.write("## Hashtags:\n")
            for tag, count in sorted(hash_counter.items(), key=lambda x: (-x[1], x[0].lower())):
                f.write(f"{tag}: {count}\n")

            

    # -------------------------------
    # Summary
    # -------------------------------
    total = len(mentions_rows) + len(no_mentions_rows)
    print(f"Mentions tweets: {len(mentions_rows)}")
    print(f"No-mentions tweets: {len(no_mentions_rows)}")
    print(f"Original total: {len(all_rows)} | Combined: {total}")
    if total == len(all_rows):
        print("Counts match.")
    else:
        print("Warning: total mismatch!")

    print("\nTop filtered keywords:")
    for kw, count in keyword_counter.most_common(20):
        print(f"  {kw}: {count}")

    


if __name__ == "__main__":
        main()

