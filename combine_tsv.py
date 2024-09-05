import glob
import os
from collections import Counter
from functools import lru_cache

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)


@lru_cache(maxsize=None)
def get_stopwords():
    return set(stopwords.words("english"))


def process_tsv_files():
    tsv_files = glob.glob("*.tsv")
    dfs = []

    for file in tsv_files:
        if file != "all.tsv":
            df = pd.read_csv(file, sep="\t")
            df["person"] = os.path.splitext(file)[0]
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values("start").reset_index(drop=True)
    combined_df.insert(0, "id", range(1, len(combined_df) + 1))
    combined_df = combined_df[["id", "person", "text", "start", "end"]]
    combined_df["duration"] = combined_df["end"] - combined_df["start"]

    combined_df.to_csv("all.tsv", sep="\t", index=False)
    print(f"Combined {len(tsv_files) - 1} TSV files into 'all.tsv'")

    return combined_df


def create_figure(plot_func, filename, alt_text):
    plt.figure(figsize=(10, 6))
    plot_func()

    # Create 'figures' folder if it doesn't exist
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Save figure in the 'figures' folder
    plt.savefig(os.path.join("figures", filename))
    plt.close()
    return f"![{alt_text}](figures/{filename})\n\n*Alt text: {alt_text}*\n\n"


def analyze_data(df):
    report = "# NLP Analysis Report\n\n"

    # Jargon explanations
    report += "## Glossary of Terms\n\n"
    report += "- **Segment**: A portion of the audio transcription, typically representing a continuous speech by a single speaker.\n"
    report += "- **POS (Part of Speech)**: Grammatical category of words, such as noun, verb, adjective, etc.\n"
    report += "- **Common POS Tags**:\n"
    report += "  - NN: Noun, singular\n"
    report += "  - NNS: Noun, plural\n"
    report += "  - VB: Verb, base form\n"
    report += "  - VBD: Verb, past tense\n"
    report += "  - JJ: Adjective\n"
    report += "  - RB: Adverb\n"
    report += "  - IN: Preposition or subordinating conjunction\n"
    report += "  - DT: Determiner\n"
    report += "- **Word Cloud**: A visual representation of word frequency where the size of each word indicates its frequency in the text.\n\n"

    # Basic statistics
    total_duration = (df["end"].max() - df["start"].min()) / 1000
    avg_segment_duration = df["duration"].mean() / 1000

    report += "## Basic Statistics\n\n"
    report += f"- Total transcription duration: {total_duration:.2f} seconds\n"
    report += f"- Number of transcribed segments: {len(df)}\n"
    report += f"- Average segment duration: {avg_segment_duration:.2f} seconds\n"
    report += f"- Earliest timestamp: {df['start'].min() / 1000:.2f} seconds\n"
    report += f"- Latest timestamp: {df['end'].max() / 1000:.2f} seconds\n\n"

    # Speaker analysis
    speaker_counts = df["person"].value_counts()
    report += "## Speaker Analysis\n\n"
    report += "| Speaker | Segment Count |\n|---------|---------------|\n"
    for speaker, count in speaker_counts.items():
        report += f"| {speaker} | {count} |\n"
    report += "\n"

    # Speaker distribution pie chart
    def plot_speaker_dist():
        plt.pie(speaker_counts.values, labels=speaker_counts.index, autopct="%1.1f%%")
        plt.title("Distribution of Speakers")

    alt_text = f"Pie chart showing the distribution of speakers. {', '.join([f'{speaker}: {count}' for speaker, count in speaker_counts.items()])}"
    report += create_figure(plot_speaker_dist, "speaker_distribution.png", alt_text)

    # Text analysis
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].replace("nan", "")
    all_text = " ".join(df["text"])

    words = word_tokenize(all_text.lower())
    stop_words = get_stopwords()
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Word frequency
    word_freq = Counter(words)
    report += "## Word Frequency Analysis\n\n"
    report += "| Word | Frequency |\n|------|----------|\n"
    for word, freq in word_freq.most_common(10):
        report += f"| {word} | {freq} |\n"
    report += "\n"

    # Word cloud
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(word_freq)

    def plot_wordcloud():
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud")

    alt_text = f"Word cloud showing the most frequent words. The largest words are: {', '.join([word for word, _ in word_freq.most_common(5)])}"
    report += create_figure(plot_wordcloud, "wordcloud.png", alt_text)

    # Parts of speech analysis
    pos_tags = pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)

    report += "## Parts of Speech Analysis\n\n"
    report += "| POS Tag | Count | Description |\n|---------|-------|-------------|\n"
    pos_descriptions = {
        "NN": "Noun, singular",
        "NNS": "Noun, plural",
        "VB": "Verb, base form",
        "VBD": "Verb, past tense",
        "JJ": "Adjective",
        "RB": "Adverb",
        "IN": "Preposition or subordinating conjunction",
        "DT": "Determiner",
    }
    for pos, count in pos_counts.most_common():
        description = pos_descriptions.get(pos, "Other")
        report += f"| {pos} | {count} | {description} |\n"
    report += "\n"

    # POS distribution bar chart
    def plot_pos_dist():
        sns.barplot(x=list(pos_counts.keys()), y=list(pos_counts.values()))
        plt.title("Distribution of Parts of Speech")
        plt.xlabel("POS Tag")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

    alt_text = f"Bar chart showing the distribution of parts of speech. The most common POS tags are: {', '.join([f'{pos}: {count}' for pos, count in pos_counts.most_common(3)])}"
    report += create_figure(plot_pos_dist, "pos_distribution.png", alt_text)

    return report


def main():
    df = process_tsv_files()
    report = analyze_data(df)

    with open("nlp_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(
        "NLP analysis complete. Results saved in 'nlp_report.md' and figures saved in 'figures' folder."
    )


if __name__ == "__main__":
    main()
