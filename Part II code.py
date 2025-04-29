# Import relevant libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import nltk
from nltk.corpus import stopwords
nltk.download ("stopwords")
stop_words = set (stopwords.words("english"))
from collections import defaultdict
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.patches as mpatches
from sklearn.decomposition import TruncatedSVD
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import gensim
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from bertopic import BERTopic
import plotly.io as pio

# Import dataset 
from datasets import load_dataset
df = load_dataset("ucberkeley-dlab/measuring-hate-speech")["train"].to_pandas()

# Drop duplicates
df.drop_duplicates(subset=["text"], inplace=True)

# Target race column
target_race_cols = ['target_race_asian','target_race_black','target_race_latinx','target_race_middle_eastern','target_race_native_american','target_race_pacific_islander','target_race_white','target_race_other','target_race']
target_race = []
for i in range(len(df)):
    for col in target_race_cols:
        if col == "target_race":
            target_race.append("None")        
        elif df.loc[i, col]:
            target_race.append(col[12:].title())
            break
df["Target Race"] = target_race
df.drop(target_race_cols, axis=1, inplace=True)

# Target religion column
target_religion_cols = ['target_religion_atheist',
 'target_religion_buddhist',
 'target_religion_christian',
 'target_religion_hindu',
 'target_religion_jewish',
 'target_religion_mormon',
 'target_religion_muslim',
 'target_religion_other',
 'target_religion']
target_religion = []
for i in range(len(df)):
    for col in target_religion_cols:
        if col == "target_religion":
            target_religion.append("None")        
        elif df.loc[i, col]:
            target_religion.append(col[16:].title())
            break
df["Target Religion"] = target_religion
df.drop(target_religion_cols, axis=1, inplace=True)

# Target gender column
target_gender_cols = ['target_gender_men',
 'target_gender_non_binary',
 'target_gender_transgender_men',
 'target_gender_transgender_unspecified',
 'target_gender_transgender_women',
 'target_gender_women',
 'target_gender_other',
 'target_gender']
target_gender = []
for i in range(len(df)):
    for col in target_gender_cols:
        if col == "target_gender":
            target_gender.append("None")        
        elif df.loc[i, col]:
            target_gender.append(col[14:].title())
            break
df["Target Gender"] = target_gender
df.drop(target_gender_cols, axis=1, inplace=True)

# Target age column
target_age_cols = ['target_age_children',
 'target_age_teenagers',
 'target_age_young_adults',
 'target_age_middle_aged',
 'target_age_seniors',
 'target_age_other',
 'target_age']
target_age = []
for i in range(len(df)):
    for col in target_age_cols:
        if col == "target_age":
            target_age.append("None")        
        elif df.loc[i, col]:
            target_age.append(col[11:].title())
            break
df["Target Age"] = target_age
df.drop(target_age_cols, axis=1, inplace=True)

# Plot showing number of social media posts per group for each target demographic (Figure II.7.3)
sns.set_theme(style="darkgrid")

cols = ["Target Race", "Target Religion", "Target Gender", "Target Age"]

color_maps = [cm.Blues, cm.Greens, cm.Reds, cm.Purples]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i in range(4):
    x = df[df.hate_speech_score > 0.5][cols[i]].value_counts().index[1:]
    y = df[df.hate_speech_score > 0.5][cols[i]].value_counts().values[1:]
    
    norm = plt.Normalize(min(y), max(y))
    colors = color_maps[i](norm(y))
    
    def add_labels(ax, values):
        for i, v in enumerate(values):
            ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    if i < 2:
        axs[0, (i - 1) % 2].bar(x, y, color=colors, edgecolor="black")
        axs[0, (i - 1) % 2].set_xticklabels(x, rotation=90)
        axs[0, (i - 1) % 2].set_ylabel("Count")
        axs[0, (i - 1) % 2].set_title(f"{cols[i]} of social media post")
        add_labels(axs[0, (i-1) % 2], y)
    else:
        axs[1, (i - 1) % 2].bar(x, y, color=colors, edgecolor="black")
        axs[1, (i - 1) % 2].set_xticklabels(x, rotation=90)
        axs[1, (i - 1) % 2].set_ylabel("Count")
        axs[1, (i - 1) % 2].set_title(f"{cols[i]} of social media post")
        add_labels(axs[1, (i-1) % 2], y)
        
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)

# Hate and non-hate corpus
hate_corpus = []
for post in df[df.hate_speech_score > 0.5].text.str.split():
    for word in post:
        hate_corpus.append(word)

non_hate_corpus = []
for post in df[df.hate_speech_score < -1].text.str.split():
    for word in post:
        non_hate_corpus.append(word)

# Plot showing top 10 most common words in hate and non-hate corpus (Figure II.7.2)
hate_word_count = defaultdict(int)
non_hate_word_count = defaultdict(int)

for word in hate_corpus:
    if word not in stop_words:
        hate_word_count[word] += 1

for word in non_hate_corpus:
    if word not in stop_words:
        non_hate_word_count[word] += 1
        
hate_10 = sorted(hate_word_count.items(), key=lambda x:x[1],reverse=True)[1:11]
non_hate_10 = sorted(non_hate_word_count.items(), key=lambda x:x[1],reverse=True)[1:11]

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

dicts = [hate_10, non_hate_10]
labels = ["Top 10 common words in hate posts", "Top 10 common words in non-hate posts"]

for i in range(2):
    word, count = zip(*dicts[i])
    axs[i].bar(word, count)
    axs[i].set_xticklabels(word)
    axs[i].set_ylabel("Count")
    axs[i].set_title(labels[i])
    add_labels(axs[i], count)

plt.tight_layout()

# Summary statistics of number of words per class
df["No. of words"] = df.text.apply(lambda x: len(x.split()))
df[df.hate_speech_score < -1]["No. of words"].describe()
df[df.hate_speech_score > 0.5]["No. of words"].describe()

# Summary statistics of average word length per class
def avg_word(tweet):
    lengths = [len(word) for word in tweet.split()]
    return round(sum(lengths) / len(lengths), 2)
df["Average word length"] = df.text.apply(avg_word)
df[df.hate_speech_score < -1]["Average word length"].describe()
df[df.hate_speech_score > 0.5]["Average word length"].describe()

# Stop words percentage for each class
hate_stop_count = 0
non_hate_stop_count = 0
for word in hate_corpus:
    if word in stop_words:
        hate_stop_count += 1
for word in non_hate_corpus:
    if word in stop_words:
        non_hate_stop_count += 1       
hate_stop_count / len(hate_corpus) , non_hate_stop_count / len(non_hate_corpus)

# Word normalisation
df.text = df.text.str.lower()
minus_punctuation = []
for post in df.text:
    minus_punctuation.append(re.sub(r"[^\w\s]","" ,post))
df.text = minus_punctuation

# Stop word removal
minus_stopwords = []
for post in df.text:
    new_post = " ".join([word for word in post.split() if word not in stop_words])
    minus_stopwords.append(new_post)
df.text = minus_stopwords

# Lemmatisation
lemmatiser = WordNetLemmatizer()
tweet_lemma = []
for post in df.text:
    tweet_lemma.append(" ".join(lemmatiser.lemmatize(word) for word in post.split()))
df.text = tweet_lemma

# Adding labels (1:Hate speech, 0:Non-hate Speech, 2:Ambiguous speech)
label = []
for score in df.hate_speech_score:
    if score < -1:
        label.append(0)
    elif score > 0.5:
        label.append(1)
    else:
        label.append(2)
df["Labels"] = label

# Train-test split
list_corpus = df[df.Labels != 2]["text"].tolist()
list_labels = df[df.Labels != 2]["Labels"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=42)

# TF-IDF
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer
X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Latent Semantic Analysis for hateful social media posts (Figure II.7.4)
def plot_LSA(test_data, test_labels, plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]
        colors = ['orange','blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            orange_patch = mpatches.Patch(color='orange', label='Non-Hate Speech')
            blue_patch = mpatches.Patch(color='blue', label='Hate Speech')
        plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})

fig = plt.figure(figsize=(16, 16))          
plot_LSA(X_train_tfidf, y_train)

# Vectorising posts and extracting corpus for hate and non-hate posts
h_posts = df[df.Labels == 1].text.tolist()
h_posts = [post.split(",")[0].split() for post in h_posts]

n_h_posts = df[df.Labels == 0].text.tolist()
n_h_posts = [post.split(",")[0].split() for post in n_h_posts]

h_id2word = Dictionary(h_posts)
h_corpus = [h_id2word.doc2bow(post) for post in h_posts]

n_h_id2word = Dictionary(n_h_posts)
n_h_corpus = [n_h_id2word.doc2bow(post) for post in n_h_posts]

# Calculating and visualising coherence scores to determine optimal number of topics (Figure II.7.5)
scores = []
for i in range(1, 6):
  n_h_lda_model = LdaModel(corpus=n_h_corpus,
               id2word=n_h_id2word,
               num_topics=i,
               random_state=0,
               chunksize=100,
               alpha='auto',
               per_word_topics=True)
  coherence_model_cv = CoherenceModel(model=n_h_lda_model, texts=n_h_posts, dictionary=n_h_id2word, coherence='c_v')
  scores.append(coherence_model_cv.get_coherence())

sns.set(style="whitegrid", palette="muted")
plt.plot([1, 2, 3, 4, 5], scores, marker='o', color='b', markersize=8, linewidth=2)
plt.xlabel('Number of topics', fontsize=10)
plt.ylabel('Coherence score', fontsize=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Latent Dirichlet Allocation for hateful social media posts (Figures II.7.6, II.7.7 and II.7.8)
h_lda_model = LdaModel(corpus=h_corpus,
               id2word=h_id2word,
               num_topics=3,
               random_state=0,
               chunksize=100,
               alpha='auto',
               per_word_topics=True)
vis = gensimvis.prepare(h_lda_model, h_corpus, h_id2word)
pyLDAvis.display(vis)

# Fitting BERTopic model
hate_df = df[df.Labels == 1]
non_hate_df = df[df.Labels == 0]

hate_corpus_2 = hate_df.text.tolist()
non_hate_corpus_2 = non_hate_df.text.tolist()

def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer ()
    train = tfidf_vectorizer.fit_transform (data)
    return train , tfidf_vectorizer

hate_tfidf , tfidf_vectorizer_hate = tfidf(hate_corpus_2)
non_hate_tfidf , tfidf_vectorizer_non_hate = tfidf(non_hate_corpus_2)

hate_BERT = BERTopic()
hate_topic, hate_probs = hate_BERT.fit_transform(hate_corpus_2)

# Intertopic distance map for BERTopic (Figure II.7.9)
fig = hate_BERT.visualize_topics()

fig.update_layout(
    title_text="BERTopic: Topic Visualisation of hateful posts",
    title_x=0.5,
    font=dict(size=14),
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white"  # Use a clean background
)

pio.show(fig)

# Word scores for top 10 topics generated by BERTopic model (Figure II.7.10)
hate_BERT.visualize_barchart(top_n_topics=10)

# Similarity matrix for topics generated by BERTopic model (Figure II.7.11)
hate_BERT.visualize_heatmap()

# Dendogram for topics generated by BERTopic model (Figure II.7.12)
hate_BERT.visualize_hierarchy()