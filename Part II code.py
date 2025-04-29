# Import relevant libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import nltk
from nltk.corpus import stopwords
nltk.download ("stopwords")
stop_words = set (stopwords.words("english"))
from collections import defaultdict
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import preprocessing
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
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, classification_report, accuracy_score, recall_score, precision_score
from tqdm import tqdm

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

### BERT_Base model
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
device

# Setting model parameters
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 10
LEARNING_RATE = 1e-05
TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME, truncation=True, do_lower_case=True)

# Create a class to preprocess the data to prepare for model implementation
class Dataset_Preprocess(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = OneHotEncoder(sparse_output=False).fit_transform(np.array(self.data["Labels"]).reshape(-1, 1))
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float)
        }

# Split dataset into training, validation and test sets
dataset = df[df.Labels != 2]

train_size = 0.8
val_size = 0.1

train_data = dataset.sample(frac = train_size)
test_data = dataset.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)
val_data = test_data.sample(frac=val_size / (1 - train_size), random_state=220).reset_index()
test_data = test_data.drop(val_data.index).reset_index(drop=True)

print(f"Full Dataset Size: {dataset.shape}")
print(f"Train Dataset Size: {train_data.shape}")
print(f"Validation Dataset Size: {val_data.shape}")
print(f"Test Dataset Size: {test_data.shape}")

training_set = Dataset_Preprocess(train_data, TOKENIZER, MAX_LEN)
validation_set = Dataset_Preprocess(val_data, TOKENIZER, MAX_LEN)
testing_set = Dataset_Preprocess(test_data, TOKENIZER, MAX_LEN)

train_params = {
    "batch_size": BATCH_SIZE,
    "shuffle": True,
    "num_workers": 0
}

val_params = {
    "batch_size": 1,
    "shuffle": False,
    "num_workers": 0
}

test_params = {
    "batch_size": 1,
    "shuffle": False,
    "num_workers": 0
}

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **val_params)
testing_loader = DataLoader(testing_set, **test_params)

# Defining class for BERT_Base model
class BERT_Base(nn.Module):
    def __init__(self, n_classes):
        super(BERT_Base, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
# Instantiating model
num_classes = dataset["Labels"].nunique()
model = BERT_Base(n_classes = num_classes)
model.to(device)

# Defining loss function for model using logistic loss
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# Adam optimiser for backpropagation
optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE)

# Defining function for training model
def train(epoch):
    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)
        # print('ids', type(ids))
        # print('mask', type(mask))
        # print('token type ids', type(token_type_ids))
        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

# Training model over 10 epochs
for epoch in range(EPOCHS):
    train(epoch)

# Validation for model 
def validation(model, loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0)):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

outputs, targets = validation(model, testing_loader)

# Illustrating results (Figure II.8.1 and Table II.8.1)
final_outputs = np.argmax(outputs, axis=1)
targets = np.argmax(targets, axis=1)

print(classification_report(targets, final_outputs))

cm = confusion_matrix(targets, final_outputs)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('BERT_Base Confusion Matrix')

plt.show()

### RoBERTa

# Train, validation, test sets
train_data, temp_data = train_test_split(dataset, test_size=0.2)
dev_data, test_data = train_test_split(temp_data, test_size=0.5)

# Function to calculate f1_score, recall, precision
def calculate_metrics(preds, labels):
    results = dict()
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    results['precision_score'] = precision_score(labels_flat, preds_flat, average='binary')
    results['recall_score'] = recall_score(labels_flat, preds_flat, average='binary')
    results['f1_score'] = f1_score(labels_flat, preds_flat, average='binary')
    return results

# Prepare dataset for use in model
def encode_data(df, tokenizer):
    input_ids = []
    attention_masks = []
    for tweet in df[["text"]].values:
        tweet = tweet.item()
        encoded_data = tokenizer.encode_plus(
                            tweet,
                            add_special_tokens = True,
                            max_length = 128,
                            padding = 'max_length',
                            truncation = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                    )
        input_ids.append(encoded_data['input_ids'])
        attention_masks.append(encoded_data['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    inputs = {
    'input_ids': input_ids,
    'input_mask': attention_masks}
    return inputs

manual_seed = 2022
torch.manual_seed(manual_seed)

def prepare_dataloaders(train_df, dev_df, test_df, model_name, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)

    data_train = encode_data(train_df, tokenizer)
    labels_train = train_df.Labels.astype(int)

    data_valid = encode_data(dev_df, tokenizer)
    labels_valid = dev_df.Labels.astype(int)

    data_test = encode_data(test_df, tokenizer)

    input_ids, attention_masks = data_train.values()
    train_labels = torch.tensor(labels_train.values)
    train_dataset = TensorDataset(input_ids, attention_masks, train_labels)

    input_ids, attention_masks = data_valid.values()
    valid_labels = torch.tensor(labels_valid.values)
    val_dataset = TensorDataset(input_ids, attention_masks, valid_labels)

    input_ids, attention_masks = data_test.values()
    test_dataset = TensorDataset(input_ids, attention_masks)

    train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset),
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_dataset,
                sampler = SequentialSampler(val_dataset),
                batch_size = batch_size
            )

    test_dataloader = DataLoader(
                test_dataset,
                sampler = SequentialSampler(test_dataset),
                batch_size = batch_size
            )

    return train_dataloader, validation_dataloader, test_dataloader

def prepare_model(total_labels, model_name, model_to_load=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = total_labels,
        output_attentions = False,
        output_hidden_states = False,
    )
    if model_to_load is not None:
        try:
            model.roberta.load_state_dict(torch.load(model_to_load))
            print("Loaded pre-trained model")
        except:
            pass
    return model

def prepare_optimizer_scheduler(total_steps, learning_rate=5e-5):
    optimizer = AdamW(model.parameters(),
                    lr = learning_rate,
                    eps = 1e-8,
                    weight_decay = 1e-2
                    )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps
                                                )
    return optimizer, scheduler

def evaluate(model, validation_dataloader):

    model.eval()
    preds = []
    true_labels = []
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in tqdm(validation_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].type(torch.LongTensor).to(device)
        with torch.no_grad():
            outputs = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        preds.append(logits)
        true_labels.append(label_ids)

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    tqdm.write(f"Avg validation loss: {avg_val_loss}")

    return preds, true_labels, avg_val_loss

# Training RoBERTa model
def train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs):
    training_stats = []
    model.train()
    total_train_loss = 0

    for epoch in tqdm(range(1, epochs+1)):
        progress_bar = tqdm(train_dataloader,
                        desc=" Epoch {:1d}".format(epoch),
                        leave=False, # to overwrite each epoch
                        disable=False)

        for batch in progress_bar:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].type(torch.LongTensor).to(device)

            model.zero_grad()
            outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        tqdm.write(f"\nEpoch: {epoch}")
        tqdm.write(f"Training loss: {avg_train_loss}")

        preds, val_labels, avg_val_loss = evaluate(model, validation_dataloader)
        predictions = np.concatenate(preds, axis=0)
        labels = np.concatenate(val_labels, axis=0)

        scores = calculate_metrics(predictions, labels)
        precision = scores['precision_score']*100
        recall = scores['recall_score']*100
        f1 = scores['f1_score']*100
        tqdm.write(f"Precision Score: {precision}")
        tqdm.write(f"Recall Score: {recall}")
        tqdm.write(f"F1 Score: {f1}")
    print("Training complete!")

MODEL = "roberta-base"
BATCH_SIZE = 8

train_dataloader, validation_dataloader, test_dataloader = prepare_dataloaders(
    train_data, dev_data, test_data, model_name=MODEL, batch_size=BATCH_SIZE
)

EPOCHS = 1
NUM_LABELS = len(dataset.Labels.unique())
TOTAL_STEPS = len(train_dataloader) * EPOCHS
model = prepare_model(total_labels=NUM_LABELS, model_name=MODEL, model_to_load=None)
model.to(device)
optimizer, scheduler = prepare_optimizer_scheduler(
    total_steps=TOTAL_STEPS, learning_rate=5e-5
)

train(model, optimizer, scheduler, train_dataloader, validation_dataloader, EPOCHS)



### DistilRoBERTa
MODEL = "distilroberta-base"
BATCH_SIZE = 64

train_dataloader, validation_dataloader, test_dataloader = prepare_dataloaders(
    train_data, dev_data, test_data, model_name=MODEL, batch_size=BATCH_SIZE
)

EPOCHS = 1
LEARNING_RATE = 5e-5
NUM_LABELS = len(dataset.Labels.unique())
TOTAL_STEPS = len(train_dataloader) * EPOCHS
model = prepare_model(total_labels=NUM_LABELS, model_name=MODEL, model_to_load=None)
model.to(device)
optimizer, scheduler = prepare_optimizer_scheduler(
    total_steps=TOTAL_STEPS, learning_rate=LEARNING_RATE
)

train(model, optimizer, scheduler, train_dataloader, validation_dataloader, EPOCHS)

# Column chart showing comparison of recall, precision and F1-score between RoBERTa and DistilRoBERTa (Figure II.8.2)
metrics = ["Precision", "Recall", "F1-score"]
roberta_scores = [0.81, 0.69, 0.75]
distilroberta_scores = [0.82, 0.81, 0.81]

score_df = pd.DataFrame({
    "Metric": metrics * 2,
    "Score": roberta_scores + distilroberta_scores,
    "Model": ["RoBERTa"] * 3 + ["DistilRoBERTa"] * 3
})

sns.set_style("darkgrid")
sns.set_palette("muted")

plt.figure(figsize=(10, 8.5))
ax = sns.barplot(x="Metric", y="Score", hue="Model", data=score_df)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=12, padding=3)

plt.title("Performance Comparison of RoBERTa vs. DistilRoBERTa", fontsize=14, fontweight="bold")
plt.ylim(0.6, 0.85)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Metric", fontsize=12)
plt.legend(title="Model", fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.show()

### BERTweet model

MODEL = "vinai/bertweet-large"
batch_size = 16

def prepare_dataloaders(train_df, dev_df, test_df, model_name, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False, normalization=True)

    data_train = encode_data(train_df, tokenizer)
    labels_train = train_df.Labels.astype(int)

    data_valid = encode_data(dev_df, tokenizer)
    labels_valid = dev_df.Labels.astype(int)

    data_test = encode_data(test_df, tokenizer)
    labels_test = test_df.Labels.astype(int)

    input_ids, attention_masks = data_train.values()
    train_labels = torch.tensor(labels_train.values)
    train_dataset = TensorDataset(input_ids, attention_masks, train_labels)

    input_ids, attention_masks = data_valid.values()
    valid_labels = torch.tensor(labels_valid.values)
    val_dataset = TensorDataset(input_ids, attention_masks, valid_labels)

    input_ids, attention_masks = data_test.values()
    test_labels = torch.tensor(labels_test.values)
    test_dataset = TensorDataset(input_ids, attention_masks, test_labels)

    train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset), 
                batch_size = batch_size 
            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset),
                batch_size = batch_size 
            )

    test_dataloader = DataLoader(
                test_dataset, 
                sampler = SequentialSampler(test_dataset), 
                batch_size = batch_size
            )

    return train_dataloader, validation_dataloader, test_dataloader

train_dataloader, validation_dataloader, test_dataloader = prepare_dataloaders(train_data, dev_data, test_data, model_name=MODEL, batch_size = batch_size)

def prepare_model(total_labels, model_name, model_to_load=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = total_labels,  
        output_attentions = False, 
        output_hidden_states = False,
    )
    if model_to_load is not None:
        try:
            model.roberta.load_state_dict(torch.load(model_to_load))
            print("Loaded pre-trained model")
        except:
            pass
    return model

def prepare_optimizer_scheduler(total_steps, learning_rate=1e-5):
    optimizer = AdamW(model.parameters(),
                    lr = learning_rate,
                    eps = 1e-8,
                    weight_decay = 1e-2
                    )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps
                                                )
    return optimizer, scheduler


EPOCHS = 2
NUM_LABELS = len(dataset.Labels.unique())
TOTAL_STEPS = len(train_dataloader) * EPOCHS
model = prepare_model(total_labels=NUM_LABELS, model_name=MODEL, model_to_load=None)
model.to(device)
optimizer, scheduler = prepare_optimizer_scheduler(total_steps=TOTAL_STEPS, learning_rate=1e-5)

def evaluate(model, validation_dataloader):
    
    model.eval()
    preds = []
    true_labels = []
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in tqdm(validation_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].type(torch.LongTensor).to(device)

        with torch.no_grad():        
            outputs = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
            
        preds.append(logits)
        true_labels.append(label_ids)
        
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    tqdm.write(f"Avg validation loss: {avg_val_loss}")

    return preds, true_labels, avg_val_loss

def train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs):
    training_stats = []
    model.train()
    total_train_loss = 0

    for epoch in tqdm(range(1, epochs+1)):
        progress_bar = tqdm(train_dataloader, 
                        desc=" Epoch {:1d}".format(epoch),
                        leave=False, # to overwrite each epoch
                        disable=False)

        for batch in progress_bar:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].type(torch.LongTensor).to(device)

            model.zero_grad()
            outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            loss = outputs.loss
            logits = outputs.logits

  

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        tqdm.write(f"\nEpoch: {epoch}")
        tqdm.write(f"Training loss: {avg_train_loss}")

        preds, val_labels = evaluate(model, validation_dataloader)
        predictions = np.argmax(np.concatenate(preds, axis=0), axis=1).flatten()
        labels = (np.concatenate(val_labels, axis=0)).flatten()
        
        print(classification_report(labels, predictions))

    print("Training complete!")

train(model, optimizer, scheduler, train_dataloader, validation_dataloader, EPOCHS)

preds, true = evaluate(model, test_dataloader)

predictions = np.argmax(np.concatenate(preds, axis=0), axis=1).flatten()
labels = (np.concatenate(true, axis=0)).flatten()
        
print(classification_report(labels, predictions))

### Testing un-optimised BERTweet model on HateCheck dataset
# Import hatecheck dataset
hatecheck = pd.read_csv("test_suite_cases.csv")

# Pre-processing dataset
groups = ["gay people", "women", "disabled people", "Muslims", "black people", "trans people", "immigrants"]
hatecheck.label_gold = hatecheck.label_gold.map({"hateful": 1, "non-hateful": 0})

minus_stopwords = []

for post in hatecheck.test_case:
    new_post = " ".join([word for word in post.split() if word not in stop_words])
    minus_stopwords.append(new_post)

hatecheck.test_case = minus_stopwords

hatecheck.test_case = hatecheck.test_case.str.lower()

minus_punctuation = []

for post in hatecheck.test_case:
    minus_punctuation.append(re.sub(r"[^\w\s]","" ,post))

hatecheck.test_case = minus_punctuation

lemmatiser = WordNetLemmatizer()

tweet_lemma = []

for post in hatecheck.test_case:
    tweet_lemma.append(" ".join(lemmatiser.lemmatize(word) for word in post.split()))

hatecheck.test_case = tweet_lemma

# Fitting BERTweet to each identity group and printing classification report
predictions = []
true_labels = []

for group in groups:    
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large",use_fast=False, normalization=True)
    
    data = encode_data(hatecheck[hatecheck.target_ident == group], tokenizer)
    labels = hatecheck[hatecheck.target_ident == group].label_gold.astype(int)
    
    input_ids, attention_masks = data.values()
    test_labels = torch.tensor(labels.values)
    test_dataset = TensorDataset(input_ids, attention_masks, test_labels)
    
    test_dataloader = DataLoader(
                test_dataset,
                sampler = RandomSampler(test_dataset), 
                batch_size = 16
            )

    preds, true = evaluate(model, test_dataloader)

    predictions.append(preds)
    true_labels.append(true)

for i in range(len(predictions)):
    preds = np.argmax(np.concatenate(predictions[i], axis=0), axis=1).flatten()
    true = (np.concatenate(true_labels[i], axis=0)).flatten()
            
    print(classification_report(true, preds))

# Column chart for recall, precision, F1-score per identity group in hatecheck dataset (Figure II.8.3)
sns.set_style("darkgrid")

precision = [0.69, 0.75, 0.81, 0.83, 0.81, 0.83, 0.83]
recall = [0.63, 0.58, 0.55, 0.68, 0.68, 0.55, 0.57]
f1_score = [0.64, 0.6, 0.58, 0.71, 0.7, 0.57, 0.59]

num_groups = len(groups)

x = np.arange(num_groups)
bar_width = 0.25

colors = sns.color_palette("coolwarm", 3)

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - bar_width, precision, width=bar_width, label='Precision', color=colors[0])
bars2 = plt.bar(x, recall, width=bar_width, label='Recall', color=colors[1])
bars3 = plt.bar(x + bar_width, f1_score, width=bar_width, label='F1 Score', color=colors[2])

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", 
                 ha='center', fontsize=8)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.xlabel('Target Identity Group', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Across Identity Groups', fontsize=14)
plt.xticks(x, groups, rotation=30, ha="right")
plt.ylim(0, 1.1)
plt.legend()

plt.tight_layout()
plt.show()

# Line chart showing change in validation and training errors for different number of epochs (Figure II.8.4)
epochs = [1, 2, 3, 4, 5]
validation_error = [1.011, 1.068, 1.068, 1.068, 1.068]
training_error = [0.024, 0.026, 0.028, 0.029, 0.031]

sns.set_style("darkgrid")
palette = sns.color_palette("coolwarm", 2)

plt.figure(figsize=(8, 5))
plt.plot(epochs, training_error, label="Training Error", marker="o", color=palette[0])
plt.plot(epochs, validation_error, label="Validation Error", marker="s", color=palette[1])

for i, (tr_err, te_err) in enumerate(zip(training_error, validation_error)):
    plt.text(epochs[i], tr_err + 0.02, f'{tr_err:.3f}', ha='center', va='bottom', fontsize=10)
    plt.text(epochs[i], te_err - 0.02, f'{te_err:.3f}', ha='center', va='top', fontsize=10)

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Training vs Validation Error")
plt.legend()
plt.show()

# Line chart showing change in validation and test errors for different dropout rates (Figure II.8.5)
dropout = [0.3, 0.5, 0.7]
validation_error = [0.307, 0.295, 0.295]
test_error = [0.283, 0.261, 0.276]

sns.set_style("darkgrid")
palette = sns.color_palette("coolwarm", 2)

plt.figure(figsize=(8, 5))
plt.plot(dropout, validation_error, label="Validation Error", marker="o", color=palette[0])
plt.plot(dropout, test_error, label="Test Error", marker="s", color=palette[1])

for i in range(len(dropout)):
    plt.text(dropout[i], validation_error[i] - 0.003, f'{validation_error[i]:.3f}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold', color=palette[0])
    
    plt.text(dropout[i], test_error[i] + 0.003, f'{test_error[i]:.3f}', 
             ha='center', va='top', fontsize=10, fontweight='bold', color=palette[1])

plt.xlabel("Dropout Rate")
plt.ylabel("Error")
plt.title("Validation and Test Errors for Each Dropout Rate")
plt.legend()
plt.show()