import pandas as pd
from rake_nltk import Rake
import nltk
from transformers import BertTokenizer
import numpy as np

nltk.download("stopwords")
nltk.download("punkt")

def keywords(text):

    r = Rake()
    r.extract_keywords_from_text(text)
    ranks = r.get_ranked_phrases_with_scores()  

    return " ".join([rank[1] for rank in ranks if rank[0] >= 4.0])

def bert_tokenizing(text):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(text)
    
    return tokenizer.convert_tokens_to_ids(tokens)



df = pd.read_csv("Nexus Labs Registration.csv")
df = df.drop(columns=["Name", "UCL email", "What experience do you have in AI? Select all that apply.", "What experience do you have in Python? Do you know any other programming languages?", "Is there anything else you would like to add or point out?"])
df = df.rename(columns={
    "What is your course?": "course",
    "Which pillar interests you most?": "pillar",
    "Are you interested in any other of the research pillars? If so, which ones?": "other pillars",
    "Why do you want to join Nexus Labs?": "motivation",
    "Which pillar interests you most? Why?": "why pillar",
    "What do you hope to get out of this experience?": "positive gain",
    "How do you think you can best contribute to Nexus Labs and to your team?": "contribution",
})

one_hot_encoding = {
    "Neuroscience": [1,0,0,0,0],
    "Responsible AI": [0,1,0,0,0],
    "Machine Vision and Robotics": [0,0,1,0,0],
    "Sustainability": [0,0,0,1,0],
    "Natural Language Processing": [0,0,0,0,1]
}

print(df.columns)

df["pillar"] = df["pillar"].apply(lambda x: one_hot_encoding[x])
df["motivation"] = df["motivation"].apply(lambda x: bert_tokenizing(keywords(x)))
df["why pillar"] = df["why pillar"].apply(lambda x: bert_tokenizing(keywords(x)))
df["positive gain"] = df["positive gain"].apply(lambda x: bert_tokenizing(keywords(x)))
df["contribution"] = df["contribution"].apply(lambda x: bert_tokenizing(keywords(x)))
df["other pillars"] = df["other pillars"].apply(lambda x: bert_tokenizing(str(x)))
df["course"] = df["course"].apply(lambda x: bert_tokenizing(str(x)))

# data = df.to_numpy()

# with open("tokenized_data.npy", "wb") as file:
#     np.save(file, data)

df.to_csv("tokenized_data.csv")