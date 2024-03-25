

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from transformers import pipeline

# Initialize the text generation pipeline with the desired model
generator = pipeline('text-generation', model='gpt2')
generated_text1 = generator("deep learning", max_length=100)
generated_text2 = generator("natural language processing", max_length=100)
generated_text3 = generator("computer vision", max_length=100)
generated_text4= generator("history of egypt", max_length=100)

documents=[generated_text1,generated_text2,generated_text3,generated_text4]

print(documents)



# Function to preprocess text
def preprocess_text(text):
        ## cleaning the data
    cleaned_data= re.sub(r'[^\w\s]','',text)

    ## Normalization
    normatized_text= cleaned_data.lower()

    ## Tokenization 
    tokenized_text= normatized_text.split()

    ## Lemmatization
    lemm= WordNetLemmatizer()
    lemmatized_text = [lemm.lemmatize(word) for word in tokenized_text]

    ## Unique Words
    stop_words = set(stopwords.words('english'))
    unique_words= [word for word in lemmatized_text if word not in stop_words]
    
    return ' '.join(unique_words)

# Preprocess each document
preprocessed_documents =[]

for doc in documents:
    temp=doc[0]['generated_text']
    processed_doc=preprocess_text(temp)
    preprocessed_documents.append(processed_doc)

# Function to calculate TF for a document
def calculate_tf(document):

    words = document.split()
    word_count = Counter(words)
    tf = {word: count/len(words) for word, count in word_count.items()}
    return tf

# Calculate TF for all preprocessed documents
tf_documents = [calculate_tf(doc) for doc in preprocessed_documents]

# Function to calculate IDF for a word across all documents
def calculate_idf(word, all_documents):
    num_documents_with_word = sum([1 for doc in all_documents if word in doc])
    idf = np.log(len(all_documents+1) / (1 + num_documents_with_word))+1
    return idf

# Get unique words across all preprocessed documents
all_words = set([word for doc in preprocessed_documents for word in doc.split()])
# Calculate IDF for each word
idf_values = {word: calculate_idf(word, preprocessed_documents) for word in all_words}

# Function to calculate TF-IDF for a document
def calculate_tfidf(tf, idf):
    tfidf = {word: tf[word] * idf[word] for word in tf.keys()}
    return tfidf

# # Calculate TF-IDF for all preprocessed documents
tfidf_documents = [calculate_tfidf(tf, idf_values) for tf in tf_documents]

print(tfidf_documents)
# Normalize TF-IDF values
def normalize_tfidf(tfidf):
    total_tfidf = sum(tfidf.values())
    normalized_tfidf = {word: value / total_tfidf for word, value in tfidf.items()}
    return normalized_tfidf

normalized_tfidf_documents = [normalize_tfidf(tfidf) for tfidf in tfidf_documents]

# Print normalized TF-IDF for each document
for i, doc_tfidf in enumerate(normalized_tfidf_documents):
    print(f"Document {i+1}:")
    for word, tfidf in doc_tfidf.items():
        print(f"{word}: {tfidf}")
    print()

