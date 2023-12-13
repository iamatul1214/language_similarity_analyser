from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd



def cosine_similarity_calculator(text_1,text_2):
    full_text = text_1 + text_2

    #tf-idf vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(full_text)

        # Debugging print statements
    print(f"Sentences: {full_text}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

     # Calculate cosine similarity between language 1 and language 2 sentences
    num_lang1_sentences = len(text_1)
    num_lang2_sentences = len(text_2)

    tfidf_lang1 = tfidf_matrix[:num_lang1_sentences]  # TF-IDF matrix for language 1
    tfidf_lang2 = tfidf_matrix[num_lang1_sentences:]  # TF-IDF matrix for language 2

    cosine_similarities = cosine_similarity(tfidf_lang1, tfidf_lang2)


    return round(cosine_similarities.flatten()[0]*100,2)

def jaccard_similarity(text_1, text_2):
    # Tokenize the sentences into sets of words

    text_1 = ' '.join(text_1)
    text_2 = ' '.join(text_2)
    set1 = set(text_1.split())
    set2 = set(text_2.split())
    
    # Compute Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    jaccard_score = intersection / union if union > 0 else 0  # Handling division by zero
    
    return round(jaccard_score * 100,2)

def dice_coefficient(text_1, text_2):
    # Tokenize the sentences into sets of words
    text_1 = ' '.join(text_1)
    text_2 = ' '.join(text_2)
    set1 = set(text_1.split())
    set2 = set(text_2.split())
    
    # Compute Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    dice_score = (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0  # Handling division by zero
    
    return round(dice_score*100,2)

def calculate_similarity(text_1,text_2):

    cosine = cosine_similarity_calculator(text_1=text_1,text_2=text_2) 
    jaccard = jaccard_similarity(text_1=text_1,text_2=text_2)
    dice = dice_coefficient(text_1=text_1,text_2=text_2)

    cosine = f"{cosine}%"
    jaccard = f"{jaccard}%"
    dice = f"{dice}%"
    
    similarity_df = pd.DataFrame({"cosine_similarity":[cosine],"jaccard_coefficeint":[jaccard],
                                  "dice_coefficeint":[dice]})
    # similarity_df = [cosine,jaccard,dice]
    return similarity_df

