import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

# Load pre-trained models
models = {
    "cnicu/t5-small-booksum": pipeline("summarization", model="cnicu/t5-small-booksum"),
    "sshleifer/distilbart-cnn-6-6": pipeline("summarization", model="sshleifer/distilbart-cnn-6-6"),
    "Falconsai/medical_summarization": pipeline("summarization", model="Falconsai/medical_summarization"),
    "Cohee/bart-factbook-summarization": pipeline("summarization", model="Cohee/bart-factbook-summarization"),
    "lidiya/bart-large-xsum-samsum": pipeline("summarization", model="lidiya/bart-large-xsum-samsum"),
}

# Sample input text
sample_text = "Your sample text goes here."


# Function to calculate ROUGE scores
def calculate_rouge(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]


# Function to calculate BLEU score
# def calculate_bleu(reference, hypothesis):
#     reference = [reference.split()]
#     hypothesis = hypothesis.split()
#     return corpus_bleu(reference, [hypothesis])


def content_overlap_evaluation(reference, hypothesis):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([reference, hypothesis])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return cosine_sim


# Create a DataFrame to store the evaluation results
columns = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "Content_Overlap"]
results_df = pd.DataFrame(columns=columns)

# Evaluate each model
for model_name, model in models.items():
    summary = \
        model(sample_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)[0][
            'summary_text']

    # Define reference summary
    reference_summary = "Your reference summary goes here."

    # Calculate evaluation metrics
    rouge_scores = calculate_rouge(reference_summary, summary)
    print(rouge_scores)
    coe_value = content_overlap_evaluation(reference_summary, summary)

    # Append results to the DataFrame
    results_df.loc[model_name] = [rouge_scores["rouge-1"]["f"], rouge_scores["rouge-2"]["f"],
                                  rouge_scores["rouge-l"]["f"], coe_value]

# Print the results DataFrame
print(results_df)
df = pd.DataFrame(results_df)
df.to_csv('result.csv', index=False)
