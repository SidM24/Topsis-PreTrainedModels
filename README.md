# Text Summarization Models Evaluation

This repository presents an evaluation of various text summarization models using popular metrics such as ROUGE-1, ROUGE-2, ROUGE-L scores, and Content Overlap Evaluation. The models included in this evaluation are:

- [t5-Small-Booksum](https://huggingface.co/cnicu/t5-small-booksum)
- [distilbart](https://huggingface.co/sshleifer/distilbart-cnn-6-6)
- [medial-summarization](https://huggingface.co/Falconsai/medical_summarization)
- [bart-facebook](https://huggingface.co/Cohee/bart-factbook-summarization)
- [bart-large-sum](https://huggingface.co/lidiya/bart-large-xsum-samsum)

## Evaluation Parameters

### ROUGE-1 Score

ROUGE-1 measures the overlap of unigrams (single words) between the generated summary and reference summaries.

### ROUGE-2 Score

ROUGE-2 measures the overlap of bigrams (pairs of consecutive words) between the generated summary and reference summaries.

### ROUGE-L Score

ROUGE-L measures the longest common subsequence (LCS) of words between the generated summary and reference summaries.

### Content Overlap Evaluation

Content Overlap Evaluation is a custom metric designed to evaluate the overlap of content between the generated summary and reference summaries.

## Evaluation Parameter Results

The result.csv file stores the calculated evaluation metrics for every model as follows

| ROUGE-1     | ROUGE-2     | ROUGE-L     | Content_Overlap |
|-------------|-------------|-------------|-----------------|
| 0.16216216  | 0.043478259 | 0.16216216  | 0.126621508     |
| 0.117647056 | 0           | 0.117647056 | 0.111593981     |
| 0.166666664 | 0.037037036 | 0.166666664 | 0.081711684     |
| 0.055555553 | 0           | 0.055555553 | 0.059981266     |
| 0.206896549 | 0.060606058 | 0.206896549 | 0.1645521       |

## Evaluation Results

The output_with_topsis.csv file stores the final result obtained after application of the TOPSIS for selection of the best model.

| Model            | ROUGE-1     | ROUGE-2     | ROUGE-L     | Content_Overlap | Topsis Score | Final Rank |
|------------------|-------------|-------------|-------------|-----------------|--------------|------------|
| t5-small-booksum | 0.16216216  | 0.043478259 | 0.16216216  | 0.126621508     | 0.699806524  | 2          |
| distlbart-cnn    | 0.117647056 | 0           | 0.117647056 | 0.111593981     | 0.28046912   | 4          |
| Falconsai        | 0.166666664 | 0.037037036 | 0.166666664 | 0.081711684     | 0.584748487  | 3          |
| bart-factbook    | 0.055555553 | 0           | 0.055555553 | 0.059981266     | 0            | 5          |
| bart-large-xsum  | 0.206896549 | 0.060606058 | 0.206896549 | 0.1645521       | 1            | 1          |

## Best Model

Based on the evaluation of the specified models using the mentioned metrics, the BART-LARGE-XSUM-SAMSUM model achieved the highest TOPSIS score, indicating superior performance across the evaluation parameters.


## Usage

The TopsisofPreTrainedModels.py file contains the code for calculation of the evaluation parameters for each of the pre-trained text-summarization models used above
The main.py file stores the code to calculate the best Model based on the results



