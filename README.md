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

## Evaluation Results

Based on the evaluation of the specified models using the mentioned metrics, the BART-LARGE-XSUM-SAMSUM model achieved the highest TOPSIS score, indicating superior performance across the evaluation parameters.

## Usage

Provide instructions on how to reproduce the evaluation results. Include details on how to run the evaluation script and analyze the metrics.

```bash
# Example command for running evaluation
python evaluate_models.py --model bart-large-xsum-samsum --input input.txt --reference reference.txt
