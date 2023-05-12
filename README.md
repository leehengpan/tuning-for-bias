# Tuning for Bias: An Analysis of Political Bias Through Title Generation

## Overview
In this project, we study how a language model trained on two divergent media houses’ data–conditioned on each organization’s political view–can exacerbate bias. We developed a transformer-based model that predicts an article’s title, conditioned on a liberal vs. conservative viewpoint. Bearing that title generation is a form of text summarization, we hope to highlight that sensationalized news headlines of media houses split across political views can present different interpretations of the same content/information.


## Architecture

<img src="https://github.com/raichandanisagar/text-to-title/blob/main/report/bert-image-example.jpeg" width="300"/>

## How to run

We've tried to make it simple to run the code and reproduce our results. Once the repository has been cloned and environment duplicated, fire up `main.ipynb` in the src directory. It handles the necessary imports and starts executing the code from the first steps related to data preparation and preprocessing, to model training and evaluation. Bear that certain sections are optional and have been marked as such depending upon the exact use case. These involve choosing to save model weights, loading custom model weights, or training the model from scratch.

## Results
| Masked Accuracy | Masked Loss    | Mean BLEU      |
|-----------------|----------------|----------------|
| [0.49, 0.57]    | [1.736, 2.316] | [0.287, 0.350] |
