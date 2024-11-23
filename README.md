# AI_Response_Classifier
Team Member: 
Hancheng Huang, Yiyang He, Guanghao Huang
Problem Statement:
As AI-generated content becomes more common and many AI-generated contents are misused in many places. What’s more, the lack of tools to effectively distinguish response between human and AI-generated causes a risk to integrity, that makes the situation worse. Also, It effects many areas, such as education and customer service. Our goal is trying to train models to solve these issues with high accuracy.
Introduction:
This project compare the LSTM model and BERT model on classify if a paragraph is generate by AI or human being.
Our proposed solution includes training two types of models: a transformer-based model BERT and a non-transformer-based model LSTM. We will fine-tune each model to classify whether the response is generated from human or AI. By comparing their performance, we will determine which model is more accurate for the process of distinguishing. We use dateset sourced from Kaggle and contains 480000 text response. These texts are labled as either human or AI-generated. The dataset includes a feature labeled “# generated” that indicates whether the response is from human(0) or AI(1). We decide to pick a subset of the dataset for model training and evaluation.
