# AI_Response_Classifier
Team Member: 
Hancheng Huang, Yiyang He, Guanghao Huang
Introduction:
This project compare the LSTM model and BERT model on classify if a paragraph is generate by AI or human being.
Our proposed solution includes training two types of models: a transformer-based model BERT and a non-transformer-based model LSTM. We will fine-tune each model to classify whether the response is generated from human or AI. By comparing their performance, we will determine which model is more accurate for the process of distinguishing. We use dateset sourced from Kaggle and contains 480000 text response. These texts are labled as either human or AI-generated. The dataset includes a feature labeled “# generated” that indicates whether the response is from human(0) or AI(1). We decide to pick a subset of the dataset for model training and evaluation.
Problem Statement:
As AI-generated content becomes more common and many AI-generated contents are misused in many places. What’s more, the lack of tools to effectively distinguish response between human and AI-generated causes a risk to integrity, that makes the situation worse. Also, It effects many areas, such as education and customer service. Our goal is trying to train models to solve these issues with high accuracy.
Proposed Solution: Our proposed solution includes training two types of models: a transformer-based model BERT and a non-transformer-based model LSTM. We will fine-tune each model to classify
whether the response is generated from human or AI. By comparing their performance, we will determine which model is more accurate for the process of distinguishing.
Solution Approach:
Preprocessing:
As the other researchers do in the paper review (Campino 2024[1]) we are considering trying their treatments on our dataset.
(i) removal of special characters and numbers: remove all the special characters in the text such as commas or parentheses. Numbers were also removed from the text. The removal of such characters did not show meaningful differences in the final result as the model is focused on the tokens (i.e.: words) and not on punctuati on, for example
(ii) convert to lower case: convert the text to lower case to make it completely uniform
(iii) combine the text into a single line: in case an abstract has several paragraphs, it was assured that all the text fits to a single line
(iv) remove extra spaces: delete any extra spaces between words resulti ng from previous data treatment, leaving a single space between words. In additon, we are going to try if Stop Words and reduce each word to its base form can help our
models predict better.
Model Architecture:
In this project, we aim to compare different models on our dataset to determine which performs best. We will primarily focus on comparing transformer-based models, such as BERT, with non-transformer models like LSTM. Both are well-suited for sequenti al data processing but employ distinct mechanisms for capturing long-range dependencies: transformers rely on attention mechanisms, while LSTMs use gated recurrent units to manage informati on flow over time. This makes their performance comparison insightf ul, as each approach has its strengths and limitations in handling sequence data. LSTM is considered the best choice for benchmarking against transformer models, providing a solid foundation for evaluating the advantages of newer architectures.
Hardware Requirements for Training:
As a student, we plan to use our only computer with a RTX3070ti GPU to do the fine-tuning. Since the fine-tuning process does not require much computi ng resource, it should be able to complete the task.
