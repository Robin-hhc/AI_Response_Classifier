# AI_Response_Classifier
## Team Member: 
Hancheng Huang, Yiyang He, Guanghao Huang

## Introduction:
### This project compare the LSTM model and BERT model on classify if a paragraph is generate by AI or human being.
Our proposed solution includes training two types of models: a transformer-based model BERT and a non-transformer-based model LSTM. We will fine-tune each model to classify whether the response is generated from human or AI. By comparing their performance, we will determine which model is more accurate for the process of distinguishing.
## Problem Statement:
As AI-generated content becomes more common and many AI-generated contents are misused in many places. What’s more, the lack of tools to effectively distinguish response between human and AI-generated causes a risk to integrity, that makes the situation worse. Also, It effects many areas, such as education and customer service. Our goal is trying to train models to solve these issues with high accuracy.
Proposed Solution: Our proposed solution includes training two types of models: a transformer-based model BERT and a non-transformer-based model LSTM. We will fine-tune each model to classify
whether the response is generated from human or AI. By comparing their performance, we will determine which model is more accurate for the process of distinguishing.

## Data:
### Data Source:
We will collect a dataset from the website(AI Vs Human Text (kaggle.com)). The dataset contains 480000 unique text responses, with each entry classified as human-generated and AI-generated. The dataset includes a feature labeled “# generated” that indicates whether the response is from human(0) or AI(1). The text block contains complete sentences, represented as string data type, which are generated by humans or AI. <br>
![image](https://github.com/user-attachments/assets/df5bf85d-de60-41ae-aae9-e5d1febd83bf)
### Data Split:
Since fine-tuning doe not require a huge data set for getting an accurate result, We pick 50000 data points for training and evaluation. From the 50000 data points, 20% of them are set to be out test set and the other 80% are set to be our train set.
### Data Preprocessing:
(i) removal of special characters and numbers: remove all the special characters in the text such as commas or parentheses. Numbers were also removed from the text. The removal of such characters did not show meaningful differences in the final result as the model is focused on the tokens (i.e.: words) and not on punctuati on, for example<br>
(ii) convert to lower case: convert the text to lower case to make it completely uniform<br>
(iii) combine the text into a single line: in case an abstract has several paragraphs, it was assured that all the text fits to a single line<br>
(iv) remove extra spaces: delete any extra spaces between words resulti ng from previous data treatment, leaving a single space between words. In additon, we are going to try if Stop Words and reduce each word to its base form can help our
models predict better.<br>

## Model Training:
### BERT diagrams:
![image](https://github.com/user-attachments/assets/2e81ea01-f22a-4c48-a99b-4f763424e096)
![97d28f12c64ac7b0fb971e2fc0f4d21](https://github.com/user-attachments/assets/655c6051-6dfe-44ba-80ed-5f8ff382e9b4)
![b5b071a31b8f5ada45552e08907b4ad](https://github.com/user-attachments/assets/9e560120-1906-48f2-8ffd-4d41445ed51a)
![27babbb4ffa20d81fc1a37d1ab7b05e](https://github.com/user-attachments/assets/02502d3c-60d9-40d1-8bae-cb0008539832)
### Hyper-parameters for BERT
Learning rate: We start as a bigger value with alpha 0.01 and find out it is too big that make our loss and accuracy bounce frequently in the graph. Therefore, we slowly descrease its value to 1e-7 and find out there is always a huge gap arround the 4000th step. Thereofre, we set 1e-5 in our final result to balance the overfitting.
Epoches: For the number of epoches to train the model, we tried from 2 to 5. Since we are able to get a fine result, we pick 3 to balance the overfitting.
Weight Decay: For the weight decay, we start with 0.1 to tune and decrease it slowly. It turns out the weights decay does not influence our model much. Therefore, we choose the 0.01 which has the best performance to avoid overfit.

## Result:
