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

### 1. BERT

![image](https://github.com/user-attachments/assets/2e81ea01-f22a-4c48-a99b-4f763424e096)
![97d28f12c64ac7b0fb971e2fc0f4d21](https://github.com/user-attachments/assets/655c6051-6dfe-44ba-80ed-5f8ff382e9b4)
![b5b071a31b8f5ada45552e08907b4ad](https://github.com/user-attachments/assets/9e560120-1906-48f2-8ffd-4d41445ed51a)
![27babbb4ffa20d81fc1a37d1ab7b05e](https://github.com/user-attachments/assets/02502d3c-60d9-40d1-8bae-cb0008539832)

### Hyper-parameters

Learning rate: We start as a bigger value with alpha 0.01 and find out it is too big that make our loss and accuracy bounce frequently in the graph. Therefore, we slowly descrease its value to 1e-7 and find out there is always a huge gap arround the 4000th step. Thereofre, we set 1e-5 in our final result to balance the overfitting.
Epoches: For the number of epoches to train the model, we tried from 2 to 5. Since we are able to get a fine result, we pick 3 to balance the overfitting.
Weight Decay: For the weight decay, we start with 0.1 to tune and decrease it slowly. It turns out the weights decay does not influence our model much. Therefore, we choose the 0.01 which has the best performance to avoid overfit.

#### 2. LSTM

<img width="876" alt="Screenshot 2024-12-02 at 6 27 03 PM" src="https://github.com/user-attachments/assets/dbd724df-fbf3-4f5a-88a5-689459c4049a">

![image](https://github.com/user-attachments/assets/146ab15d-ea21-499f-9fbc-fabdaf19fbe6)
![image](https://github.com/user-attachments/assets/f9e08229-8824-4749-a994-c9ba071be25b)
![image](https://github.com/user-attachments/assets/0d7b5d5c-5c00-4937-9557-e7cd5d3ceb70)

#### Hyper-parameters

For the LSTM model, we have a few parameters:

1. `embedding_dim`: We set our initial dimensionality of the word embedding to 50 and gradually increased it to 150 (step of 50). We found that the higher the dimension, the more semantic nuances of the words can be captured. To reduce overfitting, we decided to go with 100.
2. `hidden_dim`: We set this value to 256. We tried increasing this value, but it's too computational expensive for our computers to run with no apparent benefits.
3. `output_dim`: We set the output dimention to 2 since we are trying to differentiate between human and AI.
4. `n_layers`: We set this value to 3 intially, but since the model is more than capable of reaching a high degree of accuracy with our dataset even with a small number of layers, we ended up setting it to 2 to reduce computational runtime.
5. `num_epoches`: We set this to the same value as the BERT model for better comparison.
6. `learning_rate`: Following the BERT model, we found that increasing this value will overshoot the optimal solution as it leads to poor convergence in training. Decreasing the value will also drastically increase the training runtime. We ended up with 1e-4.

## Result:
