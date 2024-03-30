We developed the train, test and validation check in the Trainer class, not using the Evaluation Class.

We noticed that the num of labels = 1 is working only, because we made the model use sigmoid as the predictions. We need to adapt to softmax predictions if the 
model is having more than 1 true labels.