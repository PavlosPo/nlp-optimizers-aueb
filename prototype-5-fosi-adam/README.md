Here I fixed the problem with the predictions to be wrong, I do not use a softmax in the model, as the CrossEntropyLoss() method it does tht inside, so only the inputs needed (outputs of the pure model).

To calculate the metrics I use a logger class, I implement the logging per step and not per epoch.
This is the first ready-to-view results prototype