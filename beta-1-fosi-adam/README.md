This is the first beta verison : 
Multi core enabled 
Logging additional info in the tensorboard working
Fixed saving results per step (in validation too)
Needs to:
Make more than just FOSI-Adam, use the other FOSI too.

Research Tips:
Find a way to not overfit, make the user input the learning rate or use it to fine tune it?

**Problems Found**:
* The STSB dataset because it is in continuous output and not a classification problem but a regression one, doesn;t work with the rest ofthe code.
* The tcmalloc saves ram in the server even if this specific one does not gets used. TODO: Clear the reserved memory after each experiment or failure.