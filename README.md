# FOSI - Further Work
Here we will develop an interactive framework (via terminal) in order to train, validate and test LLM models via the dataset of User's choice, using the FOSI optimizer introduced at [source](https://github.com/hsivan/fosi)

# The work
This work is done in order to test more cases with the FOSI optimizer, in more datasets and more models. The terminal interaction is offered as an easy way to get results fast and with minimum code change.

# The files
There are many files covering the different combination of FOSI that can be run. There are also files that are not using FOSI at all, this was developed to compare results later , using the same pre processing for equal and robust results.

# Future work
This framework is in beta version 6, that is requiring more attention and more defensive code, in order to prevent problems like "not compatibility of this specific task with this model, e.t.c." This is in development, but use cases with distil-bert and classification datasets like mprc, sst2 are working.