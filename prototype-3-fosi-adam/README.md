Here we will try to fix the bad metrics (zeros and pure ones 0, 1) by running more examples, changing some code or both.

**Important**
Managed to mitigate to the new API from Pytorch, in order to use functional model.
See [link1](https://pytorch.org/docs/stable/func.migrating.html#function-transforms) and [link2](https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf).
Note to self: In order to work you need to pass arguments as it is in the functional model, after every time is being updated with the new params and buffers !