Here we are trying to first just try and run the functional API of library functorch.

We managed to do it but we need to fix the accuracy problem we have, the labels are actually not 
managed well, maybe it is a problem with the type of th labels.

**Change 1:**
We will try to change the functorch library to use the .make_functional() method pytorch is offering 
as alternative

**Change 2:**
If we will not go to the .make_functional() method, we can try the .torchopt.FuncOptimizer() method that does 
this:

The code: [code](https://github.com/metaopt/torchopt/blob/62c53478aee325f37c3ed4f30d1091777dcf0260/tutorials/1_Functional_Optimizer.ipynb)

``` python
def interact_with_functorch_with_wrapper():
    batch_size = 1
    dim = 1
    net = Net(dim)
    model, params = functorch.make_functional(net)  # get the functional version of the model

    learning_rate = 1.0
    optimizer = torchopt.FuncOptimizer(torchopt.adam(learning_rate))

    xs = 2 * torch.ones((batch_size, dim))
    ys = torch.ones((batch_size, 1))

    pred = model(params, xs)
    loss = mse(pred, ys)

    print('Parameters before update:', params)
    params = optimizer.step(loss, params)
    print('Parameters after update:', params)
```
