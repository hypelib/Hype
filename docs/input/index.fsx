(*** hide ***)
#r "../../src/Hype/bin/Debug/DiffSharp.dll"
#r "../../src/Hype/bin/Debug/Hype.dll"
open DiffSharp.AD.Float32

(**
Hype: Compositional Machine Learning and Hyperparameter Optimization
====================================================================

Hype is a proof-of-concept, experimental library for [compositional](http://mathworld.wolfram.com/Composition.html) machine learning, where you can perform optimization on systems of many components, even when such components themselves internally perform optimization. 

This is enabled by the nested automatic differentiation (AD) capability provided by a special numeric type __D__, which is used as the standard floating-point type in Hype,  giving you access to the exact derivative of any value in your model with respect to any other. 

### Automatic derivatives

You do not need to worry about supplying gradients (or Hessians) of your models, which are computed exactly and efficiently by AD. The underlying AD functionality is provided by [DiffSharp](http://diffsharp.github.io/DiffSharp/index.html). 

AD is a generalized form of "backpropagation" and is distinct from numerical or symbolic differentiation.

### Hypergradients

You can get exact gradients of the training or validation loss with respect to hyperparameters. These __hypergradients__ allow you to do gradient-based optimization of gradient-based optimization, meaning that you can do things like optimizing learning rate and momentum schedules, weight initialization parameters, or step sizes and mass matrices in Hamiltonian Monte Carlo models.

*)

open Hype
open Hype.Neural

// Train a network with stochastic gradient descent and a learning rate schedule
let train (x:DV) = 
    let n = FeedForward()
    n.Add(Linear(784, 300))
    n.Add(Activation(tanh))
    n.Add(Linear(300, 10))
    let loss = Layer.Train(n, data, {Params.Default with 
                                        LearningRate = Schedule x
                                        Momentum = Momentum.DefaultNesterov
                                        Batch = Minibatch 100
                                        Loss = CrossEntropyOnLinear})
    loss // Return the loss at the end of training

// Train the training, i.e., optimize the learning schedule vector by using its hypergradient
let hypertrain = 
    Optimize.Minimize(train, DV.create 200 (D 1.f), {Params.Default with Epochs = 50})

(**

You can also take derivatives with respect to your training data, to analyze training sensitivities.

### Compositionality

Nested AD handles higher-order derivatives up to any level, including in complex cases such as 

$$$
    \mathbf{min} \left(x \; \mapsto \; f(x) + \mathbf{min} \left( y \; \mapsto \; g(x,\,y) \right) \right)\, ,

where $\mathbf{min}$ uses gradient-based optimization. (Note that the inner function has a reference to the argument of the outer function.) This allows you to create complex systems where many components may internally perform optimization.

For example, you can optimize the rules of a multi-player game where the players themselves optimize their own strategy using a simple model of the opponent which they optimize according to their opponent's observed behaviour. 

Or you can perform optimization of procedures that are internally using differentiation for purposes other than optimization, such as adaptive control.

### Complex objective functions

You can use derivatives in defining objective functions for training your models. For example, your objective function can take input sensitivities into account, for training neural networks that are invariant to a set of chosen input transformations.

Roadmap
-------

<div class="row">
<div class="col-sm-6">
<div class="alert alert-info">
  <strong>In the current release</strong> 

* OpenBLAS backend by default
* Regression, feedforward neural networks
* Recurrent neural networks, LSTM
* Hamiltonian Monte Carlo
</div>
</div>

<div class="col-sm-6">
<div class="alert alert-info">
  <strong>Upcoming features</strong> 

* GPU support through CUDA
* Probabilistic inference
* Convolutional neural networks
</div>
</div>
</div>

About
-----

Hype is developed by [Atılım Güneş Baydin](http://www.cs.nuim.ie/~gunes/) and [Barak A. Pearlmutter](http://bcl.hamilton.ie/~barak/) at the [Brain and Computation Lab](http://www.bcl.hamilton.ie/), Hamilton Institute, National University of Ireland Maynooth.

License
-------

Hype is released under the MIT license.
*)

