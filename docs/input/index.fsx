(*** hide ***)
#r "../../src/Hype/bin/Debug/DiffSharp.dll"
#r "../../src/Hype/bin/Debug/FsAlg.dll"
#r "../../src/Hype/bin/Debug/Hype.dll"


(**
Hype: Compositional Machine Learning and Hyperparameter Optimization
====================================================================

Hype is a [compositional](http://mathworld.wolfram.com/Composition.html) machine learning library ...

This is enabled by nested automatic differentiation ... DiffSharp ...

You don't need to worry about ... the gradient of your models ... computed exactly by automatic differentiation.

$$$
   f(x) = x + 3
*)

open Hype
open Hype.Neural

// Train a network with stochastic gradient descent and a learning rate schedule
let train (x:Vector<_>) = 
    let par = {Params.Default with LearningRate = Scheduled x; TrainFunction = Train.MSGD}
    let net = MLP.create([|28; 15; 1|], tanh, D -1.41, D 1.41)
    net.Train par data
    Loss.Quadratic(data, net.Run)

// Train the training, i.e., optimize the learning schedule by using its hypergradient
let hypertrain = 
    Optimize.GD {Params.Default with Epochs = 50} train (Vector.create 200 (D 1.0))

(**
Roadmap
-------

<div class="row">
<div class="col-sm-5">
<div class="alert alert-info">
  <strong>Supported in the current relase</strong> 

* Linear regression
* Logistic regression
* Feedforward neural networks 
</div>

<div class="alert alert-info">
  <strong>Working on for future releases</strong> 

* GPU support through CUDA
* Recurrent neural networks
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

