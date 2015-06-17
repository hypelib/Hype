//
// This file is part of
// Hype: Machine Learning and Hyperparameter Optimization Library
//
// Copyright (c) 2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
//
// Hype is released under the MIT license.
// (See accompanying LICENSE file.)
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Brain and Computation Lab
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

(*** hide ***)
#r "../../src/Hype/bin/Debug/DiffSharp.dll"
#r "../../src/Hype/bin/Debug/FsAlg.dll"
#r "../../src/Hype/bin/Debug/Hype.dll"


(**
Hype: Compositional Machine Learning and Hyperparameter Optimization
====================================================================

Hype is a [compositional](http://mathworld.wolfram.com/Composition.html) machine learning library, where you can perform optimization on systems of many components, even when such components may themselves internally perform optimization.

This is enabled by the nested automatic differentiation (AD) capability provided by a special numeric type __D__, which is used as the standard floating point type in the system. Giving you access to the exact derivative of any value in your model with respect to any other, operations can be nested to any level, meaning that you can do things like optimizing

You do not need to worry about supplying the gradient (or Hessian) of your models, which are computed exactly and efficiently by nested AD. The underlying AD functionality is provided by [DiffSharp](http://diffsharp.github.io/DiffSharp/index.html). AD is a generalized form of "backpropagation" and is distinct from numerical or symbolic differentiation.

 ... hypergradients, if you will, ... 

$$$
   f(x) = x + 3
*)

open Hype
open Hype.Neural

// Train a network with stochastic gradient descent and a learning rate schedule
let train (x:Vector<_>) = 
    let net = MLP.create([|28; 15; 1|], tanh, D -1.41, D 1.41)
    net.Train {Params.Default with LearningRate = Scheduled x; TrainFunction = Train.MSGD} data
    Loss.Quadratic(data, net.Run) // Return the error at the end of training

// Train the training, i.e., optimize the learning schedule by using its hypergradient
let hypertrain = 
    Optimize.GD {Params.Default with Epochs = 50} train (Vector.create 200 (D 1.0))

(**
Roadmap
-------

<div class="row">
<div class="col-sm-6">
<div class="alert alert-info">
  <strong>Supported in the current relase</strong> 

* Linear regression
* Logistic regression
* Feedforward neural networks 
</div>
</div>

<div class="col-sm-6">
<div class="alert alert-info">
  <strong>Working on for future releases</strong> 

* GPU support through CUDA
* Improved memory constraints for large-scale problems
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

