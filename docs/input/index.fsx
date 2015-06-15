(*** hide ***)
#r "../../src/Hype/bin/Debug/DiffSharp.dll"
#r "../../src/Hype/bin/Debug/FsAlg.dll"
#r "../../src/Hype/bin/Debug/Hype.dll"


(**
<div style="font-size: 65px">Hype</div>

Machine Learning and Hyperparameter Optimization
==============================================================

The DiffSharp library provides several non-nested implementations of forward and reverse AD, for situations where it is known beforehand that nesting will not be needed. This can give better performance for some specific non-nested tasks.

The non-nested AD modules are provided under the DiffSharp.AD.Specialized namespace.

For a complete list of the available differentiation operations, please refer to API Overview and API Reference.

$$$
   f(x) = x + 3
*)

open DiffSharp.AD
open FsAlg.Generic
open Hype
open Hype.Neural

let dataOR = {X = matrix [[D 0.; D 0.; D 1.; D 1.]
                          [D 0.; D 1.; D 0.; D 1.]]
              Y = matrix [[D 0.; D 1.; D 1.; D 1.]]}

let dataXOR = {X = matrix [[D 0.; D 0.; D 1.; D 1.]
                           [D 0.; D 1.; D 0.; D 1.]]
               Y = matrix [[D 0.; D 1.; D 1.; D 0.]]}


let train (x:Vector<_>) =
    let par = {Params.Default with LearningRate = Scheduled x; TrainFunction = Train.GD}
    let net = MLP.create([|2; 1|], Activation.sigmoid, D -0.5, D 0.5)
    net.Train par dataOR
    Loss.Quadratic(dataOR, net.Run)

let hypertrain = 
    Optimize.GD {Params.Default with Epochs = 50} train (Vector.create 10 (D 1.0))
