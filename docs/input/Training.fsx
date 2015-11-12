(*** hide ***)
#r "../../packages/Google.DataTable.Net.Wrapper.3.1.2.0/lib/Google.DataTable.Net.Wrapper.dll"
#r "../../packages/Newtonsoft.Json.7.0.1/lib/net45/Newtonsoft.Json.dll"
#r "../../packages/XPlot.GoogleCharts.1.2.1/lib/net45/XPlot.GoogleCharts.dll"
#r "../../packages/XPlot.GoogleCharts.WPF.1.2.1/lib/net45/XPlot.GoogleCharts.WPF.dll"
#r "../../src/Hype/bin/Release/DiffSharp.dll"
#r "../../src/Hype/bin/Release/Hype.dll"
fsi.ShowDeclarationValues <- false
System.Environment.CurrentDirectory <- __SOURCE_DIRECTORY__

(**
Training
========

In [optimization,](optimization.html) we've seen how nested AD and gradient-based optimization work together.

Training a model is the optimization of model parameters to minimize a loss function, or equivalently, to maximize the likelihood of a given set of data under the model parameters. In addition to the learn ...

For supervised training, data constits of input and output pairs. We represent data using the **Dataset** type.

Dataset
-------
*)

open Hype
open DiffSharp.AD.Float32

let x = toDM [[0; 0; 1; 1]
              [0; 1; 0; 1]]
let y = toDM [[0; 1; 1; 0]]

let xor = Dataset(x, y)

(**
See API reference for the various ways of constructing Datasets.

Loading from CSV file.

Housing data.

*)

let h = Util.LoadDelimited("housing.data") |> DM.Transpose
let hx = h.[0..12, *] |> DM.appendRow (DV.create h.Cols 1.f)
let hy = h.[13..13, *]

let housing = Dataset(hx, hy)

(**
Training parameters
-------------------

### Loss function
*)

type Loss =
    | L1Loss    // L1 norm, least absolute deviations
    | L2Loss    // L2 norm
    | Quadratic // L2 norm squared, least squares
    | CrossEntropyOnLinear  // Cross entropy after linear layer
    | CrossEntropyOnSoftmax // Cross entropy after softmax layer

(**

### Batch
*)

type Batch =
    | Full
    | Minibatch of int 
    | Stochastic       // Minibatch with size 1

(**
### Validation and early stopping
*)

type EarlyStopping =
    | Early of int * int // Stagnation patience, overfitting patience
    | NoEarly
    static member DefaultEarly = Early (750, 10)
    
(**
Nested optimization of training hyperparameters
-----------------------------------------------
*)

