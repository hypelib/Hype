
#r "../../src/Hype/bin/Release/DiffSharp.dll"
#r "../../src/Hype/bin/Release/Hype.dll"
#I "../../packages/RProvider.1.1.14"
#load "RProvider.fsx"

open Hype
open Hype.Neural
open DiffSharp.AD.Float32
open DiffSharp.Util

open RDotNet
open RProvider
open RProvider.graphics




let dataor = Dataset(toDM [[0.; 0.; 1.; 1.]
                           [0.; 1.; 0.; 1.]],
                     toDM [[0.; 1.; 1.; 1.]])



let n = FeedForward()
n.Add(Linear(2, 4))
n.Add(tanh)
n.Add(Linear(4, 1))
n.Add(Activation(sigmoid))

n.ToStringFull() |> printfn "%s"

Layer.Train(n, dataor, {Params.Default with Epochs = 1000})



let train x =
    let n = FeedForward()
    n.Add(Linear(2, 4))
    n.Add(tanh)
    n.Add(Linear(4, 1))
    n.Add(sigmoid)

    Layer.Train(n, dataor, {Params.Default with LearningRate = Schedule x; Silent = true})


let hypertrain x =
    Optimize.Minimize(train, DV.create 100 (D 1.5f), {Params.Default with Epochs = x; LearningRate = Constant (D 0.001f)})

let w, _ = hypertrain 1000


namedParams [
    "x", box (w |> DV.toArray |> Array.map (float32>>float))
    "pch", box 16
    "col", box "blue"
    "ylim", box [0; 2]]
|> R.plot |> ignore