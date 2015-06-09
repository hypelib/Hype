#r "../packages/FsAlg.0.5.9/lib/FsAlg.dll"
#r "../packages/DiffSharp.0.6.2/lib/DiffSharp.dll"
#I "../packages/RProvider.1.1.8"
#load "RProvider.fsx"

#load "../src/Hype/Hype.fs"
#load "../src/Hype/Optimize.fs"
#load "../src/Hype/Neural.fs"
#load "../src/Hype/Neural.MLP.fs"

open RDotNet
open RProvider
open RProvider.graphics

open FsAlg.Generic
open DiffSharp.AD
open Hype
open Hype.Neural

let OR = LabeledSet.create [[|D 0.; D 0.|], [|D 0.|]
                            [|D 0.; D 1.|], [|D 1.|]
                            [|D 1.; D 0.|], [|D 1.|]
                            [|D 1.; D 1.|], [|D 1.|]]

let XOR = LabeledSet.create [[|D 0.; D 0.|], [|D 0.|]
                             [|D 0.; D 1.|], [|D 1.|]
                             [|D 1.; D 0.|], [|D 1.|]
                             [|D 1.; D 1.|], [|D 0.|]]

let net = MLP.create([|2; 1|])

let train (x:Vector<_>) =
    let report _ w _ =
        namedParams [   
            "x", box (w |> Vector.map float |> Vector.toArray);
            "type", box "o"; 
            "col", box "blue";
            "ylim", box [0; 4]]
        |> R.plot |> ignore

    let par = {Params.Default with LearningRate = ScheduledLearningRate x; TrainFunction = Train.GD; GDReportFunction = report}
    let net2 = MLP.create([|2; 1|], tanh, D -0.5, D 0.5)
    let op = net2.Train par OR
    op |> snd

let test2 = 
    Optimize.GD {Params.Default with Epochs = 100} train (Vector.create 15 (D 0.5))

