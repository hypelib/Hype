#r "../packages/FsAlg.0.5.11/lib/FsAlg.dll"
#r "../packages/DiffSharp.0.6.2/lib/DiffSharp.dll"
#I "../packages/RProvider.1.1.8"
#load "RProvider.fsx"

#load "../src/Hype/Hype.fs"
#load "../src/Hype/Data.fs"
#load "../src/Hype/Optimize.fs"
#load "../src/Hype/Neural.fs"
#load "../src/Hype/Neural.MLP.fs"

open RDotNet
open RProvider
open RProvider.graphics

open FsAlg.Generic
open DiffSharp.AD
open DiffSharp.AD.Vector
open Hype
open Hype.Neural


let dataOR = {X = matrix [[D 0.; D 0.; D 1.; D 1.]
                          [D 0.; D 1.; D 0.; D 1.]]
              Y = matrix [[D 0.; D 1.; D 1.; D 1.]]}

let dataXOR = {X = matrix [[D 0.; D 0.; D 1.; D 1.]
                           [D 0.; D 1.; D 0.; D 1.]]
               Y = matrix [[D 0.; D 1.; D 1.; D 0.]]}


let train (x:Vector<_>) =
    let par = {Params.Default with LearningRate = ScheduledLearningRate x; TrainFunction = Train.GD}
    let net = MLP.create([|2; 1|], Activation.sigmoid, D -0.5, D 0.5)
    net.Train par dataOR
    Loss.Quadratic(dataOR, net.Run)

let hypertrain = 
    let report i w _ =
        if i % 2 = 0 then
            namedParams [   
                "x", box (w |> Vector.map float |> Vector.toArray);
                "type", box "o"; 
                "col", box "blue"]
            |> R.plot |> ignore
    Optimize.GD {Params.Default with Epochs = 500; GDReportFunction = report} train (Vector.create 50 (D 1.0))
