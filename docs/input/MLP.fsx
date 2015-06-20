#r "../../packages/FsAlg.0.5.12/lib/FsAlg.dll"
#r "../../packages/DiffSharp.0.6.2/lib/DiffSharp.dll"
#I "../../packages/RProvider.1.1.8"
#load "RProvider.fsx"

#load "../../src/Hype/Hype.fs"
#load "../../src/Hype/Optimize.fs"
#load "../../src/Hype/Neural.fs"
#load "../../src/Hype/Neural.MLP.fs"

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
    let par = {DefaultParams with LearningRate = Scheduled x; Batch = Full; Verbose = false}
    let net = MLP.create([|2; 1|], Activation.sigmoid, D -1.41, D 1.41)
    net.Train par dataOR
    Loss.Quadratic dataOR net.Run

let hypertrain = 
    let report i (w:Vector<_>) _ =
        namedParams [   
            "x", box (w |> Vector.map float |> Vector.toArray);
            "type", box "o"; 
            //"ylim", box [0.5; 2.];
            "col", box "blue"]
        |> R.plot |> ignore
    Optimize.GD {DefaultParams with Epochs = 250; ReportFunction = report; ReportInterval = 10} train (Vector.create 50 (D 1.))
