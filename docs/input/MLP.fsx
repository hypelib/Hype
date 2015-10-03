
#r "../../src/Hype/bin/Debug/DiffSharp.dll"
#r "../../src/Hype/bin/Debug/Hype.dll"
#I "../../packages/RProvider.1.1.14"
#load "RProvider.fsx"

open Hype
open Hype.Neural
open DiffSharp.AD.Float32
open DiffSharp.Util

open RDotNet
open RProvider
open RProvider.graphics



let dataOR = {X = toDM [[0.; 0.; 1.; 1.]
                        [0.; 1.; 0.; 1.]]
              Y = toDM [[0.; 1.; 1.; 0.]]}


let train (x:DV) =
    let par = {Params.Default with LearningRate = Scheduled x; Batch = Full; Verbose = false}
    let net = FeedForwardLayers()
    net.Add(LinearLayer(2, 1, Initializer.InitSigmoid))
    net.Add(ActivationLayer(sigmoid))
    
    Layer.Train(net, dataOR, par)
    Loss.L2Loss.Func dataOR net.Run

let test = grad train (DV.create 10 0.1f)

let hypertrain = 
    let report i w _ =
        if i % 2 = 0 then
            namedParams [   
                "x", box (w |> convert |> Array.map float);
                "type", box "o"; 
                "col", box "blue"]
            |> R.plot |> ignore
    let par = {Params.Default with Epochs = 20; ReportFunction = report}
    Optimizer.Minimize(train, (DV.create 10 0.1f), par)

