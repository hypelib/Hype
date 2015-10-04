
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



let dataOR = {X = toDM [[0.; 0.; 1.; 1.]
                        [0.; 1.; 0.; 1.]]
              Y = toDM [[0.; 1.; 1.; 1.]]}



let train (x:DV) =
    let par = {Params.Default with LearningRate = Schedule x; Batch = Full; Silent = true; ReturnBest = false}
    let net = FeedForwardLayers()
    net.Add(LinearLayer(2, 1, Initializer.InitUniform(D -1.41f, D 1.41f)))
    net.Add(ActivationLayer(sigmoid))
        
    Layer.Train(net, dataOR, par)

    Loss.L2Loss.Func dataOR net.Run

let hypertrain = 
    let report i (w:DV) _ =
        namedParams [   
            "x", box (w |> convert |> Array.map float);
            "type", box "o"; 
            //"ylim", box [0.5; 2.];
            "col", box "blue"]
        |> R.plot |> ignore
    Optimize.Minimize(train, DV.create 50 (1.f), {Params.Default with Epochs = 1500; ReportFunction = report; ValidationInterval = 1; LearningRate = AdaGrad (D 0.001f); EarlyStopping = Early(400, 100)})


let test = grad' train (DV.create 40 (1.f))
