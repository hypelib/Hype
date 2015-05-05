#r "../packages/FsAlg.0.5.7/lib/FsAlg.dll"
#r "../packages/DiffSharp.0.6.0/lib/DiffSharp.dll"
#r "../src/Hype/bin/Debug/Hype.dll"
#I "../packages/RProvider.1.1.8"
#load "RProvider.fsx"

open RDotNet
open RProvider
open RProvider.graphics

open FsAlg.Generic
open DiffSharp.AD
open Hype
open Hype.Neural

let net = MLP.create([|2; 2; 1|])

let trainOR = [|vector [D 0.; D 0.], vector [D 0.]
                vector [D 0.; D 1.], vector [D 1.]
                vector [D 1.; D 0.], vector [D 1.]
                vector [D 1.; D 1.], vector [D 1.]|]

let trainXOR = [|vector [D 0.; D 0.], vector [D 0.]
                 vector [D 0.; D 1.], vector [D 1.]
                 vector [D 1.; D 0.], vector [D 1.]
                 vector [D 1.; D 1.], vector [D 0.]|]

//let _, test = Train.GD {OptimizeParams.Default with Epochs = 1000} trainOR f (net.Encode)

let test = trainLayer {Params.Default with TrainFunction = Train.SGD} trainXOR net

R.plot (test |> Array.map float)