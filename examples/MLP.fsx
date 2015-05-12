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



let OR = [|vector [D 0.; D 0.], vector [D 0.]
           vector [D 0.; D 1.], vector [D 1.]
           vector [D 1.; D 0.], vector [D 1.]
           vector [D 1.; D 1.], vector [D 1.]|]

let XOR = [|vector [D 0.; D 0.], vector [D 0.]
            vector [D 0.; D 1.], vector [D 1.]
            vector [D 1.; D 0.], vector [D 1.]
            vector [D 1.; D 1.], vector [D 0.]|]

let net = MLP.create([|2; 1|])

let train (x:Vector<_>) =
    let par = {Params.Default with LearningRate = ConstantLearningRate x.[0]; TrainFunction = Train.SGD; Epochs = 100}
    let net2 = MLP.create([|2; 1|])
    let f ww xx =
        net2.Decode ww
        net2.Run xx
    let q ww = OR |> Array.sumBy (fun (x, y) -> par.LossFunction y (f ww x))
    let op = Optimize.GD par q (net2.Encode |> Vector.map primal)
    op |> Array.last |> snd

let test2 = Optimize.GD {Params.Default with Epochs = 50} train (vector [D 15.56])

R.plot(test2 |> Array.map fst |> Array.map (fun (x:Vector<_>) -> float x.[0]))
