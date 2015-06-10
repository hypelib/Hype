#r "../packages/FsAlg.0.5.11/lib/FsAlg.dll"
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
open DiffSharp.AD.Vector
open Hype

let h (w:Vector<D>) (x:Vector<D>) = vector [w.[0] + w.[1] * x.[0]]

let t = LabeledSet.create [[|D 1.|], [|D 1.|]
                           [|D 2.|], [|D 2.|]
                           [|D 3.|], [|D 1.3|]
                           [|D 4.|], [|D 3.75|]
                           [|D 5.|], [|D 2.25|]]

let op (x:Vector<_>) = Optimize.GD {Params.Default with Epochs = 10; LearningRate = ConstantLearningRate (x.[0])} (fun w -> Loss.Quadratic t (h w)) (vector [D 0.1; D 0.1]) |> snd

let hyperop = Optimize.GD {Params.Default with Epochs = 20; LearningRate = ConstantLearningRate (D 0.00001)} op (vector [D 0.01])

