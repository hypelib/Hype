#r "../../packages/FsAlg.0.5.5/lib/FsAlg.dll"
#r "../../packages/DiffSharp.0.6.0/lib/DiffSharp.dll"
#r "bin/Debug/Hype.dll"
#I "../../packages/RProvider.1.1.8"
#load "RProvider.fsx"

open RDotNet
open RProvider
open RProvider.graphics

open FsAlg.Generic
open DiffSharp.AD
open DiffSharp.AD.Vector
open Hype



let f (x:Vector<_>) = sin x.[0] - cos x.[1]

let f2 (x:Vector<_>) =
    let par = {OptimizeParams.Default with LearningRate = ConstantLearningRate x.[0]}
    let op = Optimize.GD par f (vector [D 1.; D 1.])
    op |> snd |> Array.last

let f3 (x:Vector<_>) =
    let par = {OptimizeParams.Default with LearningRate = ScheduledLearningRate x}
    let op = Optimize.GD par f (vector [D 1.; D 1.])
    op |> snd |> Array.last

//let test = Optimize.GD DefaultParams f (vector [D 1.; D 1.])
let test = Optimize.GD {OptimizeParams.Default with Epochs = 50000} f3 (vector [D 1.; D 1.; D 1.])


let pp = test |> snd |> Array.map float

R.plot(pp)