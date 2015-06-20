#r "../../packages/FsAlg.0.5.12/lib/FsAlg.dll"
#r "../../packages/DiffSharp.0.6.2/lib/DiffSharp.dll"
#I "../../packages/RProvider.1.1.8"
#load "RProvider.fsx"

#load "../../src/Hype/Hype.fs"
#load "../../src/Hype/Optimize.fs"
fsi.ShowDeclarationValues <- false

open RDotNet
open RProvider
open RProvider.graphics

open FsAlg.Generic
open DiffSharp.AD
open DiffSharp.AD.Vector
open Hype

let housing = Util.LoadDelimited(__SOURCE_DIRECTORY__ + "/housing.data")
let housing' = housing |> Matrix.transpose |> Matrix.prependRow (Vector.create housing.Rows 1.)

let data = {X = housing'.[0..(housing'.Rows - 2), *] |> Matrix.map D
            Y = housing'.[housing'.Rows - 1..housing'.Rows - 1, *] |> Matrix.map D}

let train, test = data.[..250], data.[250..]



let h (w:Vector<D>) (x:Vector<D>) = vector [w * x]

let mutable w = Rnd.Vector(data.X.Rows)

w <- Train {DefaultParams with Epochs = 150; Batch = Minibatch 3} train h w

w <- Train {DefaultParams with Epochs = 10; Batch = Full; OptimizeFunction = Optimize.Newton} train h w

let model = h w

let trainloss = Loss.Quadratic train model
let testloss = Loss.Quadratic test model

let predict = test.ToSeq() |> Seq.map fst |> Seq.map model |> Seq.map (fun (v:Vector<_>) -> v.[0])

namedParams [
    "x", box (test.Y |> Matrix.row 0 |> Vector.map float |> Vector.toSeq)
    "pch", box 16
    "col", box "blue"
    "ylim", box [-20; 70]]
|> R.plot |> ignore

namedParams [
    "x", box (predict |> Seq.map float)
    "pch", box 16
    "col", box "red"]
|> R.points |> ignore
