#r "../packages/FsAlg.0.5.12/lib/FsAlg.dll"
#r "../packages/DiffSharp.0.6.2/lib/DiffSharp.dll"
#I "../packages/RProvider.1.1.8"
#load "RProvider.fsx"

#load "../src/Hype/Hype.fs"
#load "../src/Hype/Data.fs"
#load "../src/Hype/Optimize.fs"
#load "../src/Hype/Regression.fs"


open RDotNet
open RProvider
open RProvider.graphics

open FsAlg.Generic
open DiffSharp.AD
open DiffSharp.AD.Vector
open Hype

let h (w:Vector<D>) (x:Vector<D>) = w * x

let housing = Data.LoadDelimited(__SOURCE_DIRECTORY__ + "/housing.data")
let housing' = housing |> Matrix.transpose |> Matrix.prependRow (Vector.create housing.Rows 1.)

let x = housing'.[0..(housing'.Rows - 2), *] |> Matrix.map D
let y = housing' |> Matrix.row (housing'.Rows - 1) |> Vector.map D

let data = {X = x; y = y}.Shuffle()
let train, test = data.[..250], data.[250..]

let model = Regression.Linear {Params.Default with Epochs = 150} data h (Rnd.Vector(data.X.Rows))

let trainloss = Loss.Quadratic(train, model)
let testloss = Loss.Quadratic(test, model)

let predict = Matrix.toCols test.X |> Seq.map model

namedParams [
    "x", box (test.y |> Vector.map float |> Vector.toSeq)
    "pch", box 16
    "col", box "blue"
    "ylim", box [-20; 70]]
|> R.plot |> ignore

namedParams [
    "x", box (predict |> Seq.map float)
    "pch", box 16
    "col", box "red"]
|> R.points |> ignore
