#r "../packages/FsAlg.0.5.11/lib/FsAlg.dll"
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

let data = {X = x; y = y}
let data2 = data.Minibatch 10

let w0 = Rnd.Vector(data2.X.Rows)

let w, v = Regression.Linear {Params.Default with Epochs = 150; LearningRate = DecreasingLearningRate (D 0.0000001)} data2 h w0

let fitted = h w

let p = (h w) (fst data2.[2])
let r = snd data2.[2]