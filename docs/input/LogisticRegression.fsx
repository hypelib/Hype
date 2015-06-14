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
open RProvider.grDevices

open FsAlg.Generic
open DiffSharp.AD
open DiffSharp.AD.Vector
open Hype

let m = Data.LoadImage @"diffsharp-logo.png"

namedParams [   
    "x", box (m |> Matrix.toArray2D);
    "col", box (R.grey_colors(10))]
|> R.image