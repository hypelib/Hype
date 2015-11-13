(*** hide ***)
#r "../../src/Hype/bin/Release/DiffSharp.dll"
#r "../../src/Hype/bin/Release/Hype.dll"
#I "../../packages/R.NET.Community.1.6.4/lib/net40/"
#I "../../packages/R.NET.Community.FSharp.1.6.4/lib/net40/"
#I "../../packages/RProvider.1.1.14"
#load "RProvider.fsx"
fsi.ShowDeclarationValues <- true

open Hype
open Hype.Neural
open DiffSharp.AD.Float32
open DiffSharp.Util

open RDotNet
open RProvider
open RProvider.graphics




let dataor = {X = toDM [[0.; 0.; 1.; 1.]
                        [0.; 1.; 0.; 1.]]
              Y = toDM [[0.; 1.; 1.; 1.]]}



let n = FeedForward()
n.Add(Linear(2, 4))
n.Add(Activation(sigmoid))
n.Add(Linear(4, 1))
n.Add(Activation(sigmoid))

n.ToStringFull() |> printfn "%s"

Layer.Train(n, dataor)