#r "../../packages/FsAlg.0.5.12/lib/FsAlg.dll"
#r "../../packages/DiffSharp.0.6.2/lib/DiffSharp.dll"
#I "../../packages/RProvider.1.1.8"
#load "RProvider.fsx"

#load "../../src/Hype/Hype.fs"
#load "../../src/Hype/Optimize.fs"
fsi.ShowDeclarationValues <- false

open DiffSharp.AD
open DiffSharp.AD.Vector
open FsAlg.Generic

open Hype

let f (x:Vector<D>) = (exp (x.[0] - 1)) + (exp (- x.[1] + 1)) + ((x.[0] - x.[1]) ** 2)

let test = Optimize.Newton {DefaultParams with Epochs = 20} f (vector [D 0.; D 0.])