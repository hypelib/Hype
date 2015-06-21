#r "../../packages/FsAlg.0.5.13/lib/FsAlg.dll"
#r "../../packages/DiffSharp.0.6.2/lib/DiffSharp.dll"
#I "../../packages/RProvider.1.1.8"
#load "RProvider.fsx"

#load "../../src/Hype/Hype.fs"
//fsi.ShowDeclarationValues <- false

open RDotNet
open RProvider
open RProvider.graphics

open DiffSharp.AD
open DiffSharp.AD.Vector
open FsAlg.Generic
open System.IO

open Hype


   

