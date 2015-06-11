//
// This file is part of
// Hype: Machine Learning and Hyperparameter Optimization Library
//
// Copyright (c) 2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
//
// Hype is released under the MIT license.
// (See accompanying LICENSE file.)
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Brain and Computation Lab
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

namespace Hype

open DiffSharp.AD
open FsAlg.Generic

type Rnd() =
    static let R = new System.Random()
    static member Next() = R.Next()
    static member Next(max) = R.Next(max)
    static member Next(min,max) = R.Next(min, max)
    static member NextD() = D (R.NextDouble())
    static member NextD(max) = 
        if max < D 0. then invalidArg "max" "Max should be nonnegative."
        R.NextDouble() * max
    static member NextD(min,max) = min + D (R.NextDouble()) * (max - min)
    static member Permutation(n:int) =
        let swap i j (a:_[]) =
            let tmp = a.[i]
            a.[i] <- a.[j]
            a.[j] <- tmp
        let a = Array.init n (fun i -> i)
        a |> Array.iteri (fun i _ -> swap i (R.Next(i, n)) a)
        a
    static member Vector(n) = Vector.init n (fun _ -> Rnd.NextD())
    static member Vector(n,max) = Vector.init n (fun _ -> Rnd.NextD(max))
    static member Vector(n,min,max) = Vector.init n (fun _ -> Rnd.NextD(min, max))

[<RequireQualifiedAccess>]
module Activation =
    let inline sigmoid (x:D) = D 1. / (D 1. + exp -x)
    let inline softSign (x:D) = x / (D 1. + abs x)
    let inline softPlus (x:D) = log (D 1. + exp x)
    let inline rectifiedLinear (x:D) = max (D 0.) x

