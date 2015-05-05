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

namespace Hype.Neural

open FsAlg.Generic
open DiffSharp.AD
open Hype
open Hype.Util
open Hype.Neural


type PerceptronLayer(W:Matrix<D>, b:Vector<D>, activation:D->D) =
    inherit Layer()
    let mutable W = W
    let mutable b = b
    let activation = activation
    override l.Run x = W * x + b |> Vector.map activation
    override l.Encode = Vector.append (Matrix.toVector W) b
    override l.Decode w =
        let ww = Vector.split [W.Rows * W.Cols; b.Length] w |> Array.ofSeq
        W <- Matrix.ofVector W.Rows ww.[0]
        b <- ww.[1]
    override l.EncodeLength = W.Rows * W.Cols + b.Length

type MLP =
    static member create(l:int[], activation, wmin, wmax) =
        Network(Array.init (l.Length - 1) (fun i ->
            PerceptronLayer(Matrix.init l.[i + 1] l.[i] (fun _ _ -> Rnd.NextD(wmin, wmax)), Vector.init l.[i + 1] (fun _ -> Rnd.NextD(wmin, wmax)), activation) :> Layer))
    static member create(l:int[]) = MLP.create(l, sigmoid, D -0.5, D 0.5)
