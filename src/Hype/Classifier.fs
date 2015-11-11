//
// This file is part of
// Hype: Compositional Machine Learning and Hyperparameter Optimization
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

open Hype
open Hype.Neural
open DiffSharp.AD.Float32
open DiffSharp.Util


[<AbstractClass>]
type Classifier(f:DM->DM) =
    let f = f
    member c.Run(x:DM) = f x
    member c.Run(x:DV) = x |> DM.ofDV x.Length |> f |> DM.toDV
    abstract member Classify : DM -> int[]
    abstract member Classify : DV -> int
    member c.ClassificationError(x:DM, y:int[]) =
        let cc = c.Classify(x)
        let incorrect = Array.map2 (fun c y -> if c = y then 0 else 1) cc y
        (float32 (incorrect |> Array.sum)) / (float32 incorrect.Length)
    member c.ClassificationError(d:Dataset) =
        c.ClassificationError(d.X, d.Yi)

type LogisticClassifier(f) =
    inherit Classifier(f)
    new(l:Layer) = LogisticClassifier(l.Run)
    override c.Classify(x:DM) =
        let cc = Array.zeroCreate x.Cols
        x |> f |> DM.iteriCols (fun i v -> if v.[0] > D 0.5f then cc.[i] <- 1)
        cc
    override c.Classify(x:DV) =
        if c.Run(x).[0] > D 0.5f then 1 else 0

type SoftmaxClassifier(f) =
    inherit Classifier(f)
    new(l:Layer) = SoftmaxClassifier(l.Run)
    override c.Classify(x:DM) = 
        let cc = Array.zeroCreate x.Cols
        x |> f |> DM.iteriCols (fun i v -> cc.[i] <- DV.MaxIndex(v))
        cc
    override c.Classify(x:DV) =
        DV.MaxIndex(c.Run(x))