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
open DiffSharp.AD.Vector

type Layer =
    abstract member Run : Vector<D> -> Vector<D>
    abstract member Train : D -> unit

type PerceptronLayer =
    {W:Matrix<D>
     b:Vector<D>
     activation:D->D}
    interface Layer with
        member l.Run x = l.W * x + l.b |> Vector.map l.activation
        member l.Train learningrate =
            l.W |> Matrix.replace (fun (x:D) -> x.P - learningrate * x.A)
            l.b |> Vector.replace (fun (x:D) -> x.P - learningrate * x.A)

type Network =
    {layers:Layer[]}
    member n.Run (x:Vector<D>) =
        let runLayer (x:Vector<D>) (l:Layer) = l.Run x
        Array.fold runLayer x n.layers