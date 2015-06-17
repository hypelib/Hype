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

namespace Hype.Neural

open FsAlg.Generic
open DiffSharp.AD
open Hype

[<AbstractClass>]
type Layer() =
    abstract member Run : Vector<D> -> Vector<D>
    abstract member Encode : Vector<D>
    abstract member Decode : Vector<D> -> unit
    abstract member EncodeLength: int
    member l.Train (par:Params) (t:DataVV) =
        let f w x =
            l.Decode w
            l.Run x
        par.TrainFunction par t f (l.Encode)
        |> l.Decode


[<AutoOpen>]
module LayerOps =
    let runLayer x (l:Layer) = l.Run x
    let encodeLayer (l:Layer) = l.Encode
    let decodeLayer w (l:Layer) = l.Decode w
    let encodeLength (l:Layer) = l.EncodeLength
    let trainLayer par t (l:Layer) = l.Train par t


type Network(layers:Layer[]) =
    inherit Layer()
    let layers = layers
    override n.Run (x:Vector<D>) = Array.fold runLayer x layers
    override n.Encode = layers |> Array.map encodeLayer |> Array.reduce Vector.append
    override n.Decode w =
        let ww = Vector.split (layers |> Array.map encodeLength) w |> Array.ofSeq
        Array.iter2 decodeLayer ww layers
    override n.EncodeLength = layers |> Array.map encodeLength |> Array.sum
