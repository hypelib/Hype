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

open FsAlg.Generic
open DiffSharp.AD
open Hype


type DataVS =
    {X:Matrix<D>
     y:Vector<D>}
    member d.Item
        with get i =
            d.X.[*,i], d.y.[i]
    member d.Length = d.X.Cols
    member d.ToSeq() =
        Seq.init d.Length (fun i -> d.[i])
    member d.Minibatch n =
        let bi = Rnd.Permutation(d.Length)
        {X = Matrix.init d.X.Rows n (fun i j -> d.X.[i,bi.[j]])
         y = Vector.init n (fun i -> d.y.[bi.[i]])}
    member d.Shuffle() = d.Minibatch d.Length
    member d.Sub(i, n) =
        {X = d.X.[*,i..(i+n-1)]
         y = d.y.[i..(i+n-1)]}
    member d.GetSlice(lower, upper) =
        let l = defaultArg lower 0
        let u = defaultArg upper (d.Length - 1)
        d.Sub(l, u - l + 1)

type DataVV =
    {X:Matrix<D>
     Y:Matrix<D>}
    member d.Item
        with get i =
            d.X.[*,i], d.Y.[*,i]
    member d.Length = d.X.Cols
    member d.ToSeq() =
        Seq.init d.Length (fun i -> d.[i])
    member d.Minibatch n =
        let bi = Rnd.Permutation(d.Length)
        {X = Matrix.init d.X.Rows n (fun i j -> d.X.[i,bi.[j]])
         Y = Matrix.init d.Y.Rows n (fun i j -> d.Y.[i,bi.[j]])}
    member d.Shuffle() = d.Minibatch d.Length
    member d.Sub(i, n) =
        {X = d.X.[*,i..(i+n-1)]
         Y = d.Y.[*,i..(i+n-1)]}
    member d.GetSlice(lower, upper) =
        let l = defaultArg lower 0
        let u = defaultArg upper (d.Length - 1)
        d.Sub(l, u - l + 1)

type Data =
    static member LoadImage(filename:string) =
        let bmp = new System.Drawing.Bitmap(filename)
        let m = Matrix.init bmp.Height bmp.Width (fun i j -> float (bmp.GetPixel(i, j).GetBrightness()))
        bmp.Dispose()
        m
    static member LoadDelimited(filename:string, separators:char[]) =
        System.IO.File.ReadLines(filename)
        |> Seq.map (fun x -> x.Split(separators) |> Array.map float)
        |> Seq.map vector
        |> Matrix.ofRows 
    static member LoadDelimited(filename:string) =
        Data.LoadDelimited(filename, [|' '; ','; '\t'|])