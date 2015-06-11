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

open FsAlg.Generic
open DiffSharp.AD
open Hype


type DataVS =
    {X:Matrix<D>
     y:Vector<D>}
    member t.Item
        with get i =
            t.X.[*,i], t.y.[i]
    member t.Length = t.X.Cols
    member t.ToSeq() =
        Seq.init t.Length (fun i -> t.[i])
    member t.Minibatch n =
        let bi = Rnd.Permutation(t.Length)
        {X = Matrix.init t.X.Rows n (fun i j -> t.X.[i,bi.[j]])
         y = Vector.init n (fun i -> t.y.[bi.[i]])}
    member t.Shuffle() = t.Minibatch t.Length
    member t.Split n =
        let m = t.Length - n
        if m <= 0 then invalidArg "n" "Split must occur before the end of the data set."
        let a = {X = Matrix.init t.X.Rows n (fun i j -> t.X.[i,j])
                 y = Vector.init n (fun i -> t.y.[i])}
        let b = {X = Matrix.init t.X.Rows m (fun i j -> t.X.[i,n + j])
                 y = Vector.init m (fun i -> t.y.[n + i])}
        a, b

type DataVV =
    {X:Matrix<D>
     Y:Matrix<D>}
    member t.Item
        with get i =
            t.X.[*,i], t.Y.[*,i]
    member t.Length = t.X.Cols
    member t.ToSeq() =
        Seq.init t.Length (fun i -> t.[i])
    member t.Minibatch n =
        let bi = Rnd.Permutation(t.Length)
        {X = Matrix.init t.X.Rows n (fun i j -> t.X.[i,bi.[j]])
         Y = Matrix.init t.Y.Rows n (fun i j -> t.Y.[i,bi.[j]])}
    member t.Shuffle() = t.Minibatch t.Length
    member t.Split n =
        let m = t.Length - n
        if m <= 0 then invalidArg "n" "Split must occur before the end of the data set."
        let a = {X = Matrix.init t.X.Rows n (fun i j -> t.X.[i,j])
                 Y = Matrix.init t.Y.Rows n (fun i j -> t.Y.[i,j])}
        let b = {X = Matrix.init t.X.Rows m (fun i j -> t.X.[i,n + j])
                 Y = Matrix.init t.Y.Rows m (fun i j -> t.Y.[i,n + j])}
        a, b

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