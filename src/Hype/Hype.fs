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

open System.IO
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


type Data =
    {X:Matrix<D>
     Y:Matrix<D>}
    static member ofSeq (s:seq<Vector<D>*Vector<D>>) =
        let x, y = s |> Seq.toArray |> Array.unzip
        {X = x |> Matrix.ofCols
         Y = y |> Matrix.ofCols}
    member d.Item
        with get i =
            d.X.[*,i] |> Matrix.toVector, d.Y.[*,i] |> Matrix.toVector
    member d.Length = d.X.Cols
    member d.ToSeq() =
        Seq.init d.Length (fun i -> d.[i])
    member d.Minibatch n =
        let bi = Rnd.Permutation(d.Length)
        let x = Seq.init n (fun i -> d.X |> Matrix.col bi.[i] |> Matrix.toVector)
        let y = Seq.init n (fun i -> d.Y |> Matrix.col bi.[i] |> Matrix.toVector)
        {X = Matrix.ofCols x
         Y = Matrix.ofCols y}
    member d.Shuffle() = d.Minibatch d.Length
    member d.Sub(i, n) =
        {X = d.X.[*,i..(i+n-1)]
         Y = d.Y.[*,i..(i+n-1)]}
    member d.GetSlice(lower, upper) =
        let l = defaultArg lower 0
        let u = defaultArg upper (d.Length - 1)
        d.Sub(l, u - l + 1)
    member d.Filter (predicate:(Vector<D>*Vector<D>)->bool) =
        d.ToSeq() |> Seq.filter predicate |> Data.ofSeq


type Util =
    static member printLog (s:string) = printfn "[%A] %s" System.DateTime.Now s
    static member printModel (f:Vector<D>->Vector<D>) (d:Data) =
        d.ToSeq()
        |> Seq.map (fun (x, y) -> f x, y)
        |> Seq.iter (fun (x, y) -> printfn "f x: %A, y: %A" x y)
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
        Util.LoadDelimited(filename, [|' '; ','; '\t'|])
    static member LoadMNIST(filename, items) =
        let d = new BinaryReader(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read))
        let magicnumber = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2049 -> // Labels
            let maxitems = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            d.ReadBytes(min items maxitems)
            |> Array.map float
            |> Matrix.ofArray 1
        | 2051 -> // Images
            let maxitems = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let rows = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let cols = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let n = min items maxitems
            d.ReadBytes(n * rows * cols)
            |> Array.map float
            |> Matrix.ofArray n
            |> Matrix.transpose
        | _ -> failwith "Given file is not in the MNIST format."
    static member LoadMNIST(filename) = Util.LoadMNIST(filename, System.Int32.MaxValue)


[<AutoOpen>]
module Activation =
    let inline sigmoid (x:D) = D 1. / (D 1. + exp -x)
    let inline softSign (x:D) = x / (D 1. + abs x)
    let inline softPlus (x:D) = log (D 1. + exp x)
    let inline rectifiedLinear (x:D) = max (D 0.) x


[<RequireQualifiedAccess>]
module Loss =
    let inline Quadratic (d:Data) (f:Vector<D>->Vector<D>) = 
        (d.ToSeq() |> Seq.sumBy (fun (x, y) -> Vector.normSq (y - f x))) / d.Length
    let inline Manhattan (d:Data) (f:Vector<D>->Vector<D>) =
        (d.ToSeq() |> Seq.sumBy (fun (x, y) -> Vector.l1norm (y - f x))) / d.Length
