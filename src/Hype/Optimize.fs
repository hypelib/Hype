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

open Hype
open FsAlg.Generic
open DiffSharp.AD
open DiffSharp.AD.Vector


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

type LabeledSet = 
    {mutable x: Matrix<D> 
     mutable y: Matrix<D>}
    member t.Item
        with get i =
            t.x.[*,i], t.y.[*,i]
    member t.Length = t.x.Rows
    member t.Append(x, y) =
        t.x <- Matrix.appendCol x t.x
        t.y <- Matrix.appendCol y t.y
    member t.ToSeq() =
        Seq.init t.Length (fun i -> t.[i])
    member t.Minibatch n =
        let bi = Array.init n (fun _ -> Rnd.Next(t.x.Cols))
        let bx = Seq.init n (fun i -> t.x.[*,bi.[i]]) |> Matrix.ofCols
        let by = Seq.init n (fun i -> t.y.[*,bi.[i]]) |> Matrix.ofCols
        {x = bx; y = by}
    static member create (s:seq<D[]*D[]>) =
        let x, y = s |> Seq.toArray |> Array.unzip
        {x = Matrix.ofArrayArray x; y = Matrix.ofArrayArray y}
        

type LearningRate =
    | ConstantLearningRate of D // Constant learning rate value
    | DecreasingLearningRate of D // Initial value of the learning rate, linearly drops to zero in Params.Epochs steps
    | DecayingLearningRate of D * D // Initial value and exponential decay parameter of the learning rate, drops to zero in Params.Epochs steps
    | ScheduledLearningRate of Vector<D> // Scheduled learning rate vector, its length overrides Params.Epochs

type Params =
    {Epochs : int
     MinibatchSize : int
     LearningRate : LearningRate
     LossFunction : LabeledSet->(Vector<D>->Vector<D>)->D
     TrainFunction: Params->LabeledSet->(Vector<D>->Vector<D>->Vector<D>)->Vector<D>->(Vector<D> * D)
     GDReportFunction : int->Vector<D>->D->unit}
    static member Default =
        {Epochs = 100
         MinibatchSize = 3
         LearningRate = ConstantLearningRate (D 0.001)
         LossFunction = Loss.Quadratic
         TrainFunction = Train.GD
         GDReportFunction = fun _ _ _ -> ()}



and Optimize =
    // w_t+1 = w_t - learningrate * grad Q(w)
    static member GD (par:Params) (q:Vector<D>->D) (w0:Vector<D>) =
        match par.LearningRate with
        | ConstantLearningRate l ->
            let mutable i = 0
            let mutable w = Vector.copy w0
            let mutable vv = D 0.
            while i < par.Epochs do
                let v, g = grad' q w
                par.GDReportFunction i w v
                vv <- v
                w <- w - l * g
                i <- i + 1
            w, vv
        | DecreasingLearningRate l ->
            let epochs = float par.Epochs
            let mutable i = 0
            let mutable w = Vector.copy w0
            let mutable vv = D 0.
            while i < par.Epochs do
                let v, g = grad' q w
                par.GDReportFunction i w v
                vv <- v
                w <- w - (l * (1. - (float i + 1.) / epochs)) * g
                i <- i + 1
            w, vv
        | DecayingLearningRate (l, r) ->
            let mutable i = 0
            let mutable w = Vector.copy w0
            let mutable vv = D 0.
            while i < par.Epochs do
                let v, g = grad' q w
                par.GDReportFunction i w v
                vv <- v
                w <- w - l * (exp (-r * (float i))) * g
                i <- i + 1
            w, vv
        | ScheduledLearningRate l ->
            let mutable i = 0
            let mutable w = Vector.copy w0
            let mutable vv = D 0.
            while i < l.Length do
                let v, g = grad' q w
                par.GDReportFunction i w v
                vv <- v
                w <- w - l.[i] * g
                i <- i + 1
            w, vv

and Loss =
    static member Quadratic (t:LabeledSet) (f:Vector<D>->Vector<D>) = 
        t.ToSeq() |> Seq.map (fun (x, y) -> Vector.normSq (y - f x)) |> Seq.sum

and Train =
    /// Gradient descent
    static member GD (par:Params) (t:LabeledSet) (f:Vector<D>->Vector<D>->Vector<D>) (w0:Vector<D>)  =
        let q w = par.LossFunction t (f w)
        Optimize.GD par q w0

    /// Minibatch stochastic gradient descent
    static member MSGD (par:Params) (t:LabeledSet) (f:Vector<D>->Vector<D>->Vector<D>) (w0:Vector<D>) =
        let q w = par.LossFunction (t.Minibatch par.MinibatchSize) (f w)
        Optimize.GD par q w0

    /// Stochastic gradient descent
    static member SGD (par:Params) (t:LabeledSet) (f:Vector<D>->Vector<D>->Vector<D>) (w0:Vector<D>) =
        Train.MSGD {par with MinibatchSize = 1} t f w0