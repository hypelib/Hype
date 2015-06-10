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

     

type LearningRate =
    | ConstantLearningRate of D // Constant learning rate value
    | DecreasingLearningRate of D // Initial value of the learning rate, linearly drops to zero in Params.Epochs steps
    | DecayingLearningRate of D * D // Initial value and exponential decay parameter of the learning rate, drops to zero in Params.Epochs steps
    | ScheduledLearningRate of Vector<D> // Scheduled learning rate vector, its length overrides Params.Epochs


type Params =
    {Epochs : int
     MinibatchSize : int
     LearningRate : LearningRate
     LossFunction : (DataVV*(Vector<D>->Vector<D>))->D
     TrainFunction: Params->DataVV->(Vector<D>->Vector<D>->Vector<D>)->Vector<D>->(Vector<D> * D)
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
    static member Quadratic(t:DataVS, f:Vector<D>->D) =
        (t.ToSeq() |> Seq.map (fun (x, y) -> y - f x) |> Seq.sumBy (fun x -> x * x)) * D 0.5
    static member Quadratic(t:DataVV, f:Vector<D>->Vector<D>) = 
        (t.ToSeq() |> Seq.map (fun (x, y) -> Vector.normSq (y - f x)) |> Seq.sum) * D 0.5

and Train =
    /// Gradient descent
    static member GD (par:Params) (t:DataVV) (f:Vector<D>->Vector<D>->Vector<D>) (w0:Vector<D>)  =
        let q w = par.LossFunction(t, f w)
        Optimize.GD par q w0

    /// Minibatch stochastic gradient descent
    static member MSGD (par:Params) (t:DataVV) (f:Vector<D>->Vector<D>->Vector<D>) (w0:Vector<D>) =
        let q w = par.LossFunction(t.Minibatch par.MinibatchSize, f w)
        Optimize.GD par q w0

    /// Stochastic gradient descent
    static member SGD (par:Params) (t:DataVV) (f:Vector<D>->Vector<D>->Vector<D>) (w0:Vector<D>) =
        Train.MSGD {par with MinibatchSize = 1} t f w0