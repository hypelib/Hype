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
    | Constant of D // Constant learning rate value
    | Decreasing of D * D // Initial value, decay rate of the learning rate
    | Scheduled of Vector<D> // Scheduled learning rate vector, its length overrides Params.Epochs
    | Backtracking of D * D // Backtracking line search

type Momentum =
    | Momentum of D
    | None

type Params =
    {Epochs : int
     MinibatchSize : int
     LearningRate : LearningRate
     Momentum : Momentum
     LossFunction : (DataVV*(Vector<D>->Vector<D>))->D
     TrainFunction: Params->DataVV->(Vector<D>->Vector<D>->Vector<D>)->Vector<D>->Vector<D>
     GDReportFunction : int->Vector<D>->D->unit}
    static member Default =
        {Epochs = 100
         MinibatchSize = 2
         LearningRate = Backtracking (D 0.25, D 0.75)
         Momentum = Momentum (D 0.5)
         LossFunction = Loss.Quadratic
         TrainFunction = Train.GD
         GDReportFunction = fun _ _ _ -> ()}

and Optimize =
    static member GD (par:Params) (f:Vector<D>->D) (w0:Vector<D>) =
        let epochs = 
            match par.LearningRate with
            | Scheduled l -> l.Length
            | _ -> par.Epochs
        let momentum =
            match par.Momentum with
            | Momentum m -> fun (uprev:Vector<D>) (u:Vector<D>) -> m * uprev + (D 1. - m) * u
            | None -> fun _ u -> u
        let update =
            match par.LearningRate with
            | Constant l -> 
                fun _ w f -> 
                    let v, g = grad' f w    
                    let w' = -l * g
                    v, w'
            | Decreasing (l0, t) ->
                fun i w f -> 
                    let v, g = grad' f w
                    let w' = -l0 * t  * g / (t + float i)
                    v, w'
            | Scheduled l -> 
                fun i w f -> 
                    let v, g = grad' f w
                    let w' = -l.[i] * g
                    v, w'
            | Backtracking (a, b) ->
                fun _ w f -> 
                    let v, g = grad' f w
                    let gg = Vector.normSq g
                    let mutable l = D 1.
                    while f (w - l * g) > v - a * l * gg do
                        l <- l * b
                    let w' = -l * g
                    v, w'
        let mutable i = 0
        let mutable w = Vector.copy w0
        let mutable v = f w0
        let mutable u = Vector.create w.Length (D 0.)
        while i < epochs do
            par.GDReportFunction i w v
            let v', u' = update i w f
            u <- momentum u u'
            w <- w + u
            v <- v'
            i <- i + 1
        w


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
    static member MSGD (par:Params) (trainData:DataVV) (f:Vector<D>->Vector<D>->Vector<D>) (w0:Vector<D>) =
        let i = ref 0
        let minibatch = ref trainData
        let q w = // It's better to sequence through a shuffled dataset than creating a random minibatch each time
            if !i + par.MinibatchSize >= trainData.Length then
                minibatch := trainData.[!i..]
                i := 0
            else
                minibatch := trainData.Sub(!i, par.MinibatchSize)
                i := !i + par.MinibatchSize
            par.LossFunction(!minibatch, f w)
        Optimize.GD par q w0

    /// Stochastic gradient descent
    static member SGD (par:Params) (t:DataVV) (f:Vector<D>->Vector<D>->Vector<D>) (w0:Vector<D>) =
        Train.MSGD {par with MinibatchSize = 1} t f w0