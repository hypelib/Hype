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

type Batch =
    | Full
    | Minibatch of int
    | Stochastic // Minibatch with size 1

type Params =
    {Epochs : int
     LearningRate : LearningRate
     Momentum : Momentum
     LossFunction : Data->(Vector<D>->Vector<D>)->D
     OptimizeFunction: Params->(Vector<D>->D)->Vector<D>->Vector<D>
     Batch : Batch
     Verbose : bool
     ReportFunction : int->Vector<D>->D->unit
     ReportInterval : int}

type Optimize =
    static member GD (par:Params) (f:Vector<D>->D) (w0:Vector<D>) =
        Util.printLog "Method: Gradient descent"
        Util.printLog (sprintf "Dimensions: %A" w0.Length)
        let epochs = 
            match par.LearningRate with
            | Scheduled l -> l.Length
            | _ -> par.Epochs
        Util.printLog (sprintf "Epochs: %A" epochs)
        let momentum =
            match par.Momentum with
            | Momentum m -> 
                Util.printLog (sprintf "Momentum: Constant %A" m)
                fun (uprev:Vector<D>) (u:Vector<D>) -> m * uprev + (D 1. - m) * u
            | None -> 
                Util.printLog "Momentum: none"
                fun _ u -> u
        let update =
            match par.LearningRate with
            | Constant l -> 
                Util.printLog (sprintf "Learning rate: Constant %A" l)
                fun _ w f -> 
                    let v, g = grad' f w    
                    let w' = -l * g
                    v, w'
            | Decreasing (l0, t) ->
                Util.printLog (sprintf "Learning rate: Decreasing %A %A" l0 t)
                fun i w f -> 
                    let v, g = grad' f w
                    let w' = -l0 * t  * g / (t + float i)
                    v, w'
            | Scheduled l -> 
                Util.printLog (sprintf "Learning rate: Scheduled of length %A" l.Length)
                fun i w f -> 
                    let v, g = grad' f w
                    let w' = -l.[i] * g
                    v, w'
            | Backtracking (a, b) ->
                Util.printLog (sprintf "Learning rate: Backtracking %A %A" a b)
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
        if par.Verbose then 
            Util.printLog (sprintf "Epoch %A function value: %A" i (primal v))
        par.ReportFunction i w v
        while i < epochs do
            let v', u' = update i w f
            u <- momentum u u'
            w <- w + u
            v <- v'
            i <- i + 1
            if i % par.ReportInterval = 0 then 
                if par.Verbose then 
                    Util.printLog (sprintf "Epoch %A function value: %A" i (primal v))
                par.ReportFunction i w v
        w

    static member Newton (par:Params) (f:Vector<D>->D) (w0:Vector<D>) =
        Util.printLog "Method: Exact Newton"
        Util.printLog (sprintf "Dimensions: %A" w0.Length)
        Util.printLog (sprintf "Epochs: %A" par.Epochs)
        let mutable i = 0
        let mutable w = Vector.copy w0
        let mutable v = f w0
        par.ReportFunction i w v
        if par.Verbose then 
            Util.printLog (sprintf "Epoch %A function value: %A" i (primal v))
        while i < par.Epochs do
            let v', g, h = gradhessian' f w
            let p = - Matrix.solve h g
            w <- w + p
            v <- primal v'
            i <- i + 1
            if i % par.ReportInterval = 0 then
                if par.Verbose then 
                    Util.printLog (sprintf "Epoch %A function value: %A" i (primal v))
                par.ReportFunction i w v
        w


[<AutoOpen>]
module Ops =
    let DefaultParams = {Epochs = 100
                         LearningRate = Backtracking (D 0.25, D 0.75)
                         Momentum = Momentum (D 0.5)
                         LossFunction = Loss.Quadratic
                         OptimizeFunction = Optimize.GD
                         Batch = Minibatch 3
                         Verbose = true
                         ReportFunction = fun _ _ _ -> ()
                         ReportInterval = 10}

    let Train (par:Params) (d:Data) (f:Vector<D>->Vector<D>->Vector<D>) (w0:Vector<D>) =
        Util.printLog "Training started."
        let start = System.DateTime.Now
        let l0 = par.LossFunction d (f w0)
        let q =
            match par.Batch with
            | Full ->
                Util.printLog "Batch: Full"
                fun w -> par.LossFunction d (f w)
            | Minibatch n ->
                Util.printLog (sprintf "Batch: Minibatches of %A" n)
                fun w -> par.LossFunction (d.Minibatch n) (f w)
            | Stochastic ->
                Util.printLog "Batch: Stochastic"
                fun w -> par.LossFunction (d.Minibatch 1) (f w)
        let report i w v =
            if par.Verbose then
                let loss = par.LossFunction d (f (w |> Vector.map primal))
                Util.printLog (sprintf "Epoch %A training loss: %A" i (primal loss))
            par.ReportFunction i w v
        let wopt = par.OptimizeFunction {par with ReportFunction = report; Verbose = false} q w0
        let duration = System.DateTime.Now.Subtract(start)
        let lf = par.LossFunction d (f wopt)
        let dec = -(lf - l0)
        let perf = dec / duration.TotalSeconds
        Util.printLog "Training finished."
        Util.printLog (sprintf "Duration: %A" duration)
        Util.printLog (sprintf "Loss decrease: %A (%.2f %%)" (primal dec) (float (100 * (dec) / l0)))
        Util.printLog (sprintf "Performance (loss decrease / sec): %A\n" (primal perf))
        wopt
        