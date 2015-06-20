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
    | Constant of D // Constant
    | Decreasing of D * D // Initial value, decay rate
    | Scheduled of Vector<D> // Scheduled learning rate vector, its length overrides Params.Epochs
    | Backtracking of D * D * D // Backtracking line search, initial value, c, rho
    | StrongWolfe of D * D * D // Strong Wolfe line search, lmax, c1, c2 

type Momentum =
    | Momentum of D
    | None

type Batch =
    | Full
    | Minibatch of int
    | Stochastic // Minibatch with size 1

type OptimizeMethod =
    | GD
    | CG
    | CD
    | NonlinearCG
    | DaiYuanCG
    | NewtonCG
    | Newton
    | Custom of (Params->(Vector<D>->D)->Vector<D>->Vector<D>)

and Params =
    {Epochs : int
     LearningRate : LearningRate
     Momentum : Momentum
     LossFunction : Data->(Vector<D>->Vector<D>)->D
     OptimizeMethod: OptimizeMethod
     Batch : Batch
     Verbose : bool
     LoopLimit : int
     ReportFunction : int->Vector<D>->D->unit
     ReportInterval : int}


[<AutoOpen>]
module Ops =
    let DefaultParams = {Epochs = 100
                         LearningRate = Backtracking (D 1., D 0.0001, D 0.5)
                         Momentum = None
                         LossFunction = Loss.Quadratic
                         OptimizeMethod = GD
                         Batch = Minibatch 3
                         Verbose = true
                         LoopLimit = 1000
                         ReportFunction = fun _ _ _ -> ()
                         ReportInterval = 10}


    let Optimize (par:Params) (f:Vector<D>->D) (w0:Vector<D>) =
        match par.OptimizeMethod with
        | Custom c -> c par f w0
        | _ ->
            let dir =
                match par.OptimizeMethod with
                | GD ->
                    Util.printLog "Method: Gradient descent"
                    fun _ w f _ _ _ ->
                        let v', g' = grad' f w
                        v', g', -g'
                | CG -> // Hestenes and Stiefel 1952
                    Util.printLog "Method: Conjugate gradient"
                    fun _ w f _ g p ->
                        let v', g' = grad' f w
                        let y = g' - g
                        let b = (g' * y) / p * y
                        let p' = -g' + b * p
                        v', g', p'
                | CD -> // Fletcher 1987
                    Util.printLog "Method: Conjugate descent"
                    fun _ w f _ g p ->
                        let v', g' = grad' f w
                        let b = (Vector.normSq g') / (-p * g)
                        let p' = -g' + b * p
                        v', g', p'
                | DaiYuanCG -> // Dai and Yuan 1999
                    Util.printLog "Method: Dai & Yuan conjugate gradient"
                    fun _ w f _ g p ->
                        let v', g' = grad' f w
                        let y = g' - g
                        let b = (Vector.normSq g') / p * y
                        let p' = -g' + b * p
                        v', g', p'
                | NonlinearCG -> // Fletcher and Reeves 1964
                    Util.printLog "Method: Nonlinear conjugate gradient"
                    fun _ w f _ g p ->
                        let v', g' = grad' f w
                        let b = (Vector.normSq g') / (Vector.normSq g)
                        let p' = -g' + b * p
                        v', g', p'
                | NewtonCG ->
                    Util.printLog "Method: Newton conjugate gradient"
                    fun _ w f _ _ p ->
                        let v', g' = grad' f w
                        let hv = hessianv f w p
                        let b = (g' * hv) / (p * hv)
                        let p' = -g' + b * p
                        v', g', p'
                | Newton ->
                    Util.printLog "Method: Exact Newton"
                    fun _ w f _ _ _ ->
                        let v', g', h' = gradhessian' f w
                        v', g', - Matrix.solve h' g'
            Util.printLog (sprintf "Dimensions: %A" w0.Length)
            let epochs = 
                match par.LearningRate with
                | Scheduled a -> a.Length
                | _ -> par.Epochs
            Util.printLog (sprintf "Epochs: %A" epochs)
            let lr =
                match par.LearningRate with
                | Constant a ->
                    Util.printLog (sprintf "Learning rate: Constant %A" a)
                    fun _ _ _ _ _ _ -> a
                | Decreasing (a0, t) ->
                    Util.printLog (sprintf "Learning rate: Decreasing %A %A" a0 t)
                    fun i _ _ _ _ _ -> a0 * t / (t + float i)
                | Scheduled a ->
                    Util.printLog (sprintf "Learning rate: Scheduled of length %A" a.Length)
                    fun i _ _ _ _ _ -> a.[i]
                | Backtracking (a0, c, r) ->
                    Util.printLog (sprintf "Learning rate: Backtracking %A %A %A" a0 c r)
                    fun _ w f v g p ->
                        let mutable a = a0
                        let mutable i = 0
                        let mutable found = false
                        while not found do
                            if f (w + a * p) < v + c * a * (p * g) then 
                                found <- true
                            else
                                a <- r * a
                            i <- i + 1
                            if i > par.LoopLimit then
                                found <- true
                                Util.printLog "WARNING: Backtracking did not converge."
                        a
                | StrongWolfe (amax, c1, c2) ->
                    Util.printLog (sprintf "Learning rate: Strong Wolfe %A %A %A" amax c1 c2)
                    fun _ w f v g p ->
                        let v0 = v
                        let gp0 = g * p
                        let inline zoom a1 a2 =
                            let mutable al = a1
                            let mutable ah = a2
                            let mutable a' = a1
                            let mutable v'al = f (w + al * p)
                            let mutable i = 0
                            let mutable found = false
                            while not found do
                                a' <- (al + ah) / D 2.
                                let v', gg = grad' f (w + a' * p)
                                if (v' > v0 + c1 * a' * gp0) || (v' >= v'al) then
                                    ah <- a'
                                else
                                    let gp' = gg * p
                                    if abs gp' <= -c2 * gp0 then
                                        found <- true
                                    elif gp' * (ah - al) >= D 0. then
                                        ah <- al
                                        al <- a'
                                        v'al <- v'
                                i <- i + 1
                                if i > par.LoopLimit then
                                    found <- true
                                    Util.printLog "WARNING: Strong Wolfe (zoom) did not converge."
                            a'
                            
                        let mutable v = v0
                        let mutable v' = v0
                        let mutable gp' = gp0
                        let mutable a = D 0.
                        let mutable a' = Rnd.NextD(amax)
                        let mutable a'' = a'
                        let mutable i = 1
                        let mutable found = false
                        while not found do
                            let vv, gg = grad' f (w + a' * p)
                            v' <- vv
                            gp' <- gg * p
                            if (v' > v0 + c1 * a' * gp0) || ((i > 1) && (v' >= v)) then
                                a'' <- zoom a a'
                                found <- true
                            elif (abs gp') <= (-c2 * gp0) then
                                a'' <- a'
                                found <- true
                            elif gp' >= D 0. then
                                a'' <- zoom a' a
                                found <- true
                            else
                                a <- a'
                                v <- v'
                                a' <- Rnd.NextD(a', amax)
                                i <- i + 1
                            if i > par.LoopLimit then
                                found <- true
                                Util.printLog "WARNING: Strong Wolfe did not converge."
                        a''

            let momentum =
                match par.Momentum with
                | Momentum m ->
                    Util.printLog (sprintf "Momentum: Constant %A" m)
                    fun (p:Vector<D>) (p':Vector<D>) -> m * p + (D 1. - m) * p'
                | None ->
                    Util.printLog "Momentum: None"
                    fun _ p' -> p'
            let mutable i = 0
            let mutable w = Vector.copy w0
            let v, g = grad' f w0
            let mutable v = v
            let mutable g = g
            let mutable p = -g
            par.ReportFunction i w v
            if par.Verbose then
                Util.printLog (sprintf "Epoch %A function value: %A" i (primal v))
            while i < epochs do
                let v', g', p' = dir i w f v g p
                v <- v'
                g <- g'
                p <- momentum p p'
                w <- w + (lr i w f v g p) * p
                i <- i + 1
                if i % par.ReportInterval = 0 then
                    if par.Verbose then
                        Util.printLog (sprintf "Epoch %A function value: %A" i (primal v))
                    par.ReportFunction i w v
            w

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
                Util.printLog "Batch: Stochastic (minibatch of 1)"
                fun w -> par.LossFunction (d.Minibatch 1) (f w)
        let report i w v =
            if par.Verbose then
                let loss = par.LossFunction d (f (w |> Vector.map primal))
                Util.printLog (sprintf "Epoch %A training loss: %A" i (primal loss))
            par.ReportFunction i w v
        let wopt = Optimize {par with ReportFunction = report; Verbose = false} q w0
        let duration = System.DateTime.Now.Subtract(start)
        let lf = par.LossFunction d (f wopt)
        let dec = -(lf - l0)
        let perf = dec / duration.TotalSeconds
        Util.printLog "Training finished."
        Util.printLog (sprintf "Duration: %A" duration)
        Util.printLog (sprintf "Loss decrease: %A (%.2f %%)" (primal dec) (float (100 * (dec) / l0)))
        Util.printLog (sprintf "Performance (loss decrease / sec): %A\n" (primal perf))
        wopt
        