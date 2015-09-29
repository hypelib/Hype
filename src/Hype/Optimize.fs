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
open DiffSharp.AD.Float32
open DiffSharp.Util

     
type LearningRate =
    | Constant of D // Constant
    | Decay of D * D // 1 / t decay, a = a0 / (1 + kt). Initial value, decay rate
    | ExponentialDecay of D * D // Exponential decay, a = a0 * Exp(-kt). Initial value, decay rate
    | Scheduled of DV // Scheduled learning rate vector, its length overrides Params.Epochs
    | Backtracking of D * D * D // Backtracking line search. Initial value, c, rho
    | StrongWolfe of D * D * D // Strong Wolfe line search. lmax, c1, c2
    | AdaGrad of D // Adagrad. Initial value
    | RMSProp of D * D // RMSProp. Initial value, decay rate
    member l.Print() =
        match l with
        | Constant a -> sprintf "Constant a = %A" a
        | Decay (a0, k) -> sprintf "1/t decay a0 = %A, k = %A" a0 k
        | ExponentialDecay (a0, k) -> sprintf "Exponential decay a = %A, k = %A" a0 k
        | Scheduled a -> sprintf "Scheduled of length %A" a.Length
        | Backtracking (a0, c, r) -> sprintf "Backtracking a0 = %A, c = %A, r = %A" a0 c r
        | StrongWolfe (amax, c1, c2) -> sprintf "Strong Wolfe amax = %A, c1 = %A, c2 = %A" amax c1 c2
        | AdaGrad (a0) -> sprintf "AdaGrad a0 = %A" a0
        | RMSProp (a0, k) -> sprintf "RMSProp a0 = %A, k = %A" a0 k
    member l.Func() =
        let loopLimit = 500
        match l with
        | Constant a -> fun _ _ _ _ _ _ _ -> box a
        | Decay (a0, k) -> fun i _ _ _ _ _ _ -> box (a0 / (1.f + k * i))
        | ExponentialDecay (a0, k) -> fun i _ _ _ _ _ _ -> box (a0 * exp (-k * i))
        | Scheduled a -> fun i _ _ _ _ _ _ -> box a.[i]
        | Backtracking (a0, c, r) -> // Typical: Backtracking (D 1.f, D 0.0001f, D 0.5f)
            fun i w f v g _ p ->
                let mutable a = a0
                let mutable i = 0
                let mutable found = false
                while not found do
                    if f (w + a * p) < v + c * a * (p * g) then 
                        found <- true
                    else
                        a <- r * a
                    i <- i + 1
                    if i > loopLimit then
                        found <- true
                        Util.printLog "WARNING: Backtracking did not converge."
                box a
        | StrongWolfe (amax, c1, c2) -> // Typical: StrongWolfe (D 1.f, D 0.0001f, D 0.5f)
            fun i w f v g _ p ->
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
                        a' <- (al + ah) / D 2.f
                        let v', gg = grad' f (w + a' * p)
                        if (v' > v0 + c1 * a' * gp0) || (v' >= v'al) then
                            ah <- a'
                        else
                            let gp' = gg * p
                            if abs gp' <= -c2 * gp0 then
                                found <- true
                            elif gp' * (ah - al) >= D 0.f then
                                ah <- al
                                al <- a'
                                v'al <- v'
                        i <- i + 1
                        if i > loopLimit then
                            found <- true
                            Util.printLog "WARNING: Strong Wolfe (zoom) did not converge."
                    a'
                            
                let mutable v = v0
                let mutable v' = v0
                let mutable gp' = gp0
                let mutable a = D 0.f
                let mutable a' = Rnd.UniformD(amax)
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
                    elif gp' >= D 0.f then
                        a'' <- zoom a' a
                        found <- true
                    else
                        a <- a'
                        v <- v'
                        a' <- Rnd.UniformD(a', amax)
                        i <- i + 1
                    if i > loopLimit then
                        found <- true
                        Util.printLog "WARNING: Strong Wolfe did not converge."
                box a''
        | AdaGrad (a0) ->
            fun _ _ _ _ g (gcache:DV ref) _ ->
                gcache := !gcache + (g .* g)
                box (a0 / sqrt (!gcache + 1e-8f))
        | RMSProp (a0, k) -> 
            fun _ _ _ _ g (gcache:DV ref) _ ->
                gcache := (k * !gcache) + (1.f - k) * (g .* g)
                box (a0 / sqrt (!gcache + 1e-8f))

type Batch =
    | Full
    | Minibatch of int
    | Stochastic // Minibatch with size 1
    member b.Print() =
        match b with
        | Full -> "Full"
        | Minibatch n -> sprintf "Minibatches of %A" n
        | Stochastic -> "Stochastic (minibatch of 1)"
    member b.Func() =
        match b with
        | Full -> fun (d:Data) _ -> d
        | Minibatch n -> 
            fun d i ->
                d.[(n * i)..((n * i) + n - 1)]
        | Stochastic -> fun d i -> d.[i..i]

type Method =
    | GD
    | CG
    | CD
    | NonlinearCG
    | DaiYuanCG
    | NewtonCG
    | Newton
    member o.Print() =
        match o with
        | GD -> "Gradient descent"
        | CG -> "Conjugate gradient"
        | CD -> "Conjugate descent"
        | DaiYuanCG -> "Dai & Yuan conjugate gradient"
        | NonlinearCG -> "Nonlinear conjugate gradient"
        | NewtonCG -> "Newton conjugate gradient"
        | Newton -> "Exact Newton"

    member o.Func() =
        match o with
        | GD ->
            fun _ w f _ _ _ ->
                let v', g' = grad' f w
                let p' = -g'
                v', g', p'
        | CG -> // Hestenes and Stiefel 1952
            fun _ w f _ g p ->
                let v', g' = grad' f w
                let y = g' - g
                let b = (g' * y) / (p * y)
                let p' = -g' + b * p
                v', g', p'
        | CD -> // Fletcher 1987
            fun _ w f _ g p ->
                let v', g' = grad' f w
                let b = (DV.normSq g') / (-p * g)
                let p' = -g' + b * p
                v', g', p'
        | DaiYuanCG -> // Dai and Yuan 1999
            fun _ w f _ g p ->
                let v', g' = grad' f w
                let y = g' - g
                let b = (DV.normSq g') / (p * y)
                let p' = -g' + b * p
                v', g', p'
        | NonlinearCG -> // Fletcher and Reeves 1964
            fun _ w f _ g p ->
                let v', g' = grad' f w
                let b = (DV.normSq g') / (DV.normSq g)
                let p' = -g' + b * p
                v', g', p'
        | NewtonCG ->
            fun _ w f _ _ p ->
                let v', g' = grad' f w
                let hv = hessianv f w p
                let b = (g' * hv) / (p * hv)
                let p' = -g' + b * p
                v', g', p'
        | Newton ->
            fun _ w f _ _ _ ->
                let v', g', h' = gradhessian' f w
                let p' = -DM.solveSymmetric h' g'
                v', g', p'

type Momentum =
    | Momentum of D
    | Nesterov of D
    | NoMomentum
    member m.Print() =
        match m with
        | Momentum m -> sprintf "Constant %A" m
        | Nesterov m -> sprintf "Constant Nesterov %A" m
        | NoMomentum -> "None"
    member m.Func() =
        match m with
        | Momentum m -> fun (u:DV) (u':DV) -> (m * u) + u'
        | Nesterov m -> fun u u' -> (m * m * u) + (m + D 1.f) * u'
        | NoMomentum -> fun _ u' -> u'

type Loss =
    | L1Loss
    | L2Loss
    | CrossEntropyOnLinear
    | CrossEntropyOnSoftmax
    member l.Print() =
        match l with
        | L1Loss -> "L1"
        | L2Loss -> "L2"
        | CrossEntropyOnLinear -> "Cross entropy after linear layer"
        | CrossEntropyOnSoftmax -> "Cross entropy after softmax layer"
    member l.FuncDM() =
        match l with
        | L1Loss -> fun (d:Data) (f:DM->DM) -> ((d.Y - (f d.X)) |> DM.toCols |> Seq.sumBy DV.l1norm) / d.Length
        | L2Loss -> fun d f -> ((d.Y - (f d.X)) |> DM.toCols |> Seq.sumBy DV.l2norm) / d.Length
        | CrossEntropyOnLinear -> fun d f -> ((f d.X) |> DM.toCols |> Seq.mapi (fun i v -> (logsumexp v) - v.[int (float32 d.Y.[0, i])]) |> Seq.sum) / d.Length
        | CrossEntropyOnSoftmax -> fun d f -> -((f d.X) |> DM.toCols |> Seq.mapi (fun i v -> (DV.standardBasis v.Length (int (float32 d.Y.[0, i]))) * log v) |> Seq.sum) / d.Length

    member l.FuncDV() =
        match l with
        | L1Loss -> fun (d:Data) (f:DV->DV) -> (d.ToSeq() |> Seq.sumBy (fun (x, y) -> DV.l1norm (y - f x))) / d.Length
        | L2Loss -> fun d f -> (d.ToSeq() |> Seq.sumBy (fun (x, y) -> DV.l2norm (y - f x))) / d.Length
        | CrossEntropyOnLinear -> fun d f -> (d.ToSeq() |> Seq.sumBy (fun (x, y) ->
                                                                        let fx = f x
                                                                        (logsumexp fx) - fx.[int (float32 y.[0])])) / d.Length
        | CrossEntropyOnSoftmax -> fun d f -> -(d.ToSeq() |> Seq.sumBy (fun (x, y) ->
                                                                        let fx = f x
                                                                        (DV.standardBasis fx.Length (int (float32 y.[0]))) * log fx)) / d.Length

type Regularization =
    | L1Reg of D
    | L2Reg of D
    | NoReg
    member r.Print() =
        match r with
        | L1Reg l -> sprintf "L1 lambda = %A" l
        | L2Reg l -> sprintf "L2 lambda = %A" l
        | NoReg -> "None"
    member r.Func() =
        match r with
        | L1Reg l -> fun (w:DV) -> l * (DV.l1norm w)
        | L2Reg l -> fun w -> l * (DV.l2normSq w)

type EarlyStopping =
    | Early of int * int // Stagnation patience, overfitting patience
    | NoEarly
    member e.Print() =
        match e with
        | Early(s, o) -> sprintf "Stagnation thresh. = %A, overfit. thresh. = %A" s o
        | NoEarly -> "None"

type Params =
    {Epochs : int
     Method: Method
     LearningRate : LearningRate
     Momentum : Momentum
     Loss : Loss
     Regularization : Regularization
     Batch : Batch
     EarlyStopping : EarlyStopping
     Verbose : bool
     ValidationInterval : int}
     static member Default = {Epochs = 100
                              LearningRate = Backtracking (D 1.f, D 0.0001f, D 0.5f)
                              Momentum = NoMomentum
                              Loss = L2Loss
                              Regularization = L2Reg (D 0.0001f)
                              Method = GD
                              Batch = Minibatch 10
                              EarlyStopping = NoEarly
                              Verbose = true
                              ValidationInterval = 10}
    member p.GetEpochs() =
        match p.LearningRate with
        | Scheduled a -> a.Length
        | _ -> p.Epochs
        

type Optimize =
    static member Train (par:Params, f:DV->DM->DM, w0:DV, d:Data, ?v:Data) =
        Util.printLog "--- Training started"
        let start = System.DateTime.Now

        let b = par.Batch.Func()
        let dir = par.Method.Func()
        let lr = par.LearningRate.Func()
        let mom = par.Momentum.Func()
        let loss = par.Loss.FuncDM()
        let reg = par.Regularization.Func()
        let epochs = par.GetEpochs()

        let batches =
            match par.Batch with
            | Full -> 1
            | Minibatch n -> d.Length / n
            | Stochastic -> d.Length

        Util.printLog (sprintf "Parameters     : %A" w0.Length)
        Util.printLog (sprintf "Training data  : %i" d.Length)
        match v with
        | Some(v) ->
            Util.printLog (sprintf "Validation data: %i" v.Length)
            Util.printLog (sprintf "Valid. interval: %i" par.ValidationInterval)
        | None    -> Util.printLog (sprintf "Validation data: None")
        Util.printLog (sprintf "Epochs         : %A" epochs)
        Util.printLog (sprintf "Batches        : %s (%A per epoch)" (par.Batch.Print()) batches)
        Util.printLog (sprintf "Method         : %s" (par.Method.Print()))
        Util.printLog (sprintf "Learning rate  : %s" (par.LearningRate.Print()))
        Util.printLog (sprintf "Momentum       : %s" (par.Momentum.Print()))
        Util.printLog (sprintf "Loss           : %s" (par.Loss.Print()))
        Util.printLog (sprintf "Regularizer    : %s" (par.Regularization.Print()))
        Util.printLog (sprintf "Early stopping : %s" (par.EarlyStopping.Print()))


        let q i w = (loss (b d i) (f w)) + reg w
        let qvalid w =
            match v with
            | Some(v) -> (loss v (f w)) + reg w
            | None -> D 0.f

        // i  : epoch
        // w  : previous weights
        // w' : new weights
        // l  : previous loss
        // l' : new loss
        // g  : previous gradient
        // g' : next gradient
        // p  : previous direction
        // p' : next direction
        // u  : previous velocity
        // u' : next velocity

        let mutable epoch = 0
        let mutable batch = 0
        let mutable w = w0
        let l, g = grad' (q 0) w0
        let mutable l = l
        let mutable g = g
        let mutable p = -g
        let mutable u = DV.ZeroN g.Length
        let gcache = ref DV.Zero

        let l0 = l
        let mutable wbest = w0
        let mutable lbest = l0
        let mutable rllast= l0

        let mutable rvllast =
            match v with
            | Some(v) -> qvalid w0
            | None -> D 0.f
        let mutable rlbest = l0
        let mutable rvlbest = rvllast
        let mutable rlbestchar = " "
        let mutable rvlbestchar = " "

        let ldiffchar l = if l < D 0.f then "↓" elif l > D 0.f then "↑" else "-"

        let mutable stagnation = -par.ValidationInterval
        let mutable overfitting = 0
        let mutable vlimproved = false
        let mutable earlystop = false


        let echars = epochs.ToString().Length
        let bchars = batches.ToString().Length
        let ichars = (epochs * d.Length).ToString().Length

        while (epoch < epochs) && (not earlystop) do
            batch <- 0
            while (batch < batches) && (not earlystop) do

                let l', g', p' = dir batch w (q batch) l g p

                if l' < lbest then
                    wbest <- w
                    lbest <- l'
                    if not vlimproved then overfitting <- overfitting + 1

                if batch % par.ValidationInterval = 0 then
                    let rldiff = l' - rllast
                    if l' < rlbest then rlbest <- l'; rlbestchar <- "▼" else rlbestchar <- " "
                    rllast <- l'

                    match v with
                    | Some(_) -> 
                        let vl' = qvalid w
                        let rvldiff = vl' - rvllast
                        if vl' < rvlbest then
                            rvlbest <- vl'
                            rvlbestchar <- "▼"
                            stagnation <- 0
                            overfitting <- 0
                            vlimproved <- true
                        else 
                            rvlbestchar <- " "
                            stagnation <- stagnation + par.ValidationInterval
                            vlimproved <- false
                            match par.EarlyStopping with
                                | Early(s, o) -> 
                                    if stagnation >= s then 
                                        Util.printLog "*** EARLY STOPPING TRIGGERED: Stagnation  ***"
                                        earlystop <- true
                                    if overfitting >= o then
                                        Util.printLog "*** EARLY STOPPING TRIGGERED: Overfitting ***"
                                        earlystop <- true
                                | _ -> ()

                        if par.Verbose then
                            match par.EarlyStopping with
                            | Early(s, o) -> 
                                Util.printLog (sprintf "Ep %*i Batch %*i | Train %O [%s%s] | Valid %O [%s%s] | S:%*i O:%*i" echars epoch bchars batch l' (ldiffchar rldiff) rlbestchar vl' (ldiffchar rvldiff) rvlbestchar (s.ToString().Length) stagnation (o.ToString().Length) overfitting)
                            | _ ->
                                Util.printLog (sprintf "Ep %*i Batch %*i | Train %O [%s%s] | Valid %O [%s%s]" echars epoch bchars batch l' (ldiffchar rldiff) rlbestchar vl' (ldiffchar rvldiff) rvlbestchar)
                        rvllast <- vl'

                    | None    -> 
                        if par.Verbose then Util.printLog (sprintf "Ep %*i Batch %*i | Train %O [%s%s]" echars epoch bchars batch l' (ldiffchar rldiff) rlbestchar)

                let mutable u' = DV.Zero
                match lr batch w (q batch) l' g' gcache p' with
                | :? D as a -> u' <- a * p'  // A scalar learning rate
                | :? DV as a -> u' <- a .* p' // Vector of independent learning rates

                u' <- mom u u'

                w <- w + u'
                l <- l'
                g <- g'
                u <- u'
                batch <- batch + 1
            epoch <- epoch + 1
           

        let duration = System.DateTime.Now.Subtract(start)
        let ldec = -(lbest - l0)
        let ldecs = ldec / (float32 duration.TotalSeconds)
        let es = (float32 epoch) / (float32 duration.TotalSeconds)
        let em = (float32 epoch) / (float32 duration.TotalMinutes)
        Util.printLog (sprintf "Duration       : %A" duration)
        Util.printLog (sprintf "Loss initial   : %A" (primal l0))
        Util.printLog (sprintf "Loss final     : %A" (primal lbest))
        Util.printLog (sprintf "Loss decrease  : %A (%.2f %%)" (primal ldec) (float32 (100 * (ldec) / l0)))
        Util.printLog (sprintf "Loss decr. / s : %A" (primal ldecs))
        Util.printLog (sprintf "Epochs / s     : %A" es)
        Util.printLog (sprintf "Epochs / min   : %A" em)
        Util.printLog "--- Training finished"
        wbest
        