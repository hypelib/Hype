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
    | Constant    of D         // Constant
    | Decay       of D * D     // 1 / t decay, a = a0 / (1 + kt). Initial value, decay rate
    | ExpDecay    of D * D     // Exponential decay, a = a0 * Exp(-kt). Initial value, decay rate
    | Schedule    of DV        // Scheduled learning rate vector, its length overrides Params.Epochs
    | Backtrack   of D * D * D // Backtracking line search. Initial value, c, rho
    | StrongWolfe of D * D * D // Strong Wolfe line search. lmax, c1, c2
    | AdaGrad     of D         // Adagrad. Initial value
    | RMSProp     of D * D     // RMSProp. Initial value, decay rate
    static member DefaultConstant    = Constant (D 0.001f)
    static member DefaultDecay       = Decay (D 1.f, D 0.1f)
    static member DefaultExpDecay    = ExpDecay (D 1.f, D 0.1f)
    static member DefaultBacktrack   = Backtrack (D 1.f, D 0.0001f, D 0.5f)
    static member DefaultStrongWolfe = StrongWolfe (D 1.f, D 0.0001f, D 0.5f)
    static member DefaultAdaGrad     = AdaGrad (D 0.001f)
    static member DefaultRMSProp     = RMSProp (D 0.001f, D 0.9f)
    override l.ToString() =
        match l with
        | Constant a                 -> sprintf "Constant a = %A" a
        | Decay (a0, k)              -> sprintf "1/t decay a0 = %A, k = %A" a0 k
        | ExpDecay (a0, k)           -> sprintf "Exponential decay a = %A, k = %A" a0 k
        | Schedule a                 -> sprintf "Scheduled of length %A" a.Length
        | Backtrack (a0, c, r)       -> sprintf "Backtracking a0 = %A, c = %A, r = %A" a0 c r
        | StrongWolfe (amax, c1, c2) -> sprintf "Strong Wolfe amax = %A, c1 = %A, c2 = %A" amax c1 c2
        | AdaGrad (a0)               -> sprintf "AdaGrad a0 = %A" a0
        | RMSProp (a0, k)            -> sprintf "RMSProp a0 = %A, k = %A" a0 k
    member l.Func =
        let loopLimit = 500
        match l with
        | Constant a -> fun _ _ _ _ _ _ _ -> box a
        | Decay (a0, k) -> fun i _ _ _ _ _ _ -> box (a0 / (1.f + k * i))
        | ExpDecay (a0, k) -> fun i _ _ _ _ _ _ -> box (a0 * exp (-k * i))
        | Schedule a -> fun i _ _ _ _ _ _ -> box a.[i]
        | Backtrack (a0, c, r) ->
            fun _ w f v g _ p ->
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
                        Util.printLog "*** BACKTRACKING DID NOT CONVERGE ***"
                box a
        | StrongWolfe (amax, c1, c2) ->
            fun _ w f v g _ p ->
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
                            Util.printLog "*** STRONG WOLFE (ZOOM) DID NOT CONVERGE ***"
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
                        Util.printLog "*** STRONG WOLFE DID NOT CONVERGE ***"
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
    override b.ToString() =
        match b with
        | Full        -> "Full"
        | Minibatch n -> sprintf "Minibatches of %A" n
        | Stochastic  -> "Stochastic (minibatch of 1)"
    member b.Func =
        match b with
        | Full -> fun (d:Dataset) _ -> d
        | Minibatch n ->    fun d i -> d.[(n * i)..((n * i) + n - 1)]
        | Stochastic ->     fun d i -> d.[i..i]

type Method =
    | GD
    | CG
    | CD
    | NonlinearCG
    | DaiYuanCG
    | NewtonCG
    | Newton
    override o.ToString() =
        match o with
        | GD          -> "Gradient descent"
        | CG          -> "Conjugate gradient"
        | CD          -> "Conjugate descent"
        | DaiYuanCG   -> "Dai & Yuan conjugate gradient"
        | NonlinearCG -> "Nonlinear conjugate gradient"
        | NewtonCG    -> "Newton conjugate gradient"
        | Newton      -> "Exact Newton"
    member o.Func =
        match o with
        | GD ->
            fun w (f:DV->D) _ _ gradclip ->
                let v', g' = grad' f w
                let g' = gradclip g'
                let p' = -g'
                v', g', p'
        | CG -> // Hestenes and Stiefel 1952
            fun w f g p gradclip ->
                let v', g' = grad' f w
                let g' = gradclip g'
                let y = g' - g
                let b = (g' * y) / (p * y)
                let p' = -g' + b * p
                v', g', p'
        | CD -> // Fletcher 1987
            fun w f g p gradclip ->
                let v', g' = grad' f w
                let g' = gradclip g'
                let b = (DV.normSq g') / (-p * g)
                let p' = -g' + b * p
                v', g', p'
        | DaiYuanCG -> // Dai and Yuan 1999
            fun w f g p gradclip ->
                let v', g' = grad' f w
                let g' = gradclip g'
                let y = g' - g
                let b = (DV.normSq g') / (p * y)
                let p' = -g' + b * p
                v', g', p'
        | NonlinearCG -> // Fletcher and Reeves 1964
            fun w f g p gradclip ->
                let v', g' = grad' f w
                let g' = gradclip g'
                let b = (DV.normSq g') / (DV.normSq g)
                let p' = -g' + b * p
                v', g', p'
        | NewtonCG ->
            fun w f _ p gradclip ->
                let v', g' = grad' f w
                let g' = gradclip g'
                let hv = hessianv f w p
                let b = (g' * hv) / (p * hv)
                let p' = -g' + b * p
                v', g', p'
        | Newton ->
            fun w f _ _ gradclip ->
                let v', g', h' = gradhessian' f w
                let g' = gradclip g'
                let p' = -DM.solveSymmetric h' g'
                v', g', p'

type Momentum =
    | Momentum of D
    | Nesterov of D
    | NoMomentum
    static member DefaultMomentum = Momentum (D 0.9f)
    static member DefaultNesterov = Nesterov (D 0.9f)
    override m.ToString() =
        match m with
        | Momentum m -> sprintf "Standard %A" m
        | Nesterov m -> sprintf "Nesterov %A" m
        | NoMomentum -> "None"
    member m.Func =
        match m with
        | Momentum m -> fun (u:DV) (u':DV) -> (m * u) + u'
        | Nesterov m -> fun u u' -> (m * m * u) + (m + D 1.f) * u'
        | NoMomentum -> fun _ u' -> u'

type Loss =
    | L1Loss
    | L2Loss
    | LeastSquares
    | CrossEntropyOnLinear
    | CrossEntropyOnSoftmax
    override l.ToString() =
        match l with
        | L1Loss -> "L1 norm, least absolute deviations"
        | L2Loss -> "L2 norm"
        | LeastSquares -> "L2 norm squared, least squares"
        | CrossEntropyOnLinear -> "Cross entropy after linear layer"
        | CrossEntropyOnSoftmax -> "Cross entropy after softmax layer"
    member l.Func =
        match l with
        | L1Loss -> fun (d:Dataset) (f:DM->DM) -> ((d.Y - (f d.X)) |> DM.toCols |> Seq.sumBy DV.l1norm) / d.Length
        | L2Loss -> fun d f -> ((d.Y - (f d.X)) |> DM.toCols |> Seq.sumBy DV.l2norm) / d.Length
        | LeastSquares -> fun d f -> ((d.Y - (f d.X)) |> DM.toCols |> Seq.sumBy DV.l2normSq) / d.Length
        | CrossEntropyOnLinear -> fun d f -> ((f d.X) |> DM.mapiCols (fun i v -> toDV [(logsumexp v) - v.[d.Yi.[i]]]) |> DM.sum) / d.Length
        | CrossEntropyOnSoftmax -> fun d f -> -((DM.map2Cols (fun t (y:DV) -> toDV [t * log y]) (d.Y) (f d.X)) |> DM.sum) / d.Length


type Regularization =
    | L1Reg of D
    | L2Reg of D
    | NoReg
    static member DefaultL1Reg = L1Reg (D 0.0001f)
    static member DefaultL2Reg = L2Reg (D 0.0001f)
    override r.ToString() =
        match r with
        | L1Reg l -> sprintf "L1 lambda = %A" l
        | L2Reg l -> sprintf "L2 lambda = %A" l
        | NoReg -> "None"
    member r.Func =
        match r with
        | L1Reg l -> fun (w:DV) -> l * (DV.l1norm w)
        | L2Reg l -> fun w -> l * (DV.l2normSq w)
        | NoReg -> fun w -> D 0.f

type GradientClipping =
    | NormClip of D
    | NoClip
    static member DefaultNormClip = NormClip (D 1.f)
    override g.ToString() =
        match g with
        | NormClip threshold -> sprintf "Norm clipping threshold = %A" threshold
        | NoClip -> "None"
    member g.Func =
        match g with
        | NormClip threshold -> fun (g:DV) -> let ng = DV.norm g in if ng > threshold then (threshold / ng) * g else g
        | NoClip -> id

type EarlyStopping =
    | Early of int * int // Stagnation patience, overfitting patience
    | NoEarly
    static member DefaultEarly = Early (750, 10)
    override e.ToString() =
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
     GradientClipping : GradientClipping
     Batch : Batch
     EarlyStopping : EarlyStopping
     ImprovementThreshold : D
     Silent : bool
     ReturnBest : bool
     ValidationInterval : int
     ReportFunction : int->DV->D->unit}
     static member Default = {Epochs = 100
                              LearningRate = LearningRate.DefaultRMSProp
                              Momentum = NoMomentum
                              Loss = L2Loss
                              Regularization = Regularization.DefaultL2Reg
                              GradientClipping = NoClip
                              Method = GD
                              Batch = Full
                              EarlyStopping = NoEarly
                              ImprovementThreshold = D 0.995f
                              Silent = false
                              ReturnBest = true
                              ValidationInterval = 10
                              ReportFunction = fun _ _ _ -> ()}
        

type Optimize =
    static member Minimize (f:DV->D, w0:DV) = Optimize.Minimize(f, w0, Params.Default)
    static member Minimize (f:DV->D, w0:DV, par:Params) =
        let dir = par.Method.Func
        let lr = par.LearningRate.Func
        let gradclip = par.GradientClipping.Func
        let mom = par.Momentum.Func
        let iters = 
            match par.LearningRate with
            | Schedule a -> a.Length
            | _ -> par.Epochs

        if not par.Silent then
            Util.printLog "--- Minimization started"
            Util.printLog (sprintf "Parameters     : %A" w0.Length)
            Util.printLog (sprintf "Iterations     : %A" iters)
            Util.printLog (sprintf "Valid. interval: %i" par.ValidationInterval)
            Util.printLog (sprintf "Method         : %O" par.Method)
            Util.printLog (sprintf "Learning rate  : %O" par.LearningRate)
            Util.printLog (sprintf "Momentum       : %O" par.Momentum)
            Util.printLog (sprintf "Gradient clip. : %O" par.GradientClipping)
            Util.printLog (sprintf "Early stopping : %O" par.EarlyStopping)
            Util.printLog (sprintf "Improv. thresh.: %A" par.ImprovementThreshold)
            Util.printLog (sprintf "Return best    : %A" par.ReturnBest)

        let mutable i = 0
        let mutable w = w0
        let l, g = grad' f w0
        let mutable l = l
        let mutable l' = l
        let mutable g = g
        let mutable p = -g
        let mutable u = DV.ZeroN g.Length
        let gcache = ref DV.Zero

        let l0 = l
        let mutable wbest = w0
        let mutable lbest = l0
        let mutable repllast = l0
        let mutable replbest = l0
        let mutable replbestchar = " "

        let ldiffchar l = if l < D 0.f then "↓" elif l > D 0.f then "↑" else "-"

        let ichars = iters.ToString().Length
        
        let mutable stagnation = -par.ValidationInterval
        let mutable earlystop = false

        let start = System.DateTime.Now

        while (i < iters) && (not earlystop) do
            let l'', g', p' = dir w f g p gradclip
            l' <- l''
            if l' < par.ImprovementThreshold * lbest then
                wbest <- w
                lbest <- l'

            if i % par.ValidationInterval = 0 then
                let repldiff = l' - repllast
                if l' < par.ImprovementThreshold * replbest then
                    replbest <- l'
                    replbestchar <- "▼" 
                    stagnation <- 0
                else 
                    replbestchar <- " "
                    stagnation <- stagnation + par.ValidationInterval
                    match par.EarlyStopping with
                    | Early(s, _) ->
                        if stagnation >= s then
                            if not par.Silent then Util.printLog "*** EARLY STOPPING TRIGGERED: Stagnation ***"
                            earlystop <- true
                    | _ -> ()

                if not par.Silent then 
                    match par.EarlyStopping with
                    | Early(s, _) ->
                        Util.printLog (sprintf "%*i/%i | %O [%s%s] | Stag:%*i" ichars (i + 1) iters l' (ldiffchar repldiff) replbestchar (s.ToString().Length) stagnation)
                    | _ ->
                        Util.printLog (sprintf "%*i/%i | %O [%s%s]" ichars (i + 1) iters l' (ldiffchar repldiff) replbestchar)

                repllast <- l'
                par.ReportFunction i w l'

            let mutable u' = DV.Zero
            match lr i w f l' g' gcache p' with
            | :? D as a -> u' <- a * p'; // A scalar learning rate
            | :? DV as a -> u' <- a .* p'; // Vector of independent learning rates

            u' <- mom u u'

            w <- w + u'
            l <- l'
            g <- g'
            p <- p' // Or, p <- u'
            u <- u'
            i <- i + 1

        let l'', _, _ = dir w f g p gradclip
        l' <- l''
        if l' < par.ImprovementThreshold * lbest then
            wbest <- w
            lbest <- l'

        let duration = System.DateTime.Now.Subtract(start)

        let wfinal = if par.ReturnBest then wbest else w
        let lfinal = if par.ReturnBest then lbest else l'

        let lchg = (lfinal - l0)
        let lchgs = lchg / (float32 duration.TotalSeconds)
        let es = (float i) / (duration.TotalSeconds)
        let em = (float i) / (duration.TotalMinutes)

        if not par.Silent then
            Util.printLog (sprintf "Duration       : %A" duration)
            Util.printLog (sprintf "Value initial  : %O" (primal l0))
            Util.printLog (sprintf "Value final    : %O %s" (primal lfinal) (if par.ReturnBest then "(Best)" else "(Last)"))
            Util.printLog (sprintf "Value change   : %O (%.2f %%)" (primal lchg) (float32 (100 * lchg /l0)))
            Util.printLog (sprintf "Value chg. / s : %O" (primal lchgs))
            Util.printLog (sprintf "Iter. / s      : %A" es)
            Util.printLog (sprintf "Iter. / min    : %A" em)
            Util.printLog "--- Minimization finished"
        wfinal, lfinal

    static member Train (f:DV->DV->DV, w0:DV, d:Dataset) = Optimize.Train(f, w0, d, Dataset.empty, Params.Default)
    static member Train (f:DV->DV->DV, w0:DV, d:Dataset, par:Params) = Optimize.Train(f, w0, d, Dataset.empty, par)
    static member Train (f:DV->DV->DV, w0:DV, d:Dataset, v:Dataset) = Optimize.Train(f, w0, d, v, Params.Default)
    static member Train (f:DV->DV->DV, w0:DV, d:Dataset, v:Dataset, par:Params) = Optimize.Train (f >> DM.mapCols, w0, d, v, par)
    static member Train (f:DV->DM->DM, w0:DV, d:Dataset) = Optimize.Train(f, w0, d, Dataset.empty, Params.Default)
    static member Train (f:DV->DM->DM, w0:DV, d:Dataset, par:Params) = Optimize.Train(f, w0, d, Dataset.empty, par)
    static member Train (f:DV->DM->DM, w0:DV, d:Dataset, v:Dataset) = Optimize.Train(f, w0, d, v, Params.Default)
    static member internal Train (f:DV->DM->DM, w0:DV, d:Dataset, v:Dataset, par:Params) =
        let b = par.Batch.Func
        let dir = par.Method.Func
        let lr = par.LearningRate.Func
        let gradclip = par.GradientClipping.Func
        let mom = par.Momentum.Func
        let reg = par.Regularization.Func
        let epochs = par.Epochs
        let loss = par.Loss.Func
        let batches, batchsize =
            match par.Batch with
            | Full -> 1, d.Length
            | Minibatch n -> d.Length / n, n
            | Stochastic -> d.Length, 1
        let iters =
            match par.LearningRate with
            | Schedule a -> a.Length
            | _ -> epochs * batches

        if not par.Silent then
            Util.printLog "--- Training started"
            Util.printLog (sprintf "Parameters     : %A" w0.Length)
            Util.printLog (sprintf "Iterations     : %A" iters)
            Util.printLog (sprintf "Epochs         : %A" epochs)
            Util.printLog (sprintf "Batches        : %O (%A per epoch)" par.Batch batches)
            Util.printLog (sprintf "Training data  : %i" d.Length)
            if Dataset.isEmpty v then
                Util.printLog (sprintf "Validation data: None")
            else
                Util.printLog (sprintf "Validation data: %i" v.Length)
            Util.printLog (sprintf "Valid. interval: %i" par.ValidationInterval)
            Util.printLog (sprintf "Method         : %O" par.Method)
            Util.printLog (sprintf "Learning rate  : %O" par.LearningRate)
            Util.printLog (sprintf "Momentum       : %O" par.Momentum)
            Util.printLog (sprintf "Loss           : %O" par.Loss)
            Util.printLog (sprintf "Regularizer    : %O" par.Regularization)
            Util.printLog (sprintf "Gradient clip. : %O" par.GradientClipping)
            Util.printLog (sprintf "Early stopping : %O" par.EarlyStopping)
            Util.printLog (sprintf "Improv. thresh.: %A" par.ImprovementThreshold)
            Util.printLog (sprintf "Return best    : %A" par.ReturnBest)

        let q i w = (loss (b d i) (f w)) + reg w

        let qvalid =
            if Dataset.isEmpty v then
                fun _ -> D 0.f
            else
                fun w -> (loss v (f w)) + reg w

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
        let mutable l' = l
        let mutable g = g
        let mutable p = -g
        let mutable u = DV.ZeroN g.Length
        let gcache = ref DV.Zero

        let l0 = l
        let mutable wbest = w0
        let mutable lbest = l0
        let mutable repllast= l0
        let mutable replbest = l0
        let mutable replbestchar = " "

        let mutable repvllast = if Dataset.isEmpty v then D 0.f else qvalid w0
        let mutable repvlbest = repvllast
        let mutable repvlbestchar = " "

        let ldiffchar l = if l < D 0.f then "↓" elif l > D 0.f then "↑" else "-"

        let mutable stagnation = -par.ValidationInterval
        let mutable overfitting = 0
        let mutable validlimproved = false
        let mutable earlystop = false

        let echars = epochs.ToString().Length
        let bchars = batches.ToString().Length
        let ichars = (epochs * d.Length).ToString().Length

        let start = System.DateTime.Now

        while (epoch < epochs) && (not earlystop) do
            batch <- 0
            while (batch < batches) && (not earlystop) do

                let l'', g', p' = dir w (q batch) g p gradclip
                l' <- l''
                if l' < par.ImprovementThreshold * lbest then
                    wbest <- w
                    lbest <- l'
                    if not (Dataset.isEmpty v) then
                        if not validlimproved then 
                            overfitting <- overfitting + 1
                            match par.EarlyStopping with
                            | Early(_, o) ->
                                if overfitting >= o then 
                                    Util.printLog "*** EARLY STOPPING TRIGGERED: Overfitting ***"
                                    earlystop <- true
                            | _ -> ()

                if batch % par.ValidationInterval = 0 then
                    let repldiff = l' - repllast
                    if l' < par.ImprovementThreshold * replbest then
                        replbest <- l'
                        replbestchar <- "▼" 
                    else 
                        replbestchar <- " "
                        if Dataset.isEmpty v then
                            stagnation <- stagnation + par.ValidationInterval
                            match par.EarlyStopping with
                            | Early(s, _) ->
                                if stagnation >= s then
                                    if not par.Silent then Util.printLog "*** EARLY STOPPING TRIGGERED: Stagnation of training loss ***"
                                    earlystop <- true
                            | _ -> ()
                    repllast <- l'

                    if Dataset.isEmpty v then
                        if not par.Silent then
                            match par.EarlyStopping with
                            | Early(s, _) ->
                                Util.printLog (sprintf "%*i/%i | Batch %*i/%i | %O [%s%s] | Stag:%*i" echars (epoch + 1) epochs bchars (batch + 1) batches l' (ldiffchar repldiff) replbestchar (s.ToString().Length) stagnation)
                            | _ ->
                                Util.printLog (sprintf "%*i/%i | Batch %*i/%i | %O [%s%s]" echars (epoch + 1) epochs bchars (batch + 1) batches l' (ldiffchar repldiff) replbestchar)
                    else
                        let vl' = qvalid w
                        let repvldiff = vl' - repvllast
                        if vl' < par.ImprovementThreshold * repvlbest then
                            repvlbest <- vl'
                            repvlbestchar <- "▼"
                            validlimproved <- true
                            stagnation <- 0
                            overfitting <- 0
                        else
                            repvlbestchar <- " "
                            validlimproved <- false
                            stagnation <- stagnation + par.ValidationInterval
                            match par.EarlyStopping with
                                | Early(s, _) -> 
                                    if stagnation >= s then 
                                        if not par.Silent then Util.printLog "*** EARLY STOPPING TRIGGERED: Stagnation of validation loss ***"
                                        earlystop <- true
                                | _ -> ()

                        if not par.Silent then
                            match par.EarlyStopping with
                            | Early(s, o) -> 
                                Util.printLog (sprintf "%*i/%i | Batch %*i/%i | %O [%s%s] | Valid %O [%s%s] | Stag:%*i Ovfit:%*i" echars (epoch + 1) epochs bchars (batch + 1) batches l' (ldiffchar repldiff) replbestchar vl' (ldiffchar repvldiff) repvlbestchar (s.ToString().Length) stagnation (o.ToString().Length) overfitting)
                            | _ ->
                                Util.printLog (sprintf "%*i/%i | Batch %*i/%i | %O [%s%s] | Valid %O [%s%s]" echars (epoch + 1) epochs bchars (batch + 1) batches l' (ldiffchar repldiff) replbestchar vl' (ldiffchar repvldiff) repvlbestchar)
                        repvllast <- vl'
                        par.ReportFunction epoch w l'

                let mutable u' = DV.Zero
                match lr epoch w (q batch) l' g' gcache p' with
                | :? D as a -> u' <- a * p'  // A scalar learning rate
                | :? DV as a -> u' <- a .* p' // Vector of independent learning rates

                u' <- mom u u'

                w <- w + u'
                l <- l'
                g <- g'
                p <- p' // Or, p <- u'
                u <- u'
                batch <- batch + 1
                let iter = batches * epoch + batch
                if iter >= iters then earlystop <- true
            epoch <- epoch + 1

        let l'', _, _ = dir w (q 0) g p gradclip
        l' <- l''
        if l' < par.ImprovementThreshold * lbest then
            wbest <- w
            lbest <- l'

        let duration = System.DateTime.Now.Subtract(start)
          
        let wfinal = if par.ReturnBest then wbest else w
        let lfinal = if par.ReturnBest then lbest else l'

        let lchg = (lfinal - l0)
        let lchgs = lchg / (float32 duration.TotalSeconds)
        let es = (float epoch) / (duration.TotalSeconds)
        let em = (float epoch) / (duration.TotalMinutes)
        if not par.Silent then
            Util.printLog (sprintf "Duration       : %A" duration)
            Util.printLog (sprintf "Loss initial   : %O" (primal l0))
            Util.printLog (sprintf "Loss final     : %O %s" (primal lfinal) (if par.ReturnBest then "(Best)" else "(Last)"))
            Util.printLog (sprintf "Loss change    : %O (%.2f %%)" (primal lchg) (float32 (100 * (lchg) / l0)))
            Util.printLog (sprintf "Loss chg. / s  : %O" (primal lchgs))
            Util.printLog (sprintf "Epochs / s     : %A" es)
            Util.printLog (sprintf "Epochs / min   : %A" em)
            Util.printLog "--- Training finished"
        wfinal, lfinal
        