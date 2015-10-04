
#r "bin/Debug/DiffSharp.dll"
#r "bin/Debug/Hype.dll"
#I "../../packages/RProvider.1.1.14"
#load "RProvider.fsx"


open RProvider
open RProvider.graphics
open RProvider.grDevices

open DiffSharp.AD.Float32
open DiffSharp.Util
open Hype

let rosenbrock (x:DV) = (D 1.f - x.[0]) ** D 2.f + D 100.f * (x.[1] - x.[0] * x.[0]) ** D 2.f


let plot3d (f:DV->D) (xmin, xmax) (ymin, ymax) (theta:int) (phi:int) (w:DV) =
    let res = 20
    let xstep = ((xmax - xmin) / float res)
    let ystep = ((ymax - ymin) / float res)
    let x = [|xmin .. xstep .. xmax|]
    let y = [|ymin .. ystep .. ymax|]
    let z = Array2D.init x.Length y.Length (fun i j -> f (toDV [x.[i]; y.[j]])) |> Array2D.map (float32>>float)

    let res = 
        namedParams [
            "x", box x
            "y", box y
            "z", box z
            "theta", box theta
            "phi", box phi
            "ticktype", box "detailed"
            "xlab", box "x"
            "ylab", box "y"
            "zlab", box "z"
            "col", box "azure1"
            "shade", box 0.2]
        |> R.persp

    let pp =
        namedParams[
            "x", box 1.
            "y", box 1.
            "z", box 0.
            "pmat", box res]
        |> R.trans3d

    namedParams[
        "x", box pp
        "pch", box 19
        "col", box "black"
        ]
    |> R.points |> ignore


    let pp =
        namedParams[
            "x", box (w.[0] |> float32 |> float)
            "y", box (w.[1] |> float32 |> float)
            "z", box (f w |> float32 |> float)
            "pmat", box res]
        |> R.trans3d

    namedParams[
        "x", box pp
        "pch", box 19
        "col", box "darkorchid1"
        "type", box "o"
        ]
    |> R.points |> ignore



//plot3d rosenbrock (-1.5,1.5) (-1.5,1.5) 150 20 (vector [D -1.; D -1.])


let rosenbrock2 (x:DV) = DV.norm x

let learn (draw:bool) (l:DV) = 
    let plot (i:int) (w:DV) (v:D) =
        plot3d rosenbrock (-2.5,2.5) (-2.5,2.5) 150 25 w
    let mutable par = {Params.Default with Method = GD; LearningRate = Scheduled l; ValidationInterval = 1; Silent = true}
    if draw then par <- {par with ReportFunction = plot; ValidationInterval = 1}
    let wopt = Optimizer.Minimize(rosenbrock, toDV [-1.f; -1.f], par) |> fst
    //printfn "WOPT: %A" wopt
    let v = rosenbrock wopt
    //printfn "VVVV: %A" v
    v



let metalearn epochs metaepochs =
    //let plot _ w _ =
        //learn true w |> ignore
    let par = {Params.Default with Method = GD; Epochs = metaepochs; ValidationInterval = 1; LearningRate = AdaGrad (D 0.00001f)}
    Optimizer.Minimize((learn false), (DV.create epochs 0.001f), par)


let test = metalearn 100 400

let test2 = grad' (learn false) (DV.create 100 0.001f)

let test2b = (learn false) (DV.create 100 0.001f)
let test2c = (learn false) (test |> fst)

printfn "%s" ((test |> fst).ToMathematicaString())


let test3 = grad' rosenbrock (toDV [1.1f; 1.1f])

let test4  = grad' (learn false) (toDV [681.55f])
let test4b = grad' (learn false) (toDV [0.001f])


