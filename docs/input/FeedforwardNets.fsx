

//#r "bin/Debug/DiffSharp.dll"
//#r "bin/Debug/Hype.dll"
#r "../../src/Hype/bin/Release/DiffSharp.dll"
#r "../../src/Hype/bin/Release/Hype.dll"
fsi.ShowDeclarationValues <- false

(**
Feedforward Neural Networks
===========================

In this example, we train a softmax classifier ...

*)

open Hype
open Hype.Neural
open DiffSharp.AD.Float32
open DiffSharp.Util

#time

let MNIST = Dataset(Util.LoadMNIST("C:/datasets/MNIST/train-images.idx3-ubyte", 60000),
                    Util.LoadMNIST("C:/datasets/MNIST/train-labels.idx1-ubyte", 60000)).NormalizeX()

let MNISTtrain = MNIST.[..58999]
let MNISTvalid = MNIST.[59000..]

let MNISTtest = Dataset(Util.LoadMNIST("C:/datasets/MNIST/t10k-images.idx3-ubyte", 10000),
                        Util.LoadMNIST("C:/datasets/MNIST/t10k-labels.idx1-ubyte", 10000)).NormalizeX()


//let tt = MNIST'.Filter (fun (_, y) -> (y.[0] = D 0.f) || (y.[0] = D 1.f))
//let test = n.Run MNISTtrain.[0..10].X


let n = FeedForward()
n.Add(Linear(784, 300, Initializer.InitTanh))
n.Add(tanh)
n.Add(Linear(300, 10, Initializer.InitTanh))

n.ToString() |> printfn "%s"
n.Visualize() |> printfn "%s"



(**

### Freely implementing transformation layers

**Note: ** an important thing to note here is that the activation/transformation layers added with, for example, **n.Add(sigmoid)**, can be any matrix-to-matrix function that you can express in the language, unlike general machine learning frameworks where you are asked to select a particular layer type that has been implemented beforehand with it's (1) forward evaluation code and (2) reverse gradient code w.r.t layer inputs, and (3) reverse gradient code w.r.t any layer parameters. In such a setting, a new layer design requires you to add a new layer type to the system and implementing these components.

Here, because the system is based on nested AD, you can use any matrix-to-matrix transformation as a layer, and the forward and/or reverse AD operations of your code will be handled automatically by the underlying system. For example, you can write a layer like this: 
*)

n.Add(fun w ->
        let min = DM.Min(w)
        let range = DM.Max(w) - min
        (w - min) / range)

(** 
which will be a normalization layer, scaling the values to be between 0 and 1.
*)

let p1 = {Params.Default with 
            Epochs = 2
            EarlyStopping = Early (400, 100)
            ValidationInterval = 10
            Method = GD
            Batch = Minibatch 100
            Loss = CrossEntropyOnLinear
            Momentum = Nesterov (D 0.9f)
            LearningRate = RMSProp (D 0.001f, D 0.9f)}
n.Train(MNISTtrain, MNISTvalid, p1)

(**
### Weight initialization schemes

*)

type Initializer =
    | InitUniform of D * D
    | InitNormal of D * D
    | InitRBM of D
    | InitReLU
    | InitSigmoid
    | InitTanh
    | InitStandard
    | InitCustom of (int->int->D)
    override i.ToString() =
        match i with
        | InitUniform(min, max) -> sprintf "Uniform min=%A max=%A" min max
        | InitNormal(mu, sigma) -> sprintf "Normal mu=%A sigma=%A" mu sigma
        | InitRBM sigma -> sprintf "RBM sigma=%A" sigma
        | InitReLU -> "ReLU"
        | InitSigmoid -> "Sigmoid"
        | InitTanh -> "Tanh"
        | InitStandard -> "Standard"
        | InitCustom f -> "Custom"
    member i.InitDM(m, n) =
        let fanOut, fanIn = m, n
        match i with
        | InitUniform(min, max) -> Rnd.UniformDM(m, n, min, max)
        | InitNormal(mu, sigma) -> Rnd.NormalDM(m, n, mu, sigma)
        | InitRBM sigma -> Rnd.NormalDM(m, n, D 0.f, sigma)
        | InitReLU -> Rnd.NormalDM(m, n, D 0.f, sqrt (D 2.f / (float32 fanIn)))
        | InitSigmoid -> let r = D 4.f * sqrt (D 6.f / (fanIn + fanOut)) in Rnd.UniformDM(m, n, -r, r)
        | InitTanh -> let r = sqrt (D 6.f / (fanIn + fanOut)) in Rnd.UniformDM(m, n, -r, r)
        | InitStandard -> let r = (D 1.f) / sqrt (float32 fanIn) in Rnd.UniformDM(m, n, -r, r)
        | InitCustom f -> DM.init m n (fun _ _ -> f fanIn fanOut)
    member i.InitDM(m:DM) = i.InitDM(m.Rows, m.Cols)


type Classifier(f:DM->DM) =
    let f = f
    new(l:Layer) = Classifier(l.Run)
    member c.Run(x:DM) = f x
    member c.RunOne(x:DV) = x |> DM.ofDV x.Length |> f |> DM.toDV
    member c.Classify(x:DM) = 
        let cc = Array.zeroCreate x.Cols
        x |> f |> DM.iteriCols (fun i v -> cc.[i] <- DV.MaxIndex(v))
        cc
    member c.ClassifyOne(x:DV) =
        DV.MaxIndex(c.RunOne(x))
    member c.ClassificationError (x:DM) (y:int[]) =
        let cc = c.Classify(x)
        let incorrect = Array.map2 (fun c y -> if c = y then 0 else 1) cc y
        (float32 (incorrect |> Array.sum)) / (float32 incorrect.Length)
  

let cc = Classifier(n)

//[|"0"; "1"; "2"; "3"; "4"; "5"; "6"; "7"; "8"; "9"|]

cc.Classify(MNISTtest.X.[*,30..40]);;
MNISTtest.Y.[*, 30..40]

let targetclasses = MNISTtest.Y.[0,*] |> DV.toArray |> Array.map (float32>>int)

cc.ClassificationError MNISTtest.X targetclasses;;

let l = (n.[0] :?> Linear)
l.VisualizeWRowsAsImageGrid(28) |> printfn "%s"




// Hyperparams


let train x =
    let n = FeedForward()
    n.Add(Linear(784, 300))
    n.Add(tanh)
    n.Add(Linear(300, 10))

    let p = {Params.Default with
                LearningRate = Schedule x
                Loss = CrossEntropyOnLinear
                ValidationInterval = 1
                Silent = true
                Batch = Minibatch 100}
    let loss = n.Train(MNISTtrain, p)
    loss

//let hypertrain =
let w, _ = Optimize.Minimize(train, DV.create 10 (D 1.f), {Params.Default with Epochs = 5; ValidationInterval = 1; LearningRate = Constant (D 0.001f)})

w