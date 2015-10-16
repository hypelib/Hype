

//#r "bin/Debug/DiffSharp.dll"
//#r "bin/Debug/Hype.dll"
#r "../../src/Hype/bin/Release/DiffSharp.dll"
#r "../../src/Hype/bin/Release/Hype.dll"
fsi.ShowDeclarationValues <- false

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