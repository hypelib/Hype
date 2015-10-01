

//#r "bin/Debug/DiffSharp.dll"
//#r "bin/Debug/Hype.dll"
#r "bin/Release/DiffSharp.dll"
#r "bin/Release/Hype.dll"

open Hype
open Hype.Neural
open DiffSharp.AD.Float32
open DiffSharp.Util

#time

let MNIST = {X = Util.LoadMNIST("C:/datasets/MNIST/train-images.idx3-ubyte", 60000)
             Y = Util.LoadMNIST("C:/datasets/MNIST/train-labels.idx1-ubyte", 60000)}


let MNISTt = {X = Util.LoadMNIST("C:/datasets/MNIST/t10k-images.idx3-ubyte", 10000)
              Y = Util.LoadMNIST("C:/datasets/MNIST/t10k-labels.idx1-ubyte", 10000)}

let MNIST' = MNIST.NormalizeX().AppendBiasRowX()
let MNISTtrain = MNIST'.[..58999]
let MNISTvalid = MNIST'.[59000..]
let MNISTtest = MNISTt.NormalizeX().AppendBiasRowX()


let n = FeedForwardLayers()
n.Add(LinearNoBiasLayer(785, 300, Initializer.InitTanh))
n.Add(ActivationLayer(tanh))
n.Add(LinearNoBiasLayer(300, 10, Initializer.InitTanh))
//n.Add(ActivationLayer(fun m -> m |> DM.mapCols softmax))

printfn "%s" (n.Print())
printfn "%s" (n.PrintFull())
printfn "%s" (n.Visualize())

//let tt = MNIST'.Filter (fun (_, y) -> (y.[0] = D 0.f) || (y.[0] = D 1.f))
//let test = n.Run MNISTtrain.[0..10].X

n.Init()
let p1 = {Params.Default with 
            Epochs = 5
            EarlyStopping = Early (400, 100)
            ValidationInterval = 10
            Method = GD
            Batch = Minibatch 200
            Loss = CrossEntropyOnLinear
            Momentum = Nesterov (D 0.9f)
            LearningRate = RMSProp (D 0.001f, D 0.9f)}
n.Train(p1, MNISTtrain, MNISTvalid)

let p2 = {Params.Default with 
            Epochs = 100
            EarlyStopping = Early (400, 100)
            ValidationInterval = 20
            Method = GD
            Batch = Full
            Loss = CrossEntropyOnLinear
            LearningRate = AdaGrad (D 0.001f)}
n.Train(p2, MNISTtrain, MNISTvalid)


type Classifier(n:Layer) =
    let n = n
    member c.Run(x:DM) = n.Run x
    member c.RunOne(x:DV) = x |> DM.ofDV x.Length |> n.Run |> DM.toDV
    member c.Classify(x:DM) = 
        let cc = Array.zeroCreate x.Cols
        c.Run(x) |> DM.iteriCols (fun i v -> cc.[i] <- DV.MaxIndex(v))
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

let a = MNISTtrain.X.[*,9]
let b = a |> DM.ofDV 28 |> DM.visualize
printfn "%s" b
