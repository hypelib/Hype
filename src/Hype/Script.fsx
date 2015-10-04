

//#r "bin/Debug/DiffSharp.dll"
//#r "bin/Debug/Hype.dll"
#r "bin/Release/DiffSharp.dll"
#r "bin/Release/Hype.dll"

open Hype
open Hype.Neural
open DiffSharp.AD.Float32
open DiffSharp.Util

#time
fsi.ShowDeclarationValues <- false

let MNIST = {X = Util.LoadMNIST("C:/datasets/MNIST/train-images.idx3-ubyte", 1000)
             Y = Util.LoadMNIST("C:/datasets/MNIST/train-labels.idx1-ubyte", 1000)}


let MNIST' = MNIST.NormalizeX()
let MNISTtrain = MNIST'.[..800]
let MNISTvalid = MNIST'.[801..]


let n = FeedForwardLayers()
n.Add(LinearLayer(784, 10, Initializer.InitTanh))

//n.Add(ActivationLayer(fun m -> m |> DM.mapCols softmax))

printfn "%s" (n.Print())
//printfn "%s" (n.PrintFull())
printfn "%s" (n.Visualize())

//let tt = MNIST'.Filter (fun (_, y) -> (y.[0] = D 0.f) || (y.[0] = D 1.f))
//let test = n.Run MNISTtrain.[0..10].X

n.Init()
let p1 = {Params.Default with 
            Epochs = 1000
            EarlyStopping = Early (400, 100)
            ValidationInterval = 10
            Method = GD
            Batch = Minibatch 100
            Loss = CrossEntropyOnLinear
            Momentum = Nesterov (D 0.9f)
            LearningRate = RMSProp (D 0.001f, D 0.9f)}
Layer.Train(n, MNISTtrain, MNISTvalid, p1)
