#r "../../packages/FsAlg.0.5.13/lib/FsAlg.dll"
#r "../../packages/DiffSharp.0.6.2/lib/DiffSharp.dll"
#I "../../packages/RProvider.1.1.8"
#load "RProvider.fsx"

//#load "../../src/Hype/Hype.fs"
//#load "../../src/Hype/Optimize.fs"
//#load "../../src/Hype/Neural.fs"
//#load "../../src/Hype/Neural.MLP.fs"
#r "../../src/Hype/bin/Release/Hype.dll"
fsi.ShowDeclarationValues <- false


open RDotNet
open RProvider
open RProvider.graphics
open RProvider.grDevices

open FsAlg.Generic
open DiffSharp.AD
open DiffSharp.AD.Vector
open Hype
open Hype.Neural

let h (w:Vector<D>) (x:Vector<D>) = sigmoid (w * x)

let trainMNIST = {X = Util.LoadMNIST("C:/MNIST/train-images.idx3-ubyte", 50) |> Matrix.map D
                  Y = Util.LoadMNIST("C:/MNIST/train-labels.idx1-ubyte", 50) |> Matrix.map D}

let testMNIST = {X = Util.LoadMNIST("C:/MNIST/t10k-images.idx3-ubyte", 10) |> Matrix.map D
                 Y = Util.LoadMNIST("C:/MNIST/t10k-labels.idx1-ubyte", 10) |> Matrix.map D}


let trainMNIST01 = trainMNIST.Filter (fun (_, y) -> (y.[0] = D 0.) || (y.[0] = D 1.))

//let image = MNISTtrain.[8] |> fst |> Vector.map float |> Vector.toSeq |> Matrix.ofSeq 28 |> Matrix.toArray2D
//
//namedParams [
//    "x", box image]
//|> R.image |> ignore


let net = MLP.create([|784; 10; 1|], Activation.sigmoid, D -0.001, D 0.001)

net.Train {DefaultParams with 
            Epochs = 10
            Batch = Full
            ReportInterval = 1
            LearningRate = Constant (D 0.1)
            Momentum = Momentum (D 0.2)
            OptimizeMethod = GD} trainMNIST01



Loss.Quadratic trainMNIST01 net.Run

Loss.Quadratic testMNIST net.Run
