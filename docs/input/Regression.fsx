(*** hide ***)
#r "../../packages/Google.DataTable.Net.Wrapper.3.1.2.0/lib/Google.DataTable.Net.Wrapper.dll"
#r "../../packages/Newtonsoft.Json.7.0.1/lib/net45/Newtonsoft.Json.dll"
#r "../../packages/XPlot.GoogleCharts.1.2.1/lib/net45/XPlot.GoogleCharts.dll"
#r "../../packages/XPlot.GoogleCharts.WPF.1.2.1/lib/net45/XPlot.GoogleCharts.WPF.dll"
#r "../../src/Hype/bin/Release/DiffSharp.dll"
#r "../../src/Hype/bin/Release/Hype.dll"
fsi.ShowDeclarationValues <- false

(**
Regression
==========

*)

open Hype
open DiffSharp.AD.Float32



(**

Training a logistic regression model for MNIST, a la Theano.
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

let MNISTtrain01 = MNISTtrain.Filter(fun (x, y) -> y.[0] <= D 1.f)
let MNISTvalid01 = MNISTvalid.Filter(fun (x, y) -> y.[0] <= D 1.f)
let MNISTtest01 = MNISTtest.Filter(fun (x, y) -> y.[0] <= D 1.f)


let model = Neural.FeedForward()
model.Add(Linear(28 * 28, 1))
model.Add(sigmoid)

let p = {Params.Default with Epochs = 2; Batch = Minibatch 100}
model.Train(MNISTtrain01, p)

Loss.Quadratic.Func MNISTtest01 model.Run


let l = (model.[0] :?> Linear)
l.VisualizeWRowsAsImageGrid(28) |> printfn "%s"


  

let cc = LogisticClassifier(model)

//[|"0"; "1"; "2"; "3"; "4"; "5"; "6"; "7"; "8"; "9"|]

cc.Classify(MNISTtest01.X.[*,0..10]);;
MNISTtest01.Y.[*, 0..10]

let targetclasses = MNISTtest01.Y.[0,*] |> DV.toArray |> Array.map (float32>>int)

cc.ClassificationError(MNISTtest01.X, targetclasses);;

cc.ClassificationError(MNISTtest01);;

cc.Classify(MNISTtest01.X.[*,0]);;
MNISTtest01.X.[*,0] |> DV.visualizeAsDM 28 |> printfn "%s"