(*** hide ***)
#r "../../src/Hype/bin/Release/DiffSharp.dll"
#r "../../src/Hype/bin/Release/Hype.dll"
#I "../../packages/R.NET.Community.1.6.4/lib/net40/"
#I "../../packages/R.NET.Community.FSharp.1.6.4/lib/net40/"
#I "../../packages/RProvider.1.1.14"
#load "RProvider.fsx"


open Hype
open Hype.Neural
open DiffSharp.AD.Float32
open DiffSharp.Util

#time

fsi.ShowDeclarationValues <- false

let ttext = ". Such tents the Patriarchs loved! O long unharmed. May all its agèd boughs o'er-canopy. The small round basin, which this jutting stone. Keeps pure from falling leaves! Long may the Spring, quietly as a sleeping infant's breath, send up cold waters to the traveller. With soft and even pulse! Nor ever cease. Yon tiny cone of sand its soundless dance. "


let tdata = Dataset(ttext.Substring(0, ttext.Length - 1),
                    ttext.Substring(1, ttext.Length - 1))

//let tdata = Dataset(" Gunes", "Gunes ")

let dim = tdata.Vocabulary.Length


let n = FeedForward()
n.Add(LSTM(dim, 100))
n.Add(Linear(100, dim))
n.Add(DM.mapCols softmax)


//let n = RecurrentSynced(dim, 100, dim)
//n.Add(Activation(DM.mapCols softmax))

n.ToString() |> printfn "%s"
n.ToStringFull() |> printfn "%s"
n.Visualize() |> printfn "%s"

let par = {Params.Default with
            Batch = Minibatch 20
            //LearningRate = LearningRate.DefaultAdaGrad
            //LearningRate = Constant (D 0.1f)
            //Momentum = Momentum.DefaultNesterov
            Loss = CrossEntropyOnSoftmax
            //Loss = CrossEntropyOnLinear
            GradientClipping = GradientClipping.DefaultNormClip
            Epochs = 100}
Layer.Train(n, tdata, par)

let generate (net:Layer) (data:Dataset) (start:string) len =
    net.Reset()
    let mutable x = start
    [for i = 0 to len - 1 do
        yield x
        let p = x |> data.EncodeOneHot |> net.Run |> DM.toDV
        x <- Rnd.Choice(data.Vocabulary, p)]
    |> List.fold (+) ""


n.Init()
n.Reset()
generate n tdata " " 30

tdata.Vocabulary
n.Run(tdata.EncodeOneHot("e")) |> DM.mapCols softmax
n.Run(tdata.X)