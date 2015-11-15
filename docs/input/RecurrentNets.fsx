(*** hide ***)
#r "../../src/Hype/bin/Release/DiffSharp.dll"
#r "../../src/Hype/bin/Release/Hype.dll"
#I "../../packages/R.NET.Community.1.6.4/lib/net40/"
#I "../../packages/R.NET.Community.FSharp.1.6.4/lib/net40/"
#I "../../packages/RProvider.1.1.14"
#load "RProvider.fsx"


open Hype
open Hype.Neural
open Hype.NLP
open DiffSharp.AD.Float32
open DiffSharp.Util


#time

fsi.ShowDeclarationValues <- true

let text = "I sing of arms and the man, he who, exiled by fate, first came from the coast of Troy to Italy, and to Lavinian shores – hurled about endlessly by land and sea, by the will of the gods, by cruel Juno’s remorseless anger, long suffering also in war, until he founded a city and brought his gods to Latium: from that the Latin people came, the lords of Alba Longa, the walls of noble Rome. Muse, tell me the cause: how was she offended in her divinity, how was she grieved, the Queen of Heaven, to drive a man, noted for virtue, to endure such dangers, to face so many trials? Can there be such anger in the minds of the gods?"

let lang = Language(text)
let text' = lang.EncodeOneHot(text)
let data = Dataset(text'.[*, 0..(text'.Cols - 2)],
                   text'.[*, 1..(text'.Cols - 1)])


let dim = lang.Length
let n = FeedForward()
n.Add(Linear(dim, 20))
n.Add(LSTM(20, 50))
n.Add(LSTM(50, 100))
n.Add(Linear(100, dim))
n.Add(DM.mapCols softmax)



n.ToString() |> printfn "%s"
n.ToStringFull() |> printfn "%s"
n.Visualize() |> printfn "%s"


for i = 0 to 1000 do
    let par = {Params.Default with
                //Batch = Minibatch 5
                //LearningRate = LearningRate.DefaultAdaGrad
                //LearningRate = Constant (D 0.1f)
                LearningRate = LearningRate.RMSProp(D 0.01f, D 0.9f)
                //LearningRate = LearningRate.Decay(D 0.1f, D 0.1f)
                //Momentum = Momentum.DefaultNesterov
                Loss = CrossEntropyOnSoftmax
                //Loss = CrossEntropyOnLinear
                //GradientClipping = GradientClipping.DefaultNormClip
                Epochs = 1
                Silent = true
                ReturnBest = false}
    let loss, _ = Layer.Train(n, data, par)
    printfn "Epoch: %*i | Loss: %O | Sample: %s" 3 i loss (lang.Sample(n.Run, "I", [|"."|], 30))


let par = {Params.Default with
            //Batch = Minibatch 5
            //LearningRate = LearningRate.DefaultAdaGrad
            //LearningRate = Constant (D 0.1f)
            LearningRate = LearningRate.RMSProp(D 0.01f, D 0.9f)
            //LearningRate = LearningRate.Decay(D 0.1f, D 0.1f)
            //Momentum = Momentum.DefaultNesterov
            Loss = CrossEntropyOnSoftmax
            //Loss = CrossEntropyOnLinear
            //GradientClipping = GradientClipping.DefaultNormClip
            Epochs = 200}
let loss, _ = Layer.Train(n, data, par)

n.Init()
n.Reset()
lang.Sample(n.Run, "want", [|"."|], 30)
