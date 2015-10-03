
namespace Hype.Neural


open Hype
open DiffSharp.AD.Float32

[<AbstractClass>]
type Layer() =
    abstract member Init : unit -> unit
    abstract member Run : DM -> DM
    abstract member Encode : unit -> DV
    abstract member EncodeLength : int
    abstract member Decode : DV -> unit
    abstract member Print : unit -> string
    abstract member PrintFull : unit -> string
    abstract member Visualize : unit -> string
    static member Train (l:Layer, d:Dataset) = Layer.Train(l, d, Dataset.empty, Params.Default)
    static member Train (l:Layer, d:Dataset, par:Params) = Layer.Train(l, d, Dataset.empty, par)
    static member Train (l:Layer, d:Dataset, v:Dataset) = Layer.Train(l, d, v, Params.Default)
    static member Train (l:Layer, d:Dataset, v:Dataset, par:Params) =
        let f =
            fun w x ->
                l.Decode w
                l.Run x
        let w0 = l.Encode()
//        try
//            grad (fun w -> Loss.L1Loss.FuncDM(d) (f w)) w0 |> ignore
//        with
//            | _ -> failwith "Input/output dimensions mismatch between dataset and the layer."
        Optimizer.Train(f, w0, d, v, par) |> fst
        |> l.Decode

[<AutoOpen>]
module LayerOps =
    let inline initLayer (l:Layer) = l.Init()
    let inline runLayer x (l:Layer) = l.Run x
    let inline encodeLayer (l:Layer) = l.Encode()
    let inline encodeLength (l:Layer) = l.EncodeLength
    let inline decodeLayer (l:Layer) w = l.Decode w

type FeedForwardLayers() =
    inherit Layer()
    let mutable (layers:Layer[]) = Array.empty
    let mutable encodelength = 0
    member n.Add(l:Layer) =
        layers <- Array.append layers [|l|]
        encodelength <- layers |> Array.map encodeLength |> Array.sum
    member n.Length = layers.Length

    override n.Init() = layers |> Array.iter initLayer
    override n.Run(x:DM) = Array.fold runLayer x layers
    override n.Encode() = layers |> Array.map encodeLayer |> Array.reduce DV.append
    override n.EncodeLength = encodelength
    override n.Decode(w) =
        w |> DV.split (layers |> Array.map encodeLength)
        |> Seq.iter2 decodeLayer layers
    override n.Print() =
        let s = System.Text.StringBuilder()
        s.AppendLine("Feedforward layers") |> ignore
        s.AppendLine(sprintf "Learnable parameters: %i" encodelength) |> ignore
        for i = 0 to layers.Length - 1 do
            s.AppendLine((i + 1).ToString() + ": " + layers.[i].Print()) |> ignore
        s.ToString()
    override n.PrintFull() =
        let s = System.Text.StringBuilder()
        s.AppendLine("Feedforward layers") |> ignore
        s.AppendLine(sprintf "Learnable parameters: %i" encodelength) |> ignore
        for i = 0 to layers.Length - 1 do
            s.AppendLine((i + 1).ToString() + ": " + layers.[i].PrintFull()) |> ignore
        s.ToString()
    override n.Visualize() =
        let s = System.Text.StringBuilder()
        s.AppendLine("Feedforward layers") |> ignore
        s.AppendLine(sprintf "Learnable parameters: %i" encodelength) |> ignore
        for i = 0 to layers.Length - 1 do
            s.AppendLine((i + 1).ToString() + ": " + layers.[i].Visualize()) |> ignore
        s.ToString()

type Initializer =
    | InitUniform of D * D
    | InitNormal of D * D
    | InitRBM of D
    | InitReLU
    | InitSigmoid
    | InitTanh
    | InitCustom of (int->int->D)
    member i.Name =
        match i with
        | InitUniform(min, max) -> sprintf "Uniform min=%A max=%A" min max
        | InitNormal(mu, sigma) -> sprintf "Normal mu=%A sigma=%A" mu sigma
        | InitRBM sigma -> sprintf "RBM sigma=%A" sigma
        | InitReLU -> "ReLU"
        | InitSigmoid -> "Sigmoid"
        | InitTanh -> "Tanh"
        | InitCustom f -> "Custom"
    member i.InitDV(n, fanIn:int, fanOut:int) =
        match i with
        | InitUniform(min, max) -> Rnd.UniformDV(n, min, max)
        | InitNormal(mu, sigma) -> Rnd.NormalDV(n, mu, sigma)
        | InitRBM sigma -> Rnd.NormalDV(n, D 0.f, sigma)
        | InitReLU -> Rnd.NormalDV(n, D 0.f, sqrt (D 2.f / fanIn))
        | InitSigmoid ->
            let r = D 4.f * sqrt (D 6.f / (fanIn + fanOut))
            Rnd.UniformDV(n, -r, r)
        | InitTanh ->
            let r = sqrt (D 6.f / (fanIn + fanOut))
            Rnd.UniformDV(n, -r, r)
        | InitCustom f -> DV.init n (fun _ -> f fanIn fanOut)
    member i.InitDM(m, n, fanIn:int, fanOut:int) =
        match i with
        | InitUniform(min, max) -> Rnd.UniformDM(m, n, min, max)
        | InitNormal(mu, sigma) -> Rnd.NormalDM(m, n, mu, sigma)
        | InitRBM sigma -> Rnd.NormalDM(m, n, D 0.f, sigma)
        | InitReLU -> Rnd.NormalDM(m, n, D 0.f, sqrt (D 2.f / (float32 fanIn)))
        | InitSigmoid ->
            let r = D 4.f * sqrt (D 6.f / (fanIn + fanOut))
            Rnd.UniformDM(m, n, -r, r)
        | InitTanh ->
            let r = sqrt (D 6.f / (fanIn + fanOut))
            Rnd.UniformDM(m, n, -r, r)
        | InitCustom f -> DM.init m n (fun _ _ -> f fanIn fanOut)



type LinearLayer(inputs:int, outputs:int, ?initializer:Initializer) =
    inherit Layer()
    let initializer = defaultArg initializer Initializer.InitTanh
    let mutable W = initializer.InitDM(outputs, inputs, inputs, outputs)
    let mutable b = DV.zeroCreate outputs

    override l.Init() =
        W <- initializer.InitDM(W.Rows, W.Cols, W.Cols, W.Rows)
        b <- DV.zeroCreate b.Length
    override l.Run (x:DM) = W * x + (b |> Array.create x.Cols |> DM.ofCols)
    override l.Encode () = DV.append (DM.toDV W) b
    override l.EncodeLength = W.Length + b.Length
    override l.Decode w =
        let ww = w |> DV.split [W.Length; b.Length] |> Array.ofSeq
        W <- ww.[0] |> DM.ofDV W.Rows
        b <- ww.[1]
    override l.Print() =
        "LinearLayer\n   " 
            + W.Cols.ToString() + " -> " + W.Rows.ToString() + "\n   Init: " + initializer.Name
    override l.PrintFull() =
        l.Print() + "\n"
            + sprintf "   W:\n%s\n" (W.ToString())
            + sprintf "   b:\n%s\n" (b.ToString())
    override l.Visualize() =
        l.Print() + "\n"
            + sprintf "   W:\n%s\n" (W.Visualize())
            + sprintf "   b:\n%s\n" (b.Visualize())

type LinearNoBiasLayer(inputs:int, outputs:int, ?initializer:Initializer) =
    inherit Layer()
    let initializer = defaultArg initializer Initializer.InitTanh
    let mutable W = initializer.InitDM(outputs, inputs, inputs, outputs)

    override l.Init() = W <- initializer.InitDM(W.Rows, W.Cols, W.Cols, W.Rows)
    override l.Run (x:DM) = W * x
    override l.Encode () = W |> DM.toDV
    override l.EncodeLength = W.Length
    override l.Decode w = W <- w |> DM.ofDV W.Rows
    override l.Print() =
        "LinearNoBiasLayer\n   " 
            + W.Cols.ToString() + " -> " + W.Rows.ToString() + "\n   Init: " + initializer.Name
    override l.PrintFull() =
        l.Print() + "\n"
            + sprintf "   W:\n%s\n" (W.ToString())
    override l.Visualize() =
        l.Print() + "\n"
            + sprintf "   W:\n%s\n" (W.Visualize())

type ActivationLayer(f:DM->DM) =
    inherit Layer()
    let f = f

    override l.Init() = ()
    override l.Run (x:DM) = f x
    override l.Encode () = DV.empty
    override l.EncodeLength = 0
    override l.Decode w = ()
    override l.Print() =
        "ActivationLayer"
    override l.PrintFull() =
        "ActivationLayer"
    override l.Visualize() =
        "ActivationLayer"
