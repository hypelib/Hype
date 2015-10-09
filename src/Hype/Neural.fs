
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
    abstract member ToStringFull : unit -> string
    abstract member Visualize : unit -> string
    static member init (l:Layer) = l.Init()
    static member run (x:DM) (l:Layer) = l.Run(x)
    static member encode (l:Layer) = l.Encode()
    static member encodeLength (l:Layer) = l.EncodeLength
    static member decode (l:Layer) (w:DV) = l.Decode(w)
    static member toString (l:Layer) = l.ToString()
    static member toStringFull (l:Layer) = l.ToStringFull()
    static member visualize (l:Layer) = l.Visualize()
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
        Optimize.Train(f, w0, d, v, par) |> fst
        |> l.Decode


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


type FeedForward() =
    inherit Layer()
    let mutable (layers:Layer[]) = Array.empty
    let mutable encodelength = 0
    let update() = 
        encodelength <- layers |> Array.map Layer.encodeLength |> Array.sum
    member n.Add(l) =
        layers <- Array.append layers [|l|]
        update()
    member n.Insert(i, l) =
        let a = ResizeArray(layers)
        a.Insert(i, l)
        layers <- a.ToArray()
        update()
    member n.Remove(i) =
        let a = ResizeArray(layers)
        a.RemoveAt(i)
        layers <- a.ToArray()
        update()
    member n.Length = layers.Length
    member n.Item
        with get i = layers.[i]
    override n.Init() = layers |> Array.iter Layer.init
    override n.Run(x:DM) = Array.fold Layer.run x layers
    override n.Encode() = layers |> Array.map Layer.encode |> Array.reduce DV.append
    override n.EncodeLength = encodelength
    override n.Decode(w) =
        w |> DV.split (layers |> Array.map Layer.encodeLength)
        |> Seq.iter2 Layer.decode layers
    override n.ToString() =
        let s = System.Text.StringBuilder()
        s.Append("Hype.Neural.FeedForward\n") |> ignore
        s.Append(sprintf "   Learnable parameters: %i\n" encodelength) |> ignore
        if n.Length > 0 then
            s.Append("   ") |> ignore
            for i = 0 to layers.Length - 1 do
                s.Append("(" + (i + 1).ToString() + ") -> ") |> ignore
            s.Remove(s.Length - 4, 4) |> ignore
            s.Append("\n\n") |> ignore
            for i = 0 to layers.Length - 1 do
                s.Append("   (" + (i + 1).ToString() + "): " + layers.[i].ToString() + "\n") |> ignore
        s.ToString()
    override n.ToStringFull() =
        let s = System.Text.StringBuilder()
        s.Append(n.ToString() + "\n\n") |> ignore
        if n.Length > 0 then
            for i = 0 to layers.Length - 1 do
                s.Append("   (" + (i + 1).ToString() + "): " + layers.[i].ToStringFull() + "\n") |> ignore
        s.ToString()
    override n.Visualize() =
        let s = System.Text.StringBuilder()
        s.Append(n.ToString() + "\n\n") |> ignore
        if n.Length > 0 then
            for i = 0 to layers.Length - 1 do
                s.Append("   (" + (i + 1).ToString() + "): " + layers.[i].Visualize() + "\n") |> ignore
        s.ToString()


type LinearLayer(inputs:int, outputs:int, ?initializer:Initializer) =
    inherit Layer()
    let initializer = defaultArg initializer Initializer.InitTanh
    member val W = initializer.InitDM(outputs, inputs, inputs, outputs) with get, set
    member val b = Rnd.UniformDV(outputs) with get, set
    
    override l.Init() =
        l.W <- initializer.InitDM(l.W.Rows, l.W.Cols, l.W.Cols, l.W.Rows)
        l.b <- Rnd.UniformDV l.b.Length
    override l.Run (x:DM) = l.W * x + (l.b |> DM.createCols x.Cols)
    override l.Encode () = DV.append (DM.toDV l.W) l.b
    override l.EncodeLength = l.W.Length + l.b.Length
    override l.Decode w =
        let ww = w |> DV.split [l.W.Length; l.b.Length] |> Array.ofSeq
        l.W <- ww.[0] |> DM.ofDV l.W.Rows
        l.b <- ww.[1]
    override l.ToString() =
        "Hype.Neural.LinearLayer\n" 
            + "   Dim.:" + l.W.Cols.ToString() + " -> " + l.W.Rows.ToString() + "\n"
            + sprintf "   Init: %s\n" initializer.Name
            + sprintf "   W   : %i x %i\n" l.W.Rows l.W.Cols
            + sprintf "   b   : %i\n" l.b.Length
    override l.ToStringFull() =
        l.ToString() + "\n"
            + sprintf "   W:\n%s\n" (l.W.ToString())
            + sprintf "   b:\n%s\n" (l.b.ToString())
    override l.Visualize() =
        l.ToString() + "\n"
            + sprintf "   W:\n%s\n" (l.W.Visualize())
            + sprintf "   b:\n%s\n" (l.b.Visualize())
    member l.VisualizeWRowsAsImage(imagerows:int) =
        let s = System.Text.StringBuilder()
        s.AppendLine(l.ToString()) |> ignore
        for i = 0 to l.W.Rows - 1 do
            s.AppendLine(sprintf "   W row %i/%i:" (i + 1) l.W.Rows) |> ignore
            s.AppendLine(l.W.[i, *] |> DV.visualizeAsDM imagerows) |> ignore
        s.AppendLine(sprintf "   b:\n%s" (l.b.Visualize())) |> ignore
        l.ToString() + "\n"
            + s.ToString()
    member l.VisualizeWRowsAsImageGrid(imagerows:int) =
        l.ToString() + "\n"
            + sprintf "   W's rows %s\n" (Util.VisualizeDMRowsAsImageGrid(l.W, imagerows))
            + sprintf "   b:\n%s\n" (l.b.Visualize())


type LinearNoBiasLayer(inputs:int, outputs:int, ?initializer:Initializer) =
    inherit Layer()
    let initializer = defaultArg initializer Initializer.InitTanh
    member val W = initializer.InitDM(outputs, inputs, inputs, outputs) with get, set

    override l.Init() = l.W <- initializer.InitDM(l.W.Rows, l.W.Cols, l.W.Cols, l.W.Rows)
    override l.Run (x:DM) = l.W * x
    override l.Encode () = l.W |> DM.toDV
    override l.EncodeLength = l.W.Length
    override l.Decode w = l.W <- w |> DM.ofDV l.W.Rows
    override l.ToString() =
        "Hype.Neural.LinearNoBiasLayer\n" 
            + "   Dim.:" + l.W.Cols.ToString() + " -> " + l.W.Rows.ToString() + "\n"
            + sprintf "   Init: %s\n" initializer.Name
            + sprintf "   W   : %i x %i\n" l.W.Rows l.W.Cols
    override l.ToStringFull() =
        l.ToString() + "\n"
            + sprintf "   W:\n%s\n" (l.W.ToString())
    override l.Visualize() =
        l.ToString() + "\n"
            + sprintf "   W:\n%s\n" (l.W.Visualize())
    member l.VisualizeWRowsAsImageGrid(imagerows:int) =
        l.ToString() + "\n"
            + sprintf "   W's rows %s" (Util.VisualizeDMRowsAsImageGrid(l.W, imagerows))


type ActivationLayer(f:DM->DM) =
    inherit Layer()
    let f = f

    override l.Init() = ()
    override l.Run (x:DM) = f x
    override l.Encode () = DV.empty
    override l.EncodeLength = 0
    override l.Decode w = ()
    override l.ToString() =
        sprintf "Hype.Neural.ActivationLayer\n"
    override l.ToStringFull() = l.ToString()
    override l.Visualize() = l.ToString()
