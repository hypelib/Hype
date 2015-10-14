
namespace Hype.Neural


open Hype
open DiffSharp.AD.Float32
open DiffSharp.Util


[<AbstractClass>]
type Layer() =
    abstract member Init : unit -> unit
    abstract member Reset : unit -> unit
    abstract member Run : DM -> DM
    abstract member Encode : unit -> DV
    abstract member EncodeLength : int
    abstract member Decode : DV -> unit
    abstract member ToStringFull : unit -> string
    abstract member Visualize : unit -> string
    static member init (l:Layer) = l.Init()
    static member reset (l:Layer) = l.Reset()
    static member run x (l:Layer) = l.Run(x)
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
        let w, loss = Optimize.Train(f, w0, d, v, par)
        w |> l.Decode
        loss


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
    override n.Reset() = layers |> Array.iter Layer.reset
    override n.Run(x:DM) = Array.fold Layer.run x layers
    override n.Encode() = layers |> Array.map Layer.encode |> Array.reduce DV.append
    override n.EncodeLength = encodelength
    override n.Decode(w) =
        w |> DV.split (layers |> Array.map Layer.encodeLength)
        |> Seq.iter2 Layer.decode layers
    override n.ToString() =
        let s = System.Text.StringBuilder()
        if n.Length > 0 then
            s.Append("   ") |> ignore
            for i = 0 to layers.Length - 1 do
                s.Append("(" + i.ToString() + ") -> ") |> ignore
            s.Remove(s.Length - 4, 4) |> ignore
            s.Append("\n\n") |> ignore
            for i = 0 to layers.Length - 1 do
                s.Append("   (" + i.ToString() + "): " + layers.[i].ToString() + "\n\n") |> ignore

        "Hype.Neural.FeedForward\n"
            + sprintf "   Learnable parameters: %i\n" encodelength
            + s.ToString()
    override n.ToStringFull() =
        let s = System.Text.StringBuilder()
        if n.Length > 0 then
            s.Append("   ") |> ignore
            for i = 0 to layers.Length - 1 do
                s.Append("(" + i.ToString() + ") -> ") |> ignore
            s.Remove(s.Length - 4, 4) |> ignore
            s.Append("\n\n") |> ignore
            for i = 0 to layers.Length - 1 do
                s.Append("   (" + i.ToString() + "): " + layers.[i].ToStringFull() + "\n\n") |> ignore

        "Hype.Neural.FeedForward\n"
            + sprintf "   Learnable parameters: %i\n" encodelength
            + s.ToString()
    override n.Visualize() =
        let s = System.Text.StringBuilder()
        if n.Length > 0 then
            s.Append("   ") |> ignore
            for i = 0 to layers.Length - 1 do
                s.Append("(" + i.ToString() + ") -> ") |> ignore
            s.Remove(s.Length - 4, 4) |> ignore
            s.Append("\n\n") |> ignore
            for i = 0 to layers.Length - 1 do
                s.Append("   (" + i.ToString() + "): " + layers.[i].Visualize() + "\n\n") |> ignore

        "Hype.Neural.FeedForward\n"
            + sprintf "   Learnable parameters: %i\n" encodelength
            + s.ToString()


type Linear(inputs:int, outputs:int, initializer:Initializer) =
    inherit Layer()
    new(inputs, outputs) = Linear(inputs, outputs, Initializer.InitStandard)
    
    member val W = initializer.InitDM(outputs, inputs) with get, set
    member val b = DV.zeroCreate outputs with get, set
    
    override l.Init() =
        l.W <- initializer.InitDM(l.W)
        l.b <- DV.zeroCreate l.b.Length
    override l.Reset() = ()
    override l.Run (x:DM) = (l.W * x) + l.b
    override l.Encode () = DV.append (DM.toDV l.W) l.b
    override l.EncodeLength = l.W.Length + l.b.Length
    override l.Decode w =
        let ww = w |> DV.split [l.W.Length; l.b.Length] |> Array.ofSeq
        l.W <- ww.[0] |> DM.ofDV l.W.Rows
        l.b <- ww.[1]
    override l.ToString() =
        "Hype.Neural.Linear\n" 
            + "   " + l.W.Cols.ToString() + " -> " + l.W.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   W   : %i x %i\n" l.W.Rows l.W.Cols
            + sprintf "   b   : %i" l.b.Length
    override l.ToStringFull() =
        "Hype.Neural.Linear\n" 
            + "   " + l.W.Cols.ToString() + " -> " + l.W.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   W:\n%O\n" l.W
            + sprintf "   b:\n%O" l.b
    override l.Visualize() =
        "Hype.Neural.Linear\n" 
            + "   " + l.W.Cols.ToString() + " -> " + l.W.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   W:\n%s\n" (l.W.Visualize())
            + sprintf "   b:\n%s" (l.b.Visualize())
    member l.VisualizeWRowsAsImageGrid(imagerows:int) =
        "Hype.Neural.Linear\n" 
            + "   " + l.W.Cols.ToString() + " -> " + l.W.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   W's rows %s\n" (Util.VisualizeDMRowsAsImageGrid(l.W, imagerows))
            + sprintf "   b:\n%s" (l.b.Visualize())


type LinearNoBias(inputs:int, outputs:int, initializer:Initializer) =
    inherit Layer()
    new(inputs, outputs) = LinearNoBias(inputs, outputs, Initializer.InitStandard)

    member val W = initializer.InitDM(outputs, inputs) with get, set

    override l.Init() = l.W <- initializer.InitDM(l.W)
    override l.Reset() = ()
    override l.Run (x:DM) = l.W * x
    override l.Encode () = l.W |> DM.toDV
    override l.EncodeLength = l.W.Length
    override l.Decode w = l.W <- w |> DM.ofDV l.W.Rows
    override l.ToString() =
        "Hype.Neural.LinearNoBias\n" 
            + "   " + l.W.Cols.ToString() + " -> " + l.W.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   W   : %i x %i" l.W.Rows l.W.Cols
    override l.ToStringFull() =
        "Hype.Neural.LinearNoBias\n" 
            + "   " + l.W.Cols.ToString() + " -> " + l.W.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   W:\n%O" l.W
    override l.Visualize() =
        "Hype.Neural.LinearNoBias\n" 
            + "   " + l.W.Cols.ToString() + " -> " + l.W.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   W:\n%s" (l.W.Visualize())
    member l.VisualizeWRowsAsImageGrid(imagerows:int) =
        "Hype.Neural.LinearNoBias\n" 
            + "   " + l.W.Cols.ToString() + " -> " + l.W.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   W's rows %s" (Util.VisualizeDMRowsAsImageGrid(l.W, imagerows))


type Activation(f:DM->DM) =
    inherit Layer()
    let f = f

    override l.Init () = ()
    override l.Reset () = ()
    override l.Run (x:DM) = f x
    override l.Encode () = DV.empty
    override l.EncodeLength = 0
    override l.Decode w = ()
    override l.ToString() =
        sprintf "Hype.Neural.Activation"
    override l.ToStringFull() = l.ToString()
    override l.Visualize() = l.ToString()

type OneHot(dim:int) =
    inherit Layer()

    override l.Init () = ()
    override l.Reset () = ()
    override l.Run (x:DM) = x |> DM.mapCols (fun v -> DV.standardBasis dim (int (float32 v.[0])))
    override l.Encode () = DV.empty
    override l.EncodeLength = 0
    override l.Decode w = ()
    override l.ToString() =
        sprintf "Hype.Neural.OneHot\n"
            + sprintf "   1 -> %i" dim
    override l.ToStringFull() = l.ToString()
    override l.Visualize() = l.ToString()

type Recurrent(inputs:int, hiddenunits:int, outputs:int, activation:DV->DV, initializer:Initializer) =
    inherit Layer()
    new(inputs, hiddenunits, outputs) = Recurrent(inputs, hiddenunits, outputs, tanh, Initializer.InitTanh)
    new(inputs, hiddenunits, outputs, activation) = Recurrent(inputs, hiddenunits, outputs, activation, Initializer.InitTanh)

    member val Act = activation with get
    member val Whh = initializer.InitDM(hiddenunits, hiddenunits) with get, set
    member val Wxh = initializer.InitDM(hiddenunits, inputs) with get, set
    member val Why = initializer.InitDM(outputs, hiddenunits) with get, set
    member val bh = DV.zeroCreate hiddenunits with get, set
    member val by = DV.zeroCreate outputs with get, set
    member val h = DV.zeroCreate hiddenunits with get, set

    override l.Init() = 
        l.Whh <- initializer.InitDM(l.Whh)
        l.Wxh <- initializer.InitDM(l.Wxh)
        l.Why <- initializer.InitDM(l.Why)
        l.bh <- DV.zeroCreate hiddenunits
        l.by <- DV.zeroCreate outputs
        l.h <- DV.zeroCreate hiddenunits
    override l.Reset() = l.h <- DV.zeroCreate hiddenunits
    override l.Run (x:DM) = 
        let y = x |> DM.mapCols (fun x -> 
                                    l.h <- l.Act ((l.Whh * l.h) + (l.Wxh * x) + l.bh)
                                    (l.Why * l.h) + l.by)
        l.h <- primalDeep l.h
        y
    override l.Encode () = [l.Whh; l.Wxh; l.Why] |> List.map DM.toDV |> List.append [l.bh; l.by] |> Seq.fold DV.append DV.Zero
    override l.EncodeLength = l.Whh.Length + l.Wxh.Length + l.Why.Length + l.bh.Length + l.by.Length
    override l.Decode w =
        let ww = w |> DV.split [l.Whh.Length; l.Wxh.Length; l.Why.Length; l.bh.Length; l.by.Length] |> Array.ofSeq
        l.Whh <- ww.[0] |> DM.ofDV l.Whh.Rows
        l.Wxh <- ww.[1] |> DM.ofDV l.Wxh.Rows
        l.Why <- ww.[2] |> DM.ofDV l.Why.Rows
        l.bh <- ww.[3]
        l.by <- ww.[4]
        l.h <- DV.zeroCreate hiddenunits
    override l.ToString() =
        "Hype.Neural.Recurrent\n"
            + "   " + l.Wxh.Cols.ToString() + " -> " + l.Whh.Rows.ToString() + " -> " + l.Why.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   Whh : %i x %i\n" l.Whh.Rows l.Whh.Cols
            + sprintf "   Wxh : %i x %i\n" l.Wxh.Rows l.Wxh.Cols
            + sprintf "   Why : %i x %i\n" l.Why.Rows l.Why.Cols
            + sprintf "   bh  : %i\n" l.bh.Length
            + sprintf "   by  : %i" l.by.Length
    override l.ToStringFull() =
        "Hype.Neural.Recurrent\n"
            + "   " + l.Wxh.Cols.ToString() + " -> " + l.Whh.Rows.ToString() + " -> " + l.Why.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   Whh:\n%O\n" l.Whh
            + sprintf "   Wxh:\n%O\n" l.Wxh
            + sprintf "   Why:\n%O\n" l.Why
            + sprintf "   bh:\n%O\n" l.bh
            + sprintf "   by:\n%O" l.by
    override l.Visualize() =
        "Hype.Neural.Recurrent\n"
            + "   " + l.Wxh.Cols.ToString() + " -> " + l.Whh.Rows.ToString() + " -> " + l.Why.Rows.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   Whh:\n%s\n" (l.Whh.Visualize())
            + sprintf "   Wxh:\n%s\n" (l.Wxh.Visualize())
            + sprintf "   Why:\n%s\n" (l.Why.Visualize())
            + sprintf "   bh:\n%s\n" (l.bh.Visualize())
            + sprintf "   by:\n%s" (l.by.Visualize())

type LSTM(inputs:int, memcells:int) =
    inherit Layer()
    let initializer = Initializer.InitTanh

    member val Wxi = initializer.InitDM(memcells, inputs) with get, set
    member val Whi = initializer.InitDM(memcells, memcells) with get, set
    member val Wxc = initializer.InitDM(memcells, inputs) with get, set
    member val Whc = initializer.InitDM(memcells, memcells) with get, set
    member val Wxf = initializer.InitDM(memcells, inputs) with get, set
    member val Whf = initializer.InitDM(memcells, memcells) with get, set
    member val Wxo = initializer.InitDM(memcells, inputs) with get, set
    member val Who = initializer.InitDM(memcells, memcells) with get, set
    member val bi = DV.zeroCreate memcells with get, set
    member val bc = DV.zeroCreate memcells with get, set
    member val bf = DV.zeroCreate memcells with get, set
    member val bo = DV.zeroCreate memcells with get, set
    member val c = DV.zeroCreate memcells with get, set
    member val h = DV.zeroCreate memcells with get, set

    override l.Init() =
        l.Wxi <- initializer.InitDM(l.Wxi)
        l.Whi <- initializer.InitDM(l.Whi)
        l.Wxc <- initializer.InitDM(l.Wxc)
        l.Whc <- initializer.InitDM(l.Whc)
        l.Wxf <- initializer.InitDM(l.Wxf)
        l.Whf <- initializer.InitDM(l.Whf)
        l.Wxo <- initializer.InitDM(l.Wxo)
        l.Who <- initializer.InitDM(l.Who)
        l.bi <- DV.zeroCreate memcells
        l.bc <- DV.zeroCreate memcells
        l.bf <- DV.zeroCreate memcells
        l.bo <- DV.zeroCreate memcells
        l.c <- DV.zeroCreate memcells
        l.h <- DV.zeroCreate memcells
    override l.Reset() =
        l.c <- DV.zeroCreate memcells
        l.h <- DV.zeroCreate memcells
    override l.Run (x:DM) =
        let y = x |> DM.mapCols (fun x ->
                                    let i = sigmoid((l.Wxi * x) + (l.Whi * l.h) + l.bi)
                                    let c' = tanh((l.Wxc * x) + (l.Whc * l.h) + l.bc)
                                    let f = sigmoid((l.Wxf * x) + (l.Whf * l.h) + l.bf)
                                    l.c <- (i .* c') + (f .* l.c)
                                    let o = sigmoid((l.Wxo * x) + (l.Who * l.h) + l.bo)
                                    l.h <- o .* tanh l.c
                                    l.h)
        l.h <- primalDeep l.h
        l.c <- primalDeep l.c
        y
    override l.Encode() = [l.Wxi; l.Whi; l.Wxc; l.Whc; l.Wxf; l.Whf; l.Wxo; l.Who] |> List.map DM.toDV |> List.append [l.bi; l.bc; l.bf; l.bo] |> Seq.fold DV.append DV.Zero
    override l.EncodeLength = l.Wxi.Length + l.Whi.Length + l.Wxc.Length + l.Whc.Length + l.Wxf.Length + l.Whf.Length + l.Wxo.Length + l.Who.Length + l.bi.Length + l.bc.Length + l.bf.Length + l.bo.Length
    override l.Decode w =
        let ww = w |> DV.split [l.Wxi.Length; l.Whi.Length; l.Wxc.Length; l.Whc.Length; l.Wxf.Length; l.Whf.Length; l.Wxo.Length; l.Who.Length; l.bi.Length; l.bc.Length; l.bf.Length; l.bo.Length] |> Array.ofSeq
        l.Wxi <- ww.[0] |> DM.ofDV l.Wxi.Rows
        l.Whi <- ww.[1] |> DM.ofDV l.Whi.Rows
        l.Wxc <- ww.[2] |> DM.ofDV l.Wxc.Rows
        l.Whc <- ww.[3] |> DM.ofDV l.Whc.Rows
        l.Wxf <- ww.[4] |> DM.ofDV l.Wxf.Rows
        l.Whf <- ww.[5] |> DM.ofDV l.Whf.Rows
        l.Wxo <- ww.[6] |> DM.ofDV l.Wxo.Rows
        l.Who <- ww.[7] |> DM.ofDV l.Who.Rows
        l.bi <- ww.[8]
        l.bc <- ww.[9]
        l.bf <- ww.[10]
        l.bo <- ww.[11]
        l.c <- DV.zeroCreate memcells
        l.h <- DV.zeroCreate memcells
    override l.ToString() =
        "Hype.Neural.LSTM\n"
            + "   " + inputs.ToString() + " -> " + memcells.ToString() + " -> " + memcells.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   Wxi : %i x %i\n" l.Wxi.Rows l.Wxi.Cols
            + sprintf "   Whi : %i x %i\n" l.Whi.Rows l.Whi.Cols
            + sprintf "   Wxc : %i x %i\n" l.Wxc.Rows l.Wxc.Cols
            + sprintf "   Whc : %i x %i\n" l.Whc.Rows l.Whc.Cols
            + sprintf "   Wxf : %i x %i\n" l.Wxf.Rows l.Wxf.Cols
            + sprintf "   Whf : %i x %i\n" l.Whf.Rows l.Whf.Cols
            + sprintf "   Wxo : %i x %i\n" l.Wxo.Rows l.Wxo.Cols
            + sprintf "   Who : %i x %i\n" l.Who.Rows l.Who.Cols
            + sprintf "   bi  : %i\n" l.bi.Length
            + sprintf "   bc  : %i\n" l.bc.Length
            + sprintf "   bf  : %i\n" l.bf.Length
            + sprintf "   bo  : %i" l.bo.Length
    override l.ToStringFull() =
        "Hype.Neural.LSTM\n"
            + "   " + inputs.ToString() + " -> " + memcells.ToString() + " -> " + memcells.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   Wxi:\n%O\n" l.Wxi
            + sprintf "   Whi:\n%O\n" l.Whi
            + sprintf "   Wxc:\n%O\n" l.Wxc
            + sprintf "   Whc:\n%O\n" l.Whc
            + sprintf "   Wxf:\n%O\n" l.Wxf
            + sprintf "   Whf:\n%O\n" l.Whf
            + sprintf "   Wxo:\n%O\n" l.Wxo
            + sprintf "   Who:\n%O\n" l.Who
            + sprintf "   bi:\n%O\n" l.bi
            + sprintf "   bc:\n%O\n" l.bc
            + sprintf "   bf:\n%O\n" l.bf
            + sprintf "   bo:\n%O" l.bo
    override l.Visualize() =
        "Hype.Neural.LSTM\n"
            + "   " + inputs.ToString() + " -> " + memcells.ToString() + " -> " + memcells.ToString() + "\n"
            + sprintf "   Learnable parameters: %i\n" l.EncodeLength
            + sprintf "   Init: %O\n" initializer
            + sprintf "   Wxi:\n%s\n" (l.Wxi.Visualize())
            + sprintf "   Whi:\n%s\n" (l.Whi.Visualize())
            + sprintf "   Wxc:\n%s\n" (l.Wxc.Visualize())
            + sprintf "   Whc:\n%s\n" (l.Whc.Visualize())
            + sprintf "   Wxf:\n%s\n" (l.Wxf.Visualize())
            + sprintf "   Whf:\n%s\n" (l.Whf.Visualize())
            + sprintf "   Wxo:\n%s\n" (l.Wxo.Visualize())
            + sprintf "   Who:\n%s\n" (l.Who.Visualize())
            + sprintf "   bi:\n%s\n" (l.bi.Visualize())
            + sprintf "   bc:\n%s\n" (l.bc.Visualize())
            + sprintf "   bf:\n%s\n" (l.bf.Visualize())
            + sprintf "   bo:\n%s" (l.bo.Visualize())