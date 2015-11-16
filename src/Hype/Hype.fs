//
// This file is part of
// Hype: Compositional Machine Learning and Hyperparameter Optimization
//
// Copyright (c) 2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
//
// Hype is released under the MIT license.
// (See accompanying LICENSE file.)
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Brain and Computation Lab
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

/// Main namespace
namespace Hype

open System.IO
open DiffSharp.AD.Float32
open DiffSharp.Util

/// Random number generator
type Rnd() =
    static let mutable R = new System.Random()
    static member Seed(seed) = R <- new System.Random(seed)
    static member Permutation(n:int) =
        let swap i j (a:_[]) =
            let tmp = a.[i]
            a.[i] <- a.[j]
            a.[j] <- tmp
        let a = Array.init n (fun i -> i)
        a |> Array.iteri (fun i _ -> swap i (R.Next(i, n)) a)
        a
    static member UniformInt() = R.Next()
    static member UniformInt(max) = R.Next(max)
    static member UniformInt(min, max) = R.Next(min, max)
    static member Uniform() = float32 (R.NextDouble())
    static member UniformD() = D (float32 (R.NextDouble()))
    static member Uniform(max) = max * (float32 (R.NextDouble()))
    static member UniformD(max) = max * D (float32 (R.NextDouble()))
    static member Uniform(min, max) = min + (float32 (R.NextDouble())) * (max - min)
    static member UniformD(min, max) = min + D (float32 (R.NextDouble())) * (max - min)
    static member Normal() =
        let rec n() = 
            let x, y = (float32 (R.NextDouble())) * 2.0f - 1.0f, (float32 (R.NextDouble())) * 2.0f - 1.0f
            let s = x * x + y * y
            if s > 1.0f then n() else x * sqrt (-2.0f * (log s) / s)
        n()
    static member NormalD() = D (Rnd.Normal())
    static member Normal(mu, sigma) = Rnd.Normal() * sigma + mu
    static member NormalD(mu, sigma) = Rnd.NormalD() * sigma + mu
    
    static member UniformDV(n) = DV (Array.Parallel.init n (fun _ -> Rnd.Uniform()))
    static member UniformDV(n, max) = DV.init n (fun _ -> Rnd.UniformD(max))
    static member UniformDV(n, min, max) = DV.init n (fun _ -> Rnd.UniformD(min, max))
    static member NormalDV(n) = DV (Array.Parallel.init n (fun _ -> Rnd.Normal()))
    static member NormalDV(n, mu, sigma) = DV.init n (fun _ -> Rnd.NormalD(mu, sigma))

    static member UniformDM(m, n) = DM (Array2D.Parallel.init m n (fun _ _ -> Rnd.Uniform()))
    static member UniformDM(m, n, max) = DM.init m n (fun _ _ -> Rnd.UniformD(max))
    static member UniformDM(m, n, min, max) = DM.init m n (fun _ _ -> Rnd.UniformD(min, max))
    static member NormalDM(m, n) = DM (Array2D.Parallel.init m n (fun _ _ -> Rnd.Normal()))
    static member NormalDM(m, n, mu, sigma) = DM.init m n (fun _ _ -> Rnd.NormalD(mu, sigma))
    
    static member Choice(a:_[]) = a.[R.Next(a.Length)]
    static member Choice(a:_[], probs:float32[]) = Rnd.Choice(a, toDV probs)
    static member Choice(a:_[], probs:DV) =
        let probs' = probs / (DV.sum(probs))
        let p = float32 (R.NextDouble())
        let mutable r = 0.f
        let mutable i = 0
        let mutable hit = false
        while not hit do
            r <- r + (float32 probs'.[i])
            if r >= p then 
                hit <- true
            else
                i <- i + 1
        a.[i]

/// Dataset for holding training data
type Dataset private (x:DM, y:DM, xi:seq<int>, yi:seq<int>) =
    member val X = x with get
    member val Y = y with get
    member val Xi = xi |> Array.ofSeq with get
    member val Yi = yi |> Array.ofSeq with get
    new(x:DM, y:DM) =
        let xi = x |> DM.toCols |> Seq.toArray |> Array.map DV.maxIndex
        let yi = y |> DM.toCols |> Seq.toArray |> Array.map DV.maxIndex
        Dataset(x, y, xi, yi)
    new(xi:seq<int>, onehotdimsx:int, yi:seq<int>, onehotdimsy:int) =
        let x = xi |> Seq.map (fun i -> DV.standardBasis onehotdimsx i) |> DM.ofCols
        let y = yi |> Seq.map (fun i -> DV.standardBasis onehotdimsy i) |> DM.ofCols
        Dataset(x, y, xi, yi)
    new(xi:seq<int>, yi:seq<int>) =
        let onehotdimsx = 1 + Seq.max xi
        let onehotdimsy = 1 + Seq.max yi
        Dataset(xi, onehotdimsx, yi, onehotdimsy)
    new(x:DM, yi:seq<int>, onehotdimsy:int) =
        let xi = x |> DM.toCols |> Seq.toArray |> Array.map DV.maxIndex
        let y = yi |> Seq.map (fun i -> DV.standardBasis onehotdimsy i) |> DM.ofCols
        Dataset(x, y, xi, yi)
    new(xi:seq<int>, onehotdimsx:int, y:DM) =
        let x = xi |> Seq.map (fun i -> DV.standardBasis onehotdimsx i) |> DM.ofCols
        let yi = y |> DM.toCols |> Seq.toArray |> Array.map DV.maxIndex
        Dataset(x, y, xi, yi)
    new(x:DM, yi:seq<int>) =
        let onehotdimsy = 1 + Seq.max yi
        Dataset(x, yi, onehotdimsy)
    new(xi:seq<int>, y:DM) =
        let onehotdimsx = 1 + Seq.max xi
        Dataset(xi, onehotdimsx, y)
    new(s:seq<DV*DV>) =
        let x, y = s |> Seq.toArray |> Array.unzip
        Dataset(x |> DM.ofCols, y |> DM.ofCols)
    static member empty = Dataset(DM.empty, DM.empty)
    static member isEmpty (d:Dataset) = DM.isEmpty d.X && DM.isEmpty d.Y
    static member normalize (d:Dataset) = d.Normalize()
    static member normalizeX (d:Dataset) = d.NormalizeX()
    static member normalizeY (d:Dataset) = d.NormalizeY()
    static member standardize (d:Dataset) = d.Standardize()
    static member standardizeX (d:Dataset) = d.StandardizeX()
    static member standardizeY (d:Dataset) = d.StandardizeY()
    static member appendRowX (v:DV) (d:Dataset) = d.AppendRowX(v)
    static member appendRowY (v:DV) (d:Dataset) = d.AppendRowY(v)
    static member appendBiasRowX (d:Dataset) = d.AppendBiasRowX()
    static member toString (d:Dataset) = d.ToString()
    static member toStringFull (d:Dataset) = d.ToStringFull()
    static member toSeq (d:Dataset) = d.ToSeq()
    static member length (d:Dataset) = d.Length
    static member randomSubset (n:int) (d:Dataset) = d.RandomSubset(n)
    static member shuffle (d:Dataset) = d.Shuffle()
    static member item (i:int) (d:Dataset) = d.[i]
    member d.Item
        with get i = d.X.[*,i], d.Y.[*,i]
    member d.Length = d.X.Cols
    member d.ToSeq() =
        Seq.init d.Length (fun i -> d.[i])
    member d.RandomSubset(n) =
        let bi = Rnd.Permutation(d.Length)
        let x = Seq.init n (fun i -> d.X.[*, bi.[i]])
        let y = Seq.init n (fun i -> d.Y.[*, bi.[i]])
        Dataset(DM.ofCols x, DM.ofCols y)
    member d.Normalize() = Dataset(DM.normalize d.X, DM.normalize d.Y)
    member d.NormalizeX() = Dataset(DM.normalize d.X, d.Y)
    member d.NormalizeY() = Dataset(d.X, DM.normalize d.Y)
    member d.Standardize() = Dataset(DM.standardize d.X, DM.standardize d.Y)
    member d.StandardizeX() = Dataset(DM.standardize d.X, d.Y)
    member d.StandardizeY() = Dataset(d.X, DM.standardize d.Y)
    member d.Shuffle() = d.RandomSubset d.Length
    member d.GetSlice(lower, upper) =
        let l = max 0 (defaultArg lower 0)
        let u = min (d.X.Cols - 1) (defaultArg upper (d.Length - 1))
        Dataset(d.X.[*,l..u], d.Y.[*,l..u])
    member d.Filter (predicate:(DV*DV)->bool) =
        d.ToSeq() |> Seq.filter predicate |> Dataset
    member d.AppendRowX(v:DV) = Dataset(d.X |> DM.appendRow v, d.Y)
    member d.AppendRowY(v:DV) = Dataset(d.X, d.Y |> DM.appendRow v)
    member d.AppendBiasRowX() = d.AppendRowX(DV.create d.Length 1.f)
    override d.ToString() =
        "Hype.Dataset\n"
            + sprintf "   X: %i x %i\n" d.X.Rows d.X.Cols
            + sprintf "   Y: %i x %i" d.Y.Rows d.Y.Cols
    member d.ToStringFull() =
        "Hype.Dataset\n"
            + sprintf "   X:\n%O\n\n" d.X
            + sprintf "   Y:\n%O" d.Y
    member d.Visualize() =
        "Hype.Dataset\n"
            + sprintf "   X:\n%s\n\n" (d.X.Visualize())
            + sprintf "   Y:\n%s" (d.Y.Visualize())
    member d.VisualizeXColsAsImageGrid(imagerows:int) =
        d.ToString() + "\n"
            + "X's columns " + Util.VisualizeDMRowsAsImageGrid(d.X |> DM.transpose, imagerows)
    member d.VisualizeYColsAsImageGrid(imagerows:int) =
        d.ToString() + "\n"
            + "Y's columns " + Util.VisualizeDMRowsAsImageGrid(d.Y |> DM.transpose, imagerows)

/// Various utility functions
and Util =
    static member printLog (s:string) = printfn "[%A] %s" System.DateTime.Now s
    static member printModel (f:DV->DV) (d:Dataset) =
        d.ToSeq()
        |> Seq.map (fun (x, y) -> f x, y)
        |> Seq.iter (fun (x, y) -> printfn "f x: %A, y: %A" x y)
    static member LoadImage(filename:string) =
        let bmp = new System.Drawing.Bitmap(filename)
        let m = DM.init bmp.Height bmp.Width (fun i j -> float32 (bmp.GetPixel(i, j).GetBrightness()))
        bmp.Dispose()
        m
    static member LoadDelimited(filename:string, separators:char[]) =
        System.IO.File.ReadLines(filename)
        |> Seq.map (fun x -> x.Split(separators) |> Array.map float32)
        |> Seq.map toDV
        |> DM.ofRows 
    static member LoadDelimited(filename:string) =
        Util.LoadDelimited(filename, [|' '; ','; '\t'|])
    static member LoadMNISTPixels(filename, items) =
        let d = new BinaryReader(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read))
        let magicnumber = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2051 -> // Images
            let maxitems = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let rows = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let cols = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let n = min items maxitems
            d.ReadBytes(n * rows * cols)
            |> Array.map float32
            |> DV
            |> DM.ofDV n
            |> DM.transpose
        | _ -> failwith "Given file is not in the MNIST format."
    static member LoadMNISTLabels(filename, items) =
        let d = new BinaryReader(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read))
        let magicnumber = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2049 -> // Labels
            let maxitems = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            d.ReadBytes(min items maxitems)
            |> Array.map int
        | _ -> failwith "Given file is not in the MNIST format."
    static member LoadMNISTPixels(filename) = Util.LoadMNISTPixels(filename, System.Int32.MaxValue)
    static member LoadMNISTLabels(filename) = Util.LoadMNISTLabels(filename, System.Int32.MaxValue)
    static member VisualizeDMRowsAsImageGrid(w:DM, imagerows:int) =
        let rows = w.Rows
        let mm = int (floor (sqrt (float rows)))
        let nn = int (ceil (float rows / float mm))
        let m = imagerows
        let n = (w.[0, *] |> DV.toDM m).Cols
        let mutable mat = DM.create (mm * m) (nn * n) (DM.mean w)
        for i = 0 to mm - 1 do
            for j = 0 to nn - 1 do
                let row = i * nn + j
                if row < w.Rows then
                    mat <- DM.AddSubMatrix(mat, i * m, j * n, w.[row, *] |> DV.toDM m)
        sprintf "reshaped to (%i x %i), presented in a (%i x %i) grid:\n%s\n" m n mm nn (mat.Visualize())