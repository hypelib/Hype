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
    /// Seed the random number generator with integer `seed`
    static member Seed(seed) = R <- new System.Random(seed)
    /// Generate a random permutation of a set of length `n`
    static member Permutation(n:int) =
        let swap i j (a:_[]) =
            let tmp = a.[i]
            a.[i] <- a.[j]
            a.[j] <- tmp
        let a = Array.init n (fun i -> i)
        a |> Array.iteri (fun i _ -> swap i (R.Next(i, n)) a)
        a
    /// Sample a non-negative random integer
    static member UniformInt() = R.Next()
    /// Sample a non-negative random integer less than `max`
    static member UniformInt(max) = R.Next(max)
    /// Sample a random integer between `min` and `max`
    static member UniformInt(min, max) = R.Next(min, max)
    /// Sample a `float32` from the standard uniform distribution. X ~ U(0,1)
    static member Uniform() = float32 (R.NextDouble())
    /// Sample a `D` from the standard uniform distribution. X ~ U(0,1)
    static member UniformD() = D (float32 (R.NextDouble()))
    /// Sample a `float32` from the uniform distribution between zero and `max`. X ~ U(0,max)
    static member Uniform(max) = max * (float32 (R.NextDouble()))
    /// Sample a `D` from the unifrom distribution between zero and `max`. X ~ U(0,max)
    static member UniformD(max) = max * D (float32 (R.NextDouble()))
    /// Sample a `float32` from the uniform distribution between `min` and `max`. X ~ U(min,max)
    static member Uniform(min, max) = min + (float32 (R.NextDouble())) * (max - min)
    /// Sample a `D` from the uniform distribution between `min` and `max`. X ~ U(min,max)
    static member UniformD(min, max) = min + D (float32 (R.NextDouble())) * (max - min)
    /// Sample a `float32` from the standard normal distribution. X ~ N(0,1)
    static member Normal() =
        let rec n() = 
            let x, y = (float32 (R.NextDouble())) * 2.0f - 1.0f, (float32 (R.NextDouble())) * 2.0f - 1.0f
            let s = x * x + y * y
            if s > 1.0f then n() else x * sqrt (-2.0f * (log s) / s)
        n()
    /// Sample a `D` from the standard normal distribution. X ~ N(0,1)
    static member NormalD() = D (Rnd.Normal())
    /// Sample a `float32` from the normal distribution with given mean `mu` and standard deviation `sigma`. X ~ N(mu,sigma)
    static member Normal(mu, sigma) = Rnd.Normal() * sigma + mu
    /// Sample a `D` from the normal distribution with given mean `mu` and standard deviation `sigma`. X ~ N(mu,sigma)
    static member NormalD(mu, sigma) = Rnd.NormalD() * sigma + mu
    
    /// Sample a `DV` of length `n` from the standard uniform distribution. Elements of vector X ~ U(0,1)
    static member UniformDV(n) = DV (Array.Parallel.init n (fun _ -> Rnd.Uniform()))
    /// Sample a `DV` of length `n` from the uniform distribution between zero and `max`. Elements of vector X ~ U(0,max)
    static member UniformDV(n, max) = DV.init n (fun _ -> Rnd.UniformD(max))
    /// Sample a `DV` of length `n` from the uniform distribution between `min` and `max`. Elements of vector X ~ U(min,max)
    static member UniformDV(n, min, max) = DV.init n (fun _ -> Rnd.UniformD(min, max))
    /// Sample a `DV` of length `n` from the standard normal distribution. Elements of vector X ~ N(0,1)
    static member NormalDV(n) = DV (Array.Parallel.init n (fun _ -> Rnd.Normal()))
    /// Sample a `DV` of length `n` from the normal distribution with given mean `mu` and standard deviation `sigma`. Elements of vector X ~ N(mu,sigma)
    static member NormalDV(n, mu, sigma) = DV.init n (fun _ -> Rnd.NormalD(mu, sigma))

    /// Sample a `DM` of `m` rows and `n` columns from the standard uniform distribution. Elements of matrix X ~ U(0,1)
    static member UniformDM(m, n) = DM (Array2D.Parallel.init m n (fun _ _ -> Rnd.Uniform()))
    /// Sample a `DM` of `m` rows and `n` columns from the uniform distribution between zero and `max`. Elements of matrix X ~ U(0,max)
    static member UniformDM(m, n, max) = DM.init m n (fun _ _ -> Rnd.UniformD(max))
    /// Sample a `DM` of `m` rows and `n` columns from the uniform distribution between `min` and `max`. Elements of matrix X ~ U(min,max)
    static member UniformDM(m, n, min, max) = DM.init m n (fun _ _ -> Rnd.UniformD(min, max))
    /// Sample a `DM` of `m` rows and `n` columns from the standard normal distribution. Elements of matrix X ~ N(0,1)
    static member NormalDM(m, n) = DM (Array2D.Parallel.init m n (fun _ _ -> Rnd.Normal()))
    /// Sample a `DM` of `m` rows and `n` columns from the normal distribution with given mean `mu` and standard deviation `sigma`. Elements of matrix X ~ N(mu,sigma)
    static member NormalDM(m, n, mu, sigma) = DM.init m n (fun _ _ -> Rnd.NormalD(mu, sigma))
    
    /// Select a random element of array `a`
    static member Choice(a:_[]) = a.[R.Next(a.Length)]
    /// Select a random element of array `a`, given selection probabilities in array `probs`
    static member Choice(a:_[], probs:float32[]) = Rnd.Choice(a, toDV probs)
    /// Select a random element of array `a`, given selection probabilities in vector `probs`
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
    /// The matrix X of input values, where columns are the individual inputs Xi
    member val X = x with get
    /// The matrix Y of output values, where columns are the individual outputs Yi
    member val Y = y with get
    /// The index of the maximum elements of individual inputs Xi, used for one-hot representations
    member val Xi = xi |> Array.ofSeq with get
    /// The index of the maximum elements of individual outputs Yi, used for one-hot reprsentations
    member val Yi = yi |> Array.ofSeq with get
    /// Construct a dataset with given input matrix `x` and output matrix `y`. Columns of `x` and `y` are the individual inputs and corresponding outputs.
    new(x:DM, y:DM) =
        let xi = x |> DM.toCols |> Seq.toArray |> Array.map DV.maxIndex
        let yi = y |> DM.toCols |> Seq.toArray |> Array.map DV.maxIndex
        Dataset(x, y, xi, yi)
    /// Construct a dataset of one-hot input and output elements. `xi` are the input indices, `onehotdimsx` is the input dimensions, `yi` are the output indices, `onehotdimsy` is the output dimensions.
    new(xi:seq<int>, onehotdimsx:int, yi:seq<int>, onehotdimsy:int) =
        let x = xi |> Seq.map (fun i -> DV.standardBasis onehotdimsx i) |> DM.ofCols
        let y = yi |> Seq.map (fun i -> DV.standardBasis onehotdimsy i) |> DM.ofCols
        Dataset(x, y, xi, yi)
    /// Construct a dataset of one-hot input and output elements. `xi` are the input indices, input dimensions is max(xi) + 1, `yi` are the output indices, output dimensions is max(yi) + 1.
    new(xi:seq<int>, yi:seq<int>) =
        let onehotdimsx = 1 + Seq.max xi
        let onehotdimsy = 1 + Seq.max yi
        Dataset(xi, onehotdimsx, yi, onehotdimsy)
    /// Construct a dataset with given input matrix `x` and one-hot output elements. `yi` are the output indices, `onehotdimsy` is the output dimensions.
    new(x:DM, yi:seq<int>, onehotdimsy:int) =
        let xi = x |> DM.toCols |> Seq.toArray |> Array.map DV.maxIndex
        let y = yi |> Seq.map (fun i -> DV.standardBasis onehotdimsy i) |> DM.ofCols
        Dataset(x, y, xi, yi)
    /// Construct a dataset with one-hot input elements and given output matrix `y`. `xi` are the input indices, `onehotdimsx` is the input dimensions.
    new(xi:seq<int>, onehotdimsx:int, y:DM) =
        let x = xi |> Seq.map (fun i -> DV.standardBasis onehotdimsx i) |> DM.ofCols
        let yi = y |> DM.toCols |> Seq.toArray |> Array.map DV.maxIndex
        Dataset(x, y, xi, yi)
    /// Construct a dataset with given input matrix `x` and one-hot output elements. `yi` are the output indices, output dimensions is max(yi) + 1.
    new(x:DM, yi:seq<int>) =
        let onehotdimsy = 1 + Seq.max yi
        Dataset(x, yi, onehotdimsy)
    /// Construct a dataset with one-hot input elements and given output matrix `y`. `xi` are the input indices, input dimensions is max(xi) + 1.
    new(xi:seq<int>, y:DM) =
        let onehotdimsx = 1 + Seq.max xi
        Dataset(xi, onehotdimsx, y)
    /// Construct a dataset from the given sequence of input-output vector pairs
    new(s:seq<DV*DV>) =
        let x, y = s |> Seq.toArray |> Array.unzip
        Dataset(x |> DM.ofCols, y |> DM.ofCols)
    /// The empty dataset
    static member empty = Dataset(DM.empty, DM.empty)
    /// Check whether dataset `d` is empty
    static member isEmpty (d:Dataset) = DM.isEmpty d.X && DM.isEmpty d.Y
    /// Normalize the values in the input matrix X and output matrix Y of dataset `d` to be in the range [0,1]
    static member normalize (d:Dataset) = d.Normalize()
    /// Normalize the values in the input matrix X of dataset `d` to be in the range [0,1]
    static member normalizeX (d:Dataset) = d.NormalizeX()
    /// Normalize the values in the output matrix Y of dataset `d` to be in the range [0,1]
    static member normalizeY (d:Dataset) = d.NormalizeY()
    /// Standardize the values in the input matrix X and output matrix Y of dataset `d` to have zero mean and unit variance
    static member standardize (d:Dataset) = d.Standardize()
    /// Standardize the values in the input matrix X of dataset `d` to have zero mean and unit variance
    static member standardizeX (d:Dataset) = d.StandardizeX()
    /// Standardize the values in the output matrix Y of dataset `d` to have zero mean and unit variance
    static member standardizeY (d:Dataset) = d.StandardizeY()
    /// Append a new row `v` to the input matrix X of dataset `d`
    static member appendRowX (v:DV) (d:Dataset) = d.AppendRowX(v)
    /// Append a new tow `v` to the output matrix Y of dataset `d`
    static member appendRowY (v:DV) (d:Dataset) = d.AppendRowY(v)
    /// Append a row of ones to the input matrix X of dataset `d`
    static member appendBiasRowX (d:Dataset) = d.AppendBiasRowX()
    /// Get a summary string of dataset `d`
    static member toString (d:Dataset) = d.ToString()
    /// Get a string representation of dataset `d` showing all values
    static member toStringFull (d:Dataset) = d.ToStringFull()
    /// Get the input-output pairs of dataset `d` as a sequence
    static member toSeq (d:Dataset) = d.ToSeq()
    /// The length of dataset `d`, i.e., the number of columns in input matrix X and output matrix Y
    static member length (d:Dataset) = d.Length
    /// Sample a random subset of length `n` from dataset `d`
    static member randomSubset (n:int) (d:Dataset) = d.RandomSubset(n)
    /// Shuffle the order of elements in dataset `d`
    static member shuffle (d:Dataset) = d.Shuffle()
    /// Get the input-output pair with index `i` from dataset `d`
    static member item (i:int) (d:Dataset) = d.[i]
    /// Get element `i`
    member d.Item
        with get i = d.X.[*,i], d.Y.[*,i]
    /// The length of the dataset, i.e., the number of columns in input matrix X and output matrix Y
    member d.Length = d.X.Cols
    /// Get the input-output pairs as a sequence
    member d.ToSeq() =
        Seq.init d.Length (fun i -> d.[i])
    /// Sample a random subset of length `n` from this dataset
    member d.RandomSubset(n) =
        let bi = Rnd.Permutation(d.Length)
        let x = Seq.init n (fun i -> d.X.[*, bi.[i]])
        let y = Seq.init n (fun i -> d.Y.[*, bi.[i]])
        Dataset(DM.ofCols x, DM.ofCols y)
    /// Normalize the values in the input matrix X and output matrix Y to be in the range [0,1]
    member d.Normalize() = Dataset(DM.normalize d.X, DM.normalize d.Y)
    /// Normalize the values in the input matrix X to be in the range [0,1]
    member d.NormalizeX() = Dataset(DM.normalize d.X, d.Y)
    /// Normalize the values in the output matrix Y to be in the range [0,1]
    member d.NormalizeY() = Dataset(d.X, DM.normalize d.Y)
    /// Standardize the values in the input matrix X and output matrix Y to have zero mean and unit variance
    member d.Standardize() = Dataset(DM.standardize d.X, DM.standardize d.Y)
    /// Standardize the values in the input matrix X to have zero mean and unit variance
    member d.StandardizeX() = Dataset(DM.standardize d.X, d.Y)
    /// Standardize the values in the output matrix Y to have zero mean and unit variance
    member d.StandardizeY() = Dataset(d.X, DM.standardize d.Y)
    /// Shuffle the order of elements in the dataset
    member d.Shuffle() = d.RandomSubset d.Length
    /// Get a slice of the dataset between `lower` and `upper` indices
    member d.GetSlice(lower, upper) =
        let l = max 0 (defaultArg lower 0)
        let u = min (d.X.Cols - 1) (defaultArg upper (d.Length - 1))
        Dataset(d.X.[*,l..u], d.Y.[*,l..u])
    /// Get a new dataset of the entries for which the `predicate` is true
    member d.Filter (predicate:(DV*DV)->bool) =
        d.ToSeq() |> Seq.filter predicate |> Dataset
    /// Append a new row `v` to the input matrix X
    member d.AppendRowX(v:DV) = Dataset(d.X |> DM.appendRow v, d.Y)
    /// Append a new row `v` to the output matrix Y
    member d.AppendRowY(v:DV) = Dataset(d.X, d.Y |> DM.appendRow v)
    /// Append a row of all ones to the input matrix X
    member d.AppendBiasRowX() = d.AppendRowX(DV.create d.Length 1.f)
    /// Get a summary string of this dataset
    override d.ToString() =
        "Hype.Dataset\n"
            + sprintf "   X: %i x %i\n" d.X.Rows d.X.Cols
            + sprintf "   Y: %i x %i" d.Y.Rows d.Y.Cols
    /// Get a string representation of this dataset showing all values
    member d.ToStringFull() =
        "Hype.Dataset\n"
            + sprintf "   X:\n%O\n\n" d.X
            + sprintf "   Y:\n%O" d.Y
    /// Get a string visualization of this dataset
    member d.Visualize() =
        "Hype.Dataset\n"
            + sprintf "   X:\n%s\n\n" (d.X.Visualize())
            + sprintf "   Y:\n%s" (d.Y.Visualize())
    /// Visualize the values of the input matrix X where each column will be reshaped to an image with `imagerows` rows
    member d.VisualizeXColsAsImageGrid(imagerows:int) =
        d.ToString() + "\n"
            + "X's columns " + Util.VisualizeDMRowsAsImageGrid(d.X |> DM.transpose, imagerows)
    /// Visualize the values of the output matrix Y where each column will be reshaped to an image with `imagerows` rows
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
    /// Load bitmap image with given `filename` to `DM`
    static member LoadImage(filename:string) =
        let bmp = new System.Drawing.Bitmap(filename)
        let m = DM.init bmp.Height bmp.Width (fun i j -> float32 (bmp.GetPixel(i, j).GetBrightness()))
        bmp.Dispose()
        m
    /// Load values from delimited text file with given `filename` and separator characters `separators`
    static member LoadDelimited(filename:string, separators:char[]) =
        System.IO.File.ReadLines(filename)
        |> Seq.map (fun x -> x.Split(separators) |> Array.map float32)
        |> Seq.map toDV
        |> DM.ofRows
    /// Load values from delimited text file with given `filename` and a default set of separator characters: space, comma, or tab
    static member LoadDelimited(filename:string) =
        Util.LoadDelimited(filename, [|' '; ','; '\t'|])
    /// Load values from the MNIST database images, from given `filename`, reading `n` number of elements
    static member LoadMNISTPixels(filename, n) =
        let d = new BinaryReader(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read))
        let magicnumber = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2051 -> // Images
            let maxitems = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let rows = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let cols = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let n = min n maxitems
            d.ReadBytes(n * rows * cols)
            |> Array.map float32
            |> DV
            |> DM.ofDV n
            |> DM.transpose
        | _ -> failwith "Given file is not in the MNIST format."
    /// Load values from the MNIST database labels, from given `filename`, reading `n` number of elements
    static member LoadMNISTLabels(filename, n) =
        let d = new BinaryReader(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read))
        let magicnumber = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2049 -> // Labels
            let maxitems = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            d.ReadBytes(min n maxitems)
            |> Array.map int
        | _ -> failwith "Given file is not in the MNIST format."
    /// Load values from the MNIST database images, from given `filename`, reading all elements
    static member LoadMNISTPixels(filename) = Util.LoadMNISTPixels(filename, System.Int32.MaxValue)
    /// Load values from the MNIST database labels, from given `filename`, reading all elements
    static member LoadMNISTLabels(filename) = Util.LoadMNISTLabels(filename, System.Int32.MaxValue)
    /// Generate a string representation of matrix `w`, reshaping each row into an image with `imagerows` rows, and presenting resulting images together in an optimal grid layout.
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