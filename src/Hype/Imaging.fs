//
// This file is part of
// Hype: Machine Learning and Hyperparameter Optimization Library
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

namespace Hype

open FsAlg.Generic
open DiffSharp.AD
open Hype
open System.Drawing

type Imaging =
    static member Load (filename:string) =
        let bmp = new System.Drawing.Bitmap(filename)
        let m = Matrix.init bmp.Height bmp.Width (fun i j -> float (bmp.GetPixel(i, j).GetBrightness()))
        bmp.Dispose()
        m