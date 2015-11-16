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

/// Inference namespace
namespace Hype.Inference

open Hype
open DiffSharp.AD.Float32
open DiffSharp.Util

/// Hamiltonian MCMC sampler
type HMCSampler() =
    static member Sample(n, hdelta, hsteps, x0:DV, f:DV->D) =
        let leapFrog (u:DV->D) (k:DV->D) (d:D) steps (x0, p0) =
            let hd = d / 2.f
            [1..steps] 
            |> List.fold (fun (x, p) _ ->
                let p' = p - hd * grad u x
                let x' = x + d * grad k p'
                x', p' - hd * grad u x') (x0, p0)

        let u x = -log (f x) // potential energy
        let k p = (p * p) / D 2.f // kinetic energy
        let hamilton x p = u x + k p
        let x = ref x0
        [|for i in 1..n do
            let p = DV.init x0.Length (fun _ -> Rnd.Normal())
            let x', p' = leapFrog u k hdelta hsteps (!x, p)
            if Rnd.Uniform() < float32 (exp ((hamilton !x p) - (hamilton x' p'))) then x := x'
            yield !x|]