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

/// Natural language processing namespace
namespace Hype.NLP

open Hype
open DiffSharp.AD.Float32
open DiffSharp.Util

/// Language model
type Language(tokens:string[], punctuation:string[]) =
    member val Tokens = tokens
    
    static member TokenizeWords(text:string, punctuation) =
        //let mutable t' = text.ToLowerInvariant()
        let mutable t' = text
        punctuation |> Array.iter (fun p -> t' <- t'.Replace(p, " " + p + " "))
        t'.Split([|" "|], System.StringSplitOptions.RemoveEmptyEntries)

    new(text:string, punctuation:string[]) = Language(Language.TokenizeWords(text, punctuation) |> Set.ofArray |> Set.toArray, punctuation)
    new(text:string) = Language(text, [|"."; ","; ":"; ";"; "("; ")"; "!"; "?"|])

    member l.Length = l.Tokens.Length
    member l.EncodeOneHot(x:string) =
        Language.TokenizeWords(x, punctuation) |> l.EncodeOneHot
    member l.EncodeOneHot(x:string[]) =
        try
            //x |> Array.map (fun v -> v.ToLowerInvariant())
            x
            |> Array.map (fun v -> Array.findIndex (fun t -> t = v) l.Tokens)
            |> Array.map (DV.standardBasis l.Length) |> DM.ofCols
        with
            | _ -> failwith "Given token is not found in the language."
    member l.DecodeOneHot(x:DM) =
        try
            x |> DM.toCols |> Seq.map DV.maxIndex 
            |> Seq.map (fun i -> l.Tokens.[i]) |> Seq.toArray
        with
            | _ -> [||]
    member l.Sample(probs:DM) = probs |> DM.toCols |> Seq.map (fun v -> Rnd.Choice(l.Tokens, v)) |> Seq.toArray
    member l.Sample(probs:DV) = Rnd.Choice(l.Tokens, probs)
    member l.Sample(model:DM->DM, start:string, stop:string[], maxlen) =
        let mutable x = start
        let mutable i = 0
        let mutable t = ([while i < maxlen do
                            yield x
                            let p = x |> l.EncodeOneHot |> model
                            let d = l.Sample(p).[0]
                            match stop |> Array.tryFind (fun p -> p = d) with
                                | Some(_) ->
                                    yield d
                                    i <- maxlen
                                | _ -> 
                                    x <- d
                                    i <- i + 1]
                        |> List.map ((+) " ")
                        |> List.fold (+) "").Trim()
        punctuation |> Array.iter (fun p -> t <- t.Replace(" " + p, p))
        t