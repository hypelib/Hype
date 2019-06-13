(*** hide ***)
#r "../../src/Hype/bin/Release/netstandard2.0/DiffSharp.dll"
#r "../../src/Hype/bin/Release/netstandard2.0/Hype.dll"
#I "../../packages/R.NET.Community/lib/net40/"
#I "../../packages/R.NET.Community.FSharp/lib/net40/"
#I "../../packages/RProvider"
#load "RProvider.fsx"
fsi.ShowDeclarationValues <- false

(**
Recurrent neural networks
=========================

In this example we build a recurrent neural network (RNN) for a language modeling task and train it with a short passage of text for a quick demonstration. Hype currently has three RNN models implemented as **Hype.Neural** layers, which can be combined freely with other layer types, explained, for example, in the [neural networks](feedforwardnets.html) page. **Hype.Neural.Recurrent** implements the "vanilla" RNN layer, **Hype.Neural.LSTM** implements the LSTM layer, and **Hype.Neural.GRU** implements the gated recurrent unit (GRU) layer.

### Language modeling

RNNs are well suited for constructing [language models,](https://en.wikipedia.org/wiki/Language_model) where we need to predict the probability of a word (or token) given the history of the tokens that came before it. Here, we will use an LSTM-based RNN to construct a word-level language model from a short passage of text, for a basic demonstration of usage. This model can be scaled to larger problems. State-of-the-art models of this type can require considerable computing resources and training time.

The text is from the beginning of Virgil's Aeneid, Book I.
*)

let text = "I sing of arms and the man, he who, exiled by fate, first came from the coast of Troy to Italy, and to Lavinian shores – hurled about endlessly by land and sea, by the will of the gods, by cruel Juno’s remorseless anger, long suffering also in war, until he founded a city and brought his gods to Latium: from that the Latin people came, the lords of Alba Longa, the walls of noble Rome. Muse, tell me the cause: how was she offended in her divinity, how was she grieved, the Queen of Heaven, to drive a man, noted for virtue, to endure such dangers, to face so many trials? Can there be such anger in the minds of the gods?"

(**
Hype provides a simple **Hype.NLP.Language** type for tokenizing text. You can look at the [API reference](reference/index.html) and the [source code](https://github.com/hypelib/Hype/blob/master/src/Hype/NLP.fs) for a better understanding of its usage.
*)

open Hype
open Hype.Neural
open Hype.NLP
open DiffSharp.AD.Float32
open DiffSharp.Util

let lang = Language(text)

lang.Tokens |> printfn "%A"
lang.Length |> printfn "%A"

(**
These are the tokens extracted from the text, including some of the punctuation marks. When we are sampling from the RNN language model, we will make use of the "." token for signaling the end of a sentence. The puncutation marks are configurable when you are constructing the **Language** instance. If they are not provided, a default set is used.

<pre>
[|","; "."; ":"; "?"; "Alba"; "Can"; "Heaven"; "I"; "Italy"; "Juno’s"; "Latin";
  "Latium"; "Lavinian"; "Longa"; "Muse"; "Queen"; "Rome"; "Troy"; "a"; "about";
  "also"; "and"; "anger"; "arms"; "be"; "brought"; "by"; "came"; "cause"; "city";
  "coast"; "cruel"; "dangers"; "divinity"; "drive"; "endlessly"; "endure";
  "exiled"; "face"; "fate"; "first"; "for"; "founded"; "from"; "gods"; "grieved";
  "he"; "her"; "his"; "how"; "hurled"; "in"; "land"; "long"; "lords"; "man";
  "many"; "me"; "minds"; "noble"; "noted"; "of"; "offended"; "people";
  "remorseless"; "sea"; "she"; "shores"; "sing"; "so"; "such"; "suffering";
  "tell"; "that"; "the"; "there"; "to"; "trials"; "until"; "virtue"; "walls";
  "war"; "was"; "who"; "will"; "–"|]
  
  86
</pre>
There are 86 tokens in this language instance.

Now let's transform the full text to a dataset, using the **Language** instance holding these tokens. The text will be encoded in a matrix where each column is a representation of each word as a _one-hot_ vector.
*)

let text' = lang.EncodeOneHot(text)
text'.Visualize() |> printfn "%s"

(**
<pre>
DM : 86 x 145
</pre>

Out of these 145 words, we will construct a dataset where the inputs are the first 144 words and the target outputs are the 144 words starting with a one word shift. This means that, for each word, we want the output (the prediction) to be the following word in our text passage.
*)

let data = Dataset(text'.[*, 0..(text'.Cols - 2)],
                   text'.[*, 1..(text'.Cols - 1)])

(**
<pre>
val data : Dataset = Hype.Dataset
   X: 86 x 144
   Y: 86 x 144
</pre>

RNNs, and especially the LSTM variety that we will use, can make predictions that take long-term dependencies and contextual information into account. When the language model is trained with a large enough text corpus and the network has enough capacity, state-of-the-art RNN language models are able to learn complex grammatical relations.

For our quick demonstration, we use a linear word embedding layer of 20 units, an LSTM of 100 units and a final linear layer of 86 units (the size of our vocabulary) followed by **softmax** activation.
*)

let dim = lang.Length // Vocabulary size, here 86

let n = FeedForward()
n.Add(Linear(dim, 20))
n.Add(LSTM(20, 100))
n.Add(Linear(100, dim))
n.Add(DM.mapCols softmax)

(**
You can also easily stack multiple RNNs on top of each other.
*)

let n = FeedForward()
n.Add(Linear(dim, 20))
n.Add(LSTM(20, 100))
n.Add(LSTM(100, 100))
n.Add(Linear(100, dim))
n.Add(DM.mapCols softmax)

(**
We will observe the the performance of our RNN during training by sampling random sentences from the language model. 

Remember that the final output of the network, through the softmax activation, is a vector of word probabilities. When we are sampling, we start with a word, supply this to the network, and use the resulting probabilities at the output to sample from the vocabulary where words with higher probability are more likely to be selected. We then continue by giving the network the last sampled word and repeating this until we hit an "end of sentence" token (we use "." here) or reach a limit of maximum sentence length.

This is how we would sample a sentence starting with a specific word.
*)

n.Reset()
for i = 0 to 5 do
    lang.Sample(n.Run, "I", [|"."|], 30) // Use "." as the stop token, limit maximum sentence length to 30.
    |> printfn "%s"

(**

Because the model is not trained, we get sequences of random words from the vocabulary.

<pre>
I be: she dangers Latium endlessly gods remorseless divinity tell and his offended lords trials? about war trials and anger shores so anger Alba a Alba sing her
I? came exiled – suffering shores anger came Latium people sing sing remorseless who brought war walls endlessly anger me founded his.
I – will long of in offended cruel until Queen Italy who anger lords Queen in Longa Muse who people about suffering Italy also grieved cruel hurled who me about
I endlessly city first by face, a Heaven me hurled sea such long noted she noted many sea city anger I noted remorseless cause Queen to remorseless Italy coast
I sea noted noble me minds long sing cause people in walls Italy by Longa first, for grieved sea many walls Troy came was endlessly of in Latium Latium
I and Latin of many suffering Alba Latium war.
</pre>

We set a training cycle where we run one epoch of training followed by sampling one sentence starting with the word "I". In each epoch, we run through the whole training dataset. With a larger training corpus, we could also run the training with minibatches by stating this in the parameter set (commented out below).

Like the sample sentences above, at the beginning of training, we see mostly random orderings of words. As the training progresses, the cross-entropy loss for our dataset is decreasing and the sentences start exhibiting meaningful word patterns.
*)

for i = 0 to 1000 do
    let par = {Params.Default with
                //Batch = Minibatch 10
                LearningRate = LearningRate.RMSProp(D 0.01f, D 0.9f)
                Loss = CrossEntropyOnSoftmax
                Epochs = 1
                Silent = true       // Suppress the regular printing of training progress
                ReturnBest = false} 
    let loss, _ = Layer.Train(n, data, par)
    printfn "Epoch: %*i | Loss: %O | Sample: %s" 3 i loss (lang.Sample(n.Run, "I", [|"."|], 30))

(**

Here is a selection of sentences demonstrating the progress of training.

<pre>
Epoch:   0 | Loss: D  4.478101e+000 | Sample: I Queen drive she Alba endlessly Queen the by how tell his from grieved war her there drive people – lords coast he.
Epoch:  10 | Loss: D  4.102071e+000 | Sample: I people to,, Rome how the he of – sing fate, Muse, by,, Muse the of man Queen Latin and in her cause:
Epoch:  30 | Loss: D  3.438288e+000 | Sample: I walls long to first dangers she her, to founded to virtue sea first Can dangers a founded about Can Queen lords from sea by remorseless founded endlessly Latium
Epoch:  40 | Loss: D  2.007577e+000 | Sample: I Alba gods Alba Rome, the walls Alba Muse Rome anger me the the of the gods to who man me first founded offended endlessly until also grieved long
Epoch:  50 | Loss: D  9.753818e-001 | Sample: I sing people cruel: me the of Rome.
Epoch:  60 | Loss: D  3.944587e-001 | Sample: I sing sing Troy to so hurled endlessly by land sea, by to – hurled about by the of arms, by Juno’s such anger long also in her
Epoch:  70 | Loss: D  2.131431e-001 | Sample: I sing of and the of Longa, by Juno’s anger was in her of Heaven, to a city brought his gods to a gods to Lavinian hurled to
Epoch:  80 | Loss: D  1.895453e-001 | Sample: I sing, by will the of Rome.
Epoch:  90 | Loss: D  1.799535e-001 | Sample: I sing? there Muse the of the of the of arms by the: how she offended in the of? a, he shores hurled by land to
Epoch: 100 | Loss: D  1.733837e-001 | Sample: I sing arms the of Alba gods who, by Juno’s Rome such anger the of the of arms and, by, by from the coast Rome.
Epoch: 110 | Loss: D  1.682917e-001 | Sample: I sing Troy by, by from the of arms and, by, by from came, by Juno’s anger long in the of the of arms cruel Muse
Epoch: 120 | Loss: D  1.639529e-001 | Sample: I sing arms the of Rome.
Epoch: 130 | Loss: D  1.600647e-001 | Sample: I sing arms and, by Juno’s remorseless there and the of the of arms and, by Alba coast Troy to a – his gods by of the of
Epoch: 140 | Loss: D  1.564835e-001 | Sample: I sing arms by the of Rome.
Epoch: 150 | Loss: D  1.531392e-001 | Sample: I sing arms cruel, exiled by coast, he a city in the of the of arms.
Epoch: 160 | Loss: D  1.499920e-001 | Sample: I sing arms cruel man, by the trials arms to shores hurled endlessly by the of gods Italy, me the of Rome.
Epoch: 200 | Loss: D  1.390327e-001 | Sample: I sing arms and, by Juno’s such of the of the of arms Italy, by from the sing arms walls of the of Rome.
Epoch: 230 | Loss: D  1.322940e-001 | Sample: I sing arms the man he, tell from the of arms Italy, by fate, by the of Troy Italy, by fate first from the of the
Epoch: 260 | Loss: D  1.264137e-001 | Sample: I sing brought Muse Muse the of Heaven, by shores remorseless there he in the of arms cruel, by fate, he from the gods to Italy,
Epoch: 420 | Loss: D  1.131158e-001 | Sample: I sing of arms the of Heaven, by Juno’s remorseless hurled such in the of arms.
Epoch: 680 | Loss: D  9.938217e-002 | Sample: I of arms the man he, exiled fate, he virtue, to a? Can be such in the of the of of the of arms.
Epoch: 923 | Loss: D  9.283429e-002 | Sample: I sing of arms and the man he, by fate came from the of to Italy, by the, by Juno’s anger of Rome.
</pre>
*)
