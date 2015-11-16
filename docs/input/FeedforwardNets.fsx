(*** hide ***)
#r "../../src/Hype/bin/Release/DiffSharp.dll"
#r "../../src/Hype/bin/Release/Hype.dll"
#I "../../packages/R.NET.Community.1.6.4/lib/net40/"
#I "../../packages/R.NET.Community.FSharp.1.6.4/lib/net40/"
#I "../../packages/RProvider.1.1.14"
#load "RProvider.fsx"
fsi.ShowDeclarationValues <- true

(**
Feedforward neural networks
===========================

In this example, we implement a softmax classifier network with several hidden layers. Also see the [regression example](regression.html) for some relevant basics.

We again demonstrate the library with the MNIST database, this time using the full dataset for building a classifier with 10 outputs representing the class probabilities of an input image belonging to one of the ten categories.

### Loading the data

We load the data and form the training, validation, and test datasets. The datasets are shuffled and the input data are normalized.
*)

open Hype
open Hype.Neural
open DiffSharp.AD.Float32
open DiffSharp.Util

let MNIST = Dataset(Util.LoadMNISTPixels("C:/datasets/MNIST/train-images.idx3-ubyte", 60000),
                    Util.LoadMNISTLabels("C:/datasets/MNIST/train-labels.idx1-ubyte", 60000)).NormalizeX()

let MNISTtrain = MNIST.[..58999].Shuffle()
let MNISTvalid = MNIST.[59000..].Shuffle()

let MNISTtest = 
    Dataset(Util.LoadMNISTPixels("C:/datasets/MNIST/t10k-images.idx3-ubyte", 10000),
            Util.LoadMNISTLabels("C:/datasets/MNIST/t10k-labels.idx1-ubyte", 10000)).NormalizeX().Shuffle()

(**
<pre>
val MNISTtrain : Dataset = Hype.Dataset
   X: 784 x 59000
   Y: 1 x 59000
val MNISTvalid : Dataset = Hype.Dataset
   X: 784 x 1000
   Y: 1 x 1000
val MNISTtest : Dataset = Hype.Dataset
   X: 784 x 10000
   Y: 1 x 10000
</pre>

*)

MNISTtrain.[..5].VisualizeXColsAsImageGrid(28) |> printfn "%s"

(**

    [lang=cs]
    Hype.Dataset
       X: 784 x 6
       Y: 1 x 6
    X's columns reshaped to (28 x 28), presented in a (2 x 3) grid:
    DM : 56 x 84
                                                                                    
                                                                                    
                                                                                    
                                                  ·▴█                                   
                                                 ■■♦█·                  █■              
                                                ▪███■▪                 -██■-            
                                             ·■███♦●                    ·●██■▪          
                            -♦▪             ·████♦                         -♦█♦         
                         -♦■▪·              █■█●                             ■█·        
                      ·●██♦                 ██●·                             ██·        
             ·▴     ·●██▪                -■██▪                              ■█▪         
            ·■▪   ·▪■■▪                 ·███▪  ·▴·                         ♦█▪          
           ♦■▪  ·■■▴                    ♦███●▴▴♦██●                       ██▴           
           █   ■█·                      ■█■█■██████■                     ■█▪            
           ■█· -                        ██▴█████████▴                  ·██·             
            ●█▪                         █▪●■████■███▴                 ·███♦·            
             ·■■                        █▴●- ·    ●█·                  ●♦♦♦█▪           
               ■█·                      █●■       ▪█                       ▴█-          
               ·█●                      ███·  ·●■■██                        ■█          
               ·█●                      █●■▴▴██████●                        ██          
               ●█▴                      ♦█████████▴                        ▴█▪          
            -♦██♦                       ▪█████♦▪·                    ●     █■           
            ▴■■▴                          ▪▪▪                        █   ·■♦            
                                                                     ■♦▪♦█▪             
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
           ·■       ·♦                           ▪■♦-                  ·●●█■·           
           ██·      ♦█●                         ██■■█▪-●·              ▪■·▴●            
           ██▴      ♦██▪                       ■█-  ▴███■              █♦               
          -██       ♦██●                      ▪█·   ▪██▪              ▪█♦  ■▴           
          -██       ♦██■                      █●   ♦██▴               ♦█- ██■           
          ▴█♦       ▴██■                     ▴█  ·██■·                ♦█- ███           
          ♦█♦       -███                     ▴█·▴██♦                  ▪██■███·          
          ♦█·       ·██●                      ████▴                    ■██●■█·          
          ♦█▴    ··-███●                     ▪██♦                       ·  ♦█·          
          ♦██■■■■██████●                   ·■███                           ██           
          ▪█████████■♦█●                  ♦██■●█                           █■           
           ●██████▴-  █■                ▴██■- ●█                          ▴█▪           
           ·███♦·     ██               ♦█■▴   █●                          ██▴           
            ▪▪        ██▪             ███    ■█·                          ██▴           
                      ●██            ●████■■██-                           ██            
                      ▴██            -▪▪▪▪▪▪▪-                            ██            
                       ██                                                -██            
                       ■█                                                -█●            
                       ●█·                                               -█▪            
                       ·█                                                 █             
                                                                                    
                                                                                    

### Defining the model

We define a neural network with 3 layers: (1) a hidden layer with 300 units, followed by ReLU activation, (2) a hidden layer with 100 units, followed by ReLU activation, (3) a final layer with 10 units, followed by softmax transformation.
*)

let n = FeedForward()
n.Add(Linear(28 * 28, 300, Initializer.InitReLU))
n.Add(reLU)
n.Add(Linear(300, 100, Initializer.InitReLU))
n.Add(reLU)
n.Add(Linear(100, 10))
n.Add(fun m -> m |> DM.mapCols softmax) // Note the free inline implementation of the layer

n.ToString() |> printfn "%s"

(**
    [lang=cs]
    Hype.Neural.FeedForward
       Learnable parameters: 266610
       (0) -> (1) -> (2) -> (3) -> (4) -> (5)

       (0): Hype.Neural.Linear
       784 -> 300
       Learnable parameters: 235500
       Init: ReLU
       W   : 300 x 784
       b   : 300

       (1): Hype.Neural.Activation

       (2): Hype.Neural.Linear
       300 -> 100
       Learnable parameters: 30100
       Init: ReLU
       W   : 100 x 300
       b   : 100

       (3): Hype.Neural.Activation

       (4): Hype.Neural.Linear
       100 -> 10
       Learnable parameters: 1010
       Init: Standard
       W   : 10 x 100
       b   : 10

       (5): Hype.Neural.Activation
*)


(**

### Freely implementing transformation layers

Now let's have a closer look at how we implemented the nonlinear transformations between the linear layers. 

You might think that the instances of **reLU** in **n.Add(reLU)** above refer to a particular layer structure previously implemented as a layer module within the library. They don't. **reLU** is just a matrix-to-matrix elementwise function.

**An important thing to note** here is that the activation/transformation layers added with, for example, **n.Add(reLU)**, can be **any matrix-to-matrix function that you can express in the language,** unlike commonly seen machine learning frameworks where you are asked to select a particular layer type that has been implemented beforehand with it's (1) forward evaluation code and (2) reverse gradient code w.r.t. layer inputs, and (3) reverse gradient code w.r.t. any layer parameters. In such a setting, a new layer design would require you to add a new layer type to the system and carefully implement these components.

Here, because the system is based on nested AD, you can use any matrix-to-matrix transformation as a layer, and the forward and/or reverse AD operations of your code will be handled automatically by the underlying system. For example, you can write a layer like this: 
*)

n.Add(fun w ->
        let min = DM.Min(w)
        let range = DM.Max(w) - min
        (w - min) / range)

(** 
which will be a normalization layer, scaling the values to be between 0 and 1.

In the above model, this is how the softmax layer is implemented as a mapping of the vector-to-vector **softmax** function to the columns of a matrix. 

*)

n.Add(fun m -> m |> DM.mapCols softmax) 

(**
In this particular example, the output matrix has 10 rows (for the 10 target classes) and each column (a vector of size 10) is individually passed through the **softmax** function. The output matrix would have as many columns as the input matrix to the model, representing the class probabilities of each input.
*)


(**
### Weight initialization schemes

When layers with learnable weights are created, the weights are initialized using one of the following schemes. The correct initialization would depend on the activation function immediately following the layer and would take the fan-in/fan-out of the layer into account. If a specific scheme is not specified, the **InitStandard** scheme is used by default. These implementations are based on existing machine learning literature, such as _"Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." International conference on artificial intelligence and statistics. 2010"_.

*)

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

(**
### Training

Before training, let's visualize the weights of the first layer in a grid where each row of the weight matrix of the first layer is shown as a 28-by-28 image. It is an image of random weights, as expected.
*)

let l = (n.[0] :?> Linear)
l.VisualizeWRowsAsImageGrid(28) |> printfn "%s"

(**
<pre>
Hype.Neural.Linear
    784 -> 300
    Learnable parameters: 235500
    Init: ReLU
    W's rows reshaped to (28 x 28), presented in a (17 x 18) grid:
</pre>

<div class="row">
    <div class="span6 text-center">
        <img src="img/Feedforwardnets-1.png" alt="Chart" style="width:500px;"/>
    </div>
</div><br/>

Now let's train the network with the training and validation datasets we've prepared, using RMSProp, Nesterov momentum, and cross-entropy loss.
*)

let p = {Params.Default with 
            Epochs = 2
            EarlyStopping = Early (400, 100)
            ValidationInterval = 10
            Batch = Minibatch 100
            Loss = CrossEntropyOnSoftmax
            Momentum = Nesterov (D 0.9f)
            LearningRate = RMSProp (D 0.001f, D 0.9f)}

let _, lhist = n.Train(MNISTtrain, MNISTvalid, p)


(**
<pre>
[12/11/2015 22:42:07] --- Training started
[12/11/2015 22:42:07] Parameters     : 266610
[12/11/2015 22:42:07] Iterations     : 1180
[12/11/2015 22:42:07] Epochs         : 2
[12/11/2015 22:42:07] Batches        : Minibatches of 100 (590 per epoch)
[12/11/2015 22:42:07] Training data  : 59000
[12/11/2015 22:42:07] Validation data: 1000
[12/11/2015 22:42:07] Valid. interval: 10
[12/11/2015 22:42:07] Method         : Gradient descent
[12/11/2015 22:42:07] Learning rate  : RMSProp a0 = D 0.00100000005f, k = D 0.899999976f
[12/11/2015 22:42:07] Momentum       : Nesterov D 0.899999976f
[12/11/2015 22:42:07] Loss           : Cross entropy after softmax layer
[12/11/2015 22:42:07] Regularizer    : L2 lambda = D 9.99999975e-05f
[12/11/2015 22:42:07] Gradient clip. : None
[12/11/2015 22:42:07] Early stopping : Stagnation thresh. = 400, overfit. thresh. = 100
[12/11/2015 22:42:07] Improv. thresh.: D 0.995000005f
[12/11/2015 22:42:07] Return best    : true
[12/11/2015 22:42:07] 1/2 | Batch   1/590 | D  2.383214e+000 [- ] | Valid D  2.411374e+000 [- ] | Stag:  0 Ovfit:  0
[12/11/2015 22:42:08] 1/2 | Batch  11/590 | D  6.371681e-001 [↓▼] | Valid D  6.128169e-001 [↓▼] | Stag:  0 Ovfit:  0
[12/11/2015 22:42:08] 1/2 | Batch  21/590 | D  4.729548e-001 [↓▼] | Valid D  4.779414e-001 [↓▼] | Stag:  0 Ovfit:  0
[12/11/2015 22:42:09] 1/2 | Batch  31/590 | D  4.792733e-001 [↑ ] | Valid D  3.651254e-001 [↓▼] | Stag:  0 Ovfit:  0
[12/11/2015 22:42:10] 1/2 | Batch  41/590 | D  2.977416e-001 [↓▼] | Valid D  3.680202e-001 [↑ ] | Stag: 10 Ovfit:  0
[12/11/2015 22:42:10] 1/2 | Batch  51/590 | D  4.242567e-001 [↑ ] | Valid D  3.525212e-001 [↓▼] | Stag:  0 Ovfit:  0
[12/11/2015 22:42:11] 1/2 | Batch  61/590 | D  2.464822e-001 [↓▼] | Valid D  3.365663e-001 [↓▼] | Stag:  0 Ovfit:  0
[12/11/2015 22:42:11] 1/2 | Batch  71/590 | D  6.299557e-001 [↑ ] | Valid D  3.981607e-001 [↑ ] | Stag: 10 Ovfit:  0
...
[12/11/2015 22:43:21] 2/2 | Batch 521/590 | D  1.163270e-001 [↓ ] | Valid D  2.264248e-001 [↓ ] | Stag: 50 Ovfit:  0
[12/11/2015 22:43:21] 2/2 | Batch 531/590 | D  2.169427e-001 [↑ ] | Valid D  2.203927e-001 [↓ ] | Stag: 60 Ovfit:  0
[12/11/2015 22:43:22] 2/2 | Batch 541/590 | D  2.233351e-001 [↑ ] | Valid D  2.353653e-001 [↑ ] | Stag: 70 Ovfit:  0
[12/11/2015 22:43:22] 2/2 | Batch 551/590 | D  3.425132e-001 [↑ ] | Valid D  2.559682e-001 [↑ ] | Stag: 80 Ovfit:  0
[12/11/2015 22:43:23] 2/2 | Batch 561/590 | D  2.768238e-001 [↓ ] | Valid D  2.412431e-001 [↓ ] | Stag: 90 Ovfit:  0
[12/11/2015 22:43:24] 2/2 | Batch 571/590 | D  2.550858e-001 [↓ ] | Valid D  2.726600e-001 [↑ ] | Stag:100 Ovfit:  0
[12/11/2015 22:43:24] 2/2 | Batch 581/590 | D  2.308137e-001 [↓ ] | Valid D  2.466903e-001 [↓ ] | Stag:110 Ovfit:  0
[12/11/2015 22:43:25] Duration       : 00:01:17.5011734
[12/11/2015 22:43:25] Loss initial   : D  2.383214e+000
[12/11/2015 22:43:25] Loss final     : D  1.087980e-001 (Best)
[12/11/2015 22:43:25] Loss change    : D -2.274415e+000 (-95.43 %)
[12/11/2015 22:43:25] Loss chg. / s  : D -2.934685e-002
[12/11/2015 22:43:25] Epochs / s     : 0.02580606089
[12/11/2015 22:43:25] Epochs / min   : 1.548363654
[12/11/2015 22:43:25] --- Training finished
</pre>

<div class="row">
    <div class="span6 text-center">
        <img src="img/Feedforwardnets-3.png" alt="Chart" style="width:500px;"/>
    </div>
</div><br/>
*)

(*** hide ***)
open RProvider
open RProvider.graphics
open RProvider.grDevices

let ll = lhist |> Array.map (float32>>float)

namedParams[
    "x", box ll
    "pch", box 19
    "col", box "darkblue"
    "type", box "l"
    "xlab", box "Iteration"
    "ylab", box "Loss"
    "width", box 700
    "height", box 500
    ]
|> R.plot|> ignore


(**
Now let's visualize the weights of the first layer in the grid. We see that the network has learned the problem domain.
*)

let l = (n.[0] :?> Linear)
l.VisualizeWRowsAsImageGrid(28) |> printfn "%s"

(**

<div class="row">
    <div class="span6 text-center">
        <img src="img/Feedforwardnets-2.png" alt="Chart" style="width:500px;"/>
    </div>
</div><br/>
*)

(**

### Building the softmax classifier

As explained in [regression](regression.html), we just construct an instance of **SoftmaxClassifier** with the trained neural network as its parameter. Please see the [API reference](/reference/index.html) and the [source code](https://github.com/hypelib/Hype/blob/master/src/Hype/Classifier.fs) for a better understanding of how classifiers are implemented.
*)

let cc = SoftmaxClassifier(n)

(**

Testing class predictions for 10 random elements from the MNIST test set.

*)

let pred = cc.Classify(MNISTtest.X.[*,0..9]);;
let real = MNISTtest.Yi.[0..9]

(**
<pre>
val pred : int [] = [|5; 1; 9; 2; 6; 0; 0; 5; 7; 6|]
val real : int [] = [|5; 1; 9; 2; 6; 0; 0; 5; 7; 6|]
</pre>

Let's compute the classification error for the whole MNIST training set of 10,000 examples.
*)

cc.ClassificationError(MNISTtest)

(**
<pre>
val it : float32 = 0.0502999984f
</pre>

The classification error is around 5%. This can be lowered some more by training the model for more than 2 epochs as we did.

Classifying a single digit:
*)

let cls = cc.Classify(MNISTtest.X.[*,0]);;
MNISTtest.X.[*,0] |> DV.visualizeAsDM 28 |> printfn "%s"

(**
    [lang=cs]
    val cls : int = 5

    DM : 28 x 28
                            
                            
                            
                            
                            
                            ·   
                        ▴●██♦-  
                     ▴♦██■▴-    
                ♦█■■███▪·       
               ■████■-          
              ♦███▪             
             ♦██♦               
             ██●                
            ■█▪                 
            ██· -▴■●-           
           ▴██████■███-         
           ♦██♦▪    ▪█■-        
            ▪·       ▴█●        
                     -██        
                     ♦█●        
                    ■█■         
                 -●██■·         
             -▴▪■███▪           
          ███████●-             
                            
                            
                            

Classifying a many digits at the same time:
*)

let clss = cc.Classify(MNISTtest.X.[*,0..4]);;
MNISTtest.[0..4].VisualizeXColsAsImageGrid(28) |> printfn "%s"

(**

    [lang=cs]
    val clss : int [] = [|5; 1; 9; 2; 6|]

    Hype.Dataset
       X: 784 x 5
       Y: 1 x 5
    X's columns reshaped to (28 x 28), presented in a (2 x 3) grid:
    DM : 56 x 84
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                  ██♦                                   
                            ·                     ██                                    
                        ▴●██♦-                   ██▴                    -♦█▪            
                     ▴♦██■▴-                    ♦██                    ●█████●          
                ♦█■■███▪·                       ██♦                   ■███♦♦██          
               ■████■-                         ███                   ■██♦   ■█▴         
              ♦███▪                           ▴███                  ·██♦    ●██         
             ♦██♦                             ███                   ▪██     ■█■         
             ██●                             ▴██▴                   ·██·  ·♦██▴         
            ■█▪                              ███                     ███♦♦████▴         
            ██· -▴■●-                       ███♦                     ▴████████·         
           ▴██████■███-                     ███      ▴                ·-●- ■██          
           ♦██♦▪    ▪█■-                   ♦██▴                            ██■          
            ▪·       ▴█●                  ▴██♦                            -██▴          
                     -██                  ███▴                            -██·          
                     ♦█●                 ♦██▴                             ■██·          
                    ■█■                  ███                              ███           
                 -●██■·                 ♦██▴                             ▴██●           
             -▴▪■███▪                   ██♦                              ███            
          ███████●-                     ♦█                              -██■            
                                                                        -██♦            
                                                                        -██·            
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                ▴●█♦                                    
                ●██                           -████▴                                    
              ▪████●                         ▴████                                      
             ▴██████▴                       ▴███■                                       
             ■██▪▴██▴                       ███▪                                        
            ▴██●  ▴█■                      ■██▴                                         
           ·███    ██-                   ·♦██▴                                          
           ♦██●    ▪█▪                  -███▴                                           
           ███      ██                  ███♦                                            
           ███      █♦                 ███▪                                             
           █♦·      █♦                ●██■        ▴▴▴                                   
            ·       ██                ███    -██-■█████▪                                
              -     ██                ██■   ●███████████-                               
            ·██■♦-  ██               ▴██▴  ███●-     ▪██▴                               
            ♦█████■███               ▪██  ·██-       ·██▪                               
            ■█████████               ███▪·██▴        ♦██                                
            ♦█████████♦▪             ▪██████▴      ·♦██·                                
            -███████████■●●●·         ▪███████████████▴                                 
             ■██████■■■█████▴          -▪██████████♦-                                   
             ·████■    ▴████■              ·▴▴▴▴▴▴·                                     
              -■█-       ■■▴                                                            
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    




Nested optimization of training hyperparameters
-----------------------------------------------

As we've seen in [optimization](optimization.html), nested AD allows us to apply gradient-based optimization to functions that also internally perform optimization.

This gives us the possibility of optimizing the hyperparameters of training. We can, for example compute the gradient of the final loss of a training procedure with respect to the continuous hyperparameters of the training, such as learning rates, momentum parameters, regularization coefficients, or initialization conditions. 

As an example, let's train a neural network with a learning rate schedule of 50 elements, and optimize this schedule vector with another level of optimization on top of the training.
*)

let train lrschedule =
    Rnd.Seed(123)
    n.Init()

    let p = {Params.Default with
                LearningRate = Schedule lrschedule
                Loss = CrossEntropyOnSoftmax
                ValidationInterval = 1
                Silent = true
                ReturnBest = false
                Batch = Full}
    let loss, _ = n.Train(MNISTvalid.[..20], p)
    loss

let hypertrain epochs =
    let p = {Params.Default with 
                Epochs = epochs
                LearningRate = RMSProp(D 0.01f, D 0.9f)
                ValidationInterval = 1}
    let lr, _, _, _ = Optimize.Minimize(train, DV.create 50 (D 0.1f), p)
    lr

let lr = hypertrain 50

(*** hide ***)
open RProvider
open RProvider.graphics
open RProvider.grDevices

let lrlr = lr |> DV.toArray |> Array.map (float32>>float)

namedParams[
    "x", box lrlr
    "pch", box 19
    "col", box "darkblue"
    "type", box "o"
    "xlab", box "Iteration"
    "ylab", box "Learning rate"
    "width", box 700
    "height", box 500
    ]
|> R.plot|> ignore

(**
<div class="row">
    <div class="span6 text-center">
        <img src="img/Feedforwardnets-4.png" alt="Chart" style="width:500px;"/>
    </div>
</div><br/>
*)