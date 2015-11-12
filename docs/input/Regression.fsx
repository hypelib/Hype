(*** hide ***)
#r "../../src/Hype/bin/Release/DiffSharp.dll"
#r "../../src/Hype/bin/Release/Hype.dll"
#I "../../packages/R.NET.Community.1.6.4/lib/net40/"
#I "../../packages/R.NET.Community.FSharp.1.6.4/lib/net40/"
#I "../../packages/RProvider.1.1.14"
#load "RProvider.fsx"
fsi.ShowDeclarationValues <- true

(**
Regression
==========

In this example we implement a logistic regression based binary classifier and train it to distinguish between the MNIST digits of 0 and 1.

### Loading the data

First, let's start by loading the MNIST training and testing data and arranging these into training, validation, and testing sets.
*)

open Hype
open Hype.Neural
open DiffSharp.AD.Float32
open DiffSharp.Util

let MNIST = Dataset(Util.LoadMNIST("C:/datasets/MNIST/train-images.idx3-ubyte", 60000),
                    Util.LoadMNIST("C:/datasets/MNIST/train-labels.idx1-ubyte", 60000)).NormalizeX()

let MNISTtrain = MNIST.[..58999]
let MNISTvalid = MNIST.[59000..]

let MNISTtest = Dataset(Util.LoadMNIST("C:/datasets/MNIST/t10k-images.idx3-ubyte", 10000),
                        Util.LoadMNIST("C:/datasets/MNIST/t10k-labels.idx1-ubyte", 10000)).NormalizeX()

(**
We shuffle the columns of the datasets and filter them to only keep the digits of 0 and 1.
*)

let MNISTtrain01 = MNISTtrain.Shuffle().Filter(fun (x, y) -> y.[0] <= D 1.f)
let MNISTvalid01 = MNISTvalid.Shuffle().Filter(fun (x, y) -> y.[0] <= D 1.f)
let MNISTtest01 = MNISTtest.Shuffle().Filter(fun (x, y) -> y.[0] <= D 1.f)

(**
<pre>
val MNISTtrain01 : Dataset = Hype.Dataset
   X: 784 x 12465
   Y: 1 x 12465
val MNISTvalid01 : Dataset = Hype.Dataset
   X: 784 x 200
   Y: 1 x 200
val MNISTtest01 : Dataset = Hype.Dataset
   X: 784 x 2115
   Y: 1 x 2115
</pre>

We can visualize individual digits from the dataset.
*)

MNISTtrain.X.[*,9] |> DV.visualizeAsDM 28 |> printfn "%s"
MNISTtrain.Y.[*,9]

(**
    [lang=cs]
    DM : 28 x 28
                            
                            
                            
                            
                          ♦♦    
                         ▪█▪    
                        ▴██·    
                        ♦█♦     
                 ●     ·█■      
                ■█     ■█·      
                ♦█     ██·      
               ▴█■    ●█♦       
               ■█    ▪█■        
              ■█▪   ▴██-        
            -███♦▴  ♦█▪         
           ·███■██♦■█■          
          ·██■  ♦█████■         
          ♦■-    ♦████▪         
          -      ██·            
                ▪█■             
               ▴█■              
               ■█▴              
              ■█▪               
              ▴█·               
                            
                            
                            
                            

    val it : DV = DV [|4.0f|]

We can also visualize a series of digits in grid layout.
*)

MNISTtrain.[..5].VisualizeXColsAsImageGrid(28) |> printfn "%s"

(**
    [lang=cs]
    Hype.Dataset
       X: 784 x 6
       Y: 1 x 6
    X's columns reshaped to (28 x 28), presented in a (2 x 3) grid:
    DM : 56 x 84
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                ▪█▪                                     
                    ▴▴● ●██▴                   █████                             ■      
              -▪●█████■●██♦                   ■███■█             ·              ▴●      
            ██████████-··                    ■███♦·██▴          ▴●              ▪♦      
            ■█████♦●██                     ●██████-♦█●          ■●              █▪      
            ·▪-██♦   ▪                     ███♦-█■ ·█●          ■●             ●█▴      
               ▪█·                        ███● ·▴   ██          █●             ♦█       
               ▴█♦                       ●█■♦·      ██●        ▴█●             ■█       
                ♦█·                     ●██·        ██♦        ▪█▴            ●█■       
                 █■▪-                   ██          ██♦        ▪█           ·●██·       
                 ·███▴                 ♦█♦          ██♦        ▪█·     ▴▪▪███●██        
                   ●██▪               ·██-          ██▪        ▪██♦♦♦████♦▪·  ■█        
                    -██♦              ·█■          ▴█●          ▴●●●●●-      -█■        
                     ███              ·█■         ▴█■·                       ●█▴        
                   ▴●██♦              ·█▪        ●█●                         ●█         
                 ▪■████●              ·█■      -██▪                          ●█         
               -■████♦·               ·██▪  ·●■█■●                           ●█-        
              ■████♦·                 ·███■■███♦▴                            ●█-        
           ●■████♦·                    ♦█████■▪                              ●█▪        
         ●■█████▴                       ▴███▪                                ●█▪        
        ▴███■▴▴                                                              -█▪        
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                      ▴██                                                    -▴         
                     -███                                                 ▪♦███▪        
                     ▴███                    ▪♦██-·▪                    ▪███■■█■        
                     ██■                   ·■██♦♦███●                 ▪████■  ██        
                    ■██-                   ██♦   ●██▴                -████■   ██        
                   ▪██♦                  -██●   ·██■                 ●██■●   ·██        
                   ███                  ▴██▪    ■██·                  ▴      -██        
                  ♦██▴                 ▴██●    ·██▴                          ▪██        
                 -██●                  ■█●    ♦██●                       -▴▴▴♦█♦        
                ·██♦                   ██  ▴♦████·                      ●█████■         
                ███▪                   ■█████■■█■                     ■██■●███■▴        
               ▪███                     ██■▴  ♦█▪                   ·██▴   ♦████·       
               ■██●                           ██-                  ▴██●   ♦██▴●██●      
              ███♦                           ·██                  ▴██-   ♦█■   ·●███■   
              ███·                            ██                 -██· ·●██▴      ·●●    
             ▪███                            ·██                 ■██♦■███▴              
             ■██▪                            -██                 ♦████●▴                
             ██■                              ██                  -▪▴                   
             ██■                              ■█                                        
             ♦█■                              -█♦                                       
                                               ●█●                                      
                                                ▪█                                      
                                                                                    
*)

MNISTtrain01.[..5].VisualizeXColsAsImageGrid(28) |> printfn "%s"

(**
    [lang=cs]
    Hype.Dataset
       X: 784 x 6
       Y: 1 x 6
    X's columns reshaped to (28 x 28), presented in a (2 x 3) grid:
    DM : 56 x 84
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                     ▴●███-                      ·♦██                                   
                  ·▪■█████■                      ████●                   -▪██████       
                 ▪████████-                     ●████■                 ▪██████████      
               ●████▪ ●███▴                     ■████●                ■███■▪▪▴-███●     
              ████●   ▴███·                    ██████               -███■     ██████    
             ███▪      ██▴                    -██████               ■██▴      ████▴♦-   
            ■██·       ●█●                   ▪█████■               ■██        ■█▴█■     
           ●██·        ▴█●                   ██████               ▪██●         - ■█-    
          -██▪         ▴█●                   ██████               ██■            ▪█▴    
          ██■          ▴█●                  ■█████▴              ▴██-            ■█▴    
         ·██           ▴█●                 ██████-               ▴██             ██▴    
         ■█■           ▴█●                 █████▴                ▴█■             ██     
         ■█▪           ▴█●                ♦████-                 ▴█▪            ▴█▪     
         ██-           ●█●               ♦█████                  ▴█▪            ●█·     
         ██▴           ██·               █████♦                  -██           -█▪      
         ███          ●██·              ●█████·                   ██■           █-      
         -███●       ▴██●              ·█████                     ███          ■■       
           ■█████♦●●■███■              ♦█████                      ██-       -■█·       
            ▴■█████████♦               █████♦                      ●██●    -■██▪        
               ·▪▪●█♦▪▴                ▴████·                       ●████████■-         
                                                                     ▪█████▪-           
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                ■▴                         ·····        
                                                ▴■                       ▪■█████·       
               ♦♦♦♦♦-·                          ■█                     ▴█████████-      
             ▪████████■▪                        ██                     ███████████·     
             ■███████████■                     ▪█■                   ▴■████♦♦ ▴████     
            █████████  ♦███                    ██-                  ▴████♦▴    ■███·    
          ▪███●     ●   ·██■                   █●                  ▴████■      ♦███-    
          ██■            -██·                 ▪█▴                  ████▪        ████    
         ▪██              ██♦                 ██                   ███-         ████    
         ●█♦              ▪██                ▪██                 ·███·          ████    
         ██▴              ▴██                ●██                ·███♦          ▪████    
         ██               ♦█▪                ██●                ·███          ▪████·    
         ██-              ██                 ██·                ●███        ·●█████·    
         ♦█●             ■██                ·██-                ████·      ■██████▴     
         ▪█■            ■██-                 █♦                 ████■▴▴▴■████████▪      
         -██●         ▴███♦                 -█♦                 ■██████████████♦        
          ♦██♦-▪    ▪♦██■-                  ■█♦                 ·████████████▪▴         
           ♦███████████●                    ■█-                   -█████████▴           
            ▴■█■■■■■■·                      █■                       ▪ ▪▪               
                                            ♦▴                                          
                                                                                    
                                                                                    
                                                                                    
                                                                                    
### Defining the model

Let's now create our linear regression model. We implement this using the **Hype.Neural** module, as a linear layer with $28 \times 28 = 784$ inputs and one output. The output of the layer is passed through the sigmoid function.

*)

let n = Neural.FeedForward()
n.Add(Linear(28 * 28, 1))
n.Add(sigmoid)

(**

We can visualize the initial state of the linear model weights before the training. For information of about weight initialization parameters, please see the [neural networks example](feedforwardnets.html).

*)

let l = (n.[0] :?> Linear)
l.VisualizeWRowsAsImageGrid(28) |> printfn "%s"

(** 

    [lang=cs]
    Hype.Neural.Linear
       784 -> 1
       Learnable parameters: 785
       Init: Standard
       W's rows reshaped to (28 x 28), presented in a (1 x 1) grid:
    DM : 28 x 28
     ▴▪●●-█▴♦♦● ·▴█● ● ▴· ●●●●▪·
    ■ █- ▴●●▪ ■♦· ■▪■▪   █  ♦■●■
    ♦■ █♦●▪●♦  ♦■   ♦     ■ ▪- ■
     ■▪ ■♦■♦ █ ▪● ♦▪▴··■█ -▴●▪▪●
    ██··▴●●█▪♦■ -·█■ ▪- ··▪·  ██
    - ▪   ♦ ▪●  ▪■█♦- ▴▪ ▴·  ▪·●
    -   ●●▴▴ ▪■ ▴█ ▪▴·▴▴·♦■■♦·■■
    ♦▴ ▪■ ▪▪▴■·■--▪♦-   ·♦▪■ ♦·●
     ·▴·♦▪♦●▪··▴·▪ ● ▪ █  ▴▪·♦▪ 
    ■ ▴ ♦█▴ -  ♦●■  █▪■●▪█■▴●--█
    ♦■   ●■▴♦ ●· █· ▴· -█-▪●■■-■
     █-·▪▴-▴█ ♦ █●·♦█▪▪●●■ -   ·
     -   █ ■♦·●▪▴♦ -▴ -  ■♦· ♦ -
    ■█ ▪-  ▪■●♦█▴-█▪■  ■♦▪█■▪■ -
    ●♦█▴♦♦ ♦   ▴▪▴▴♦-▴♦♦█ ▴ ▪·● 
     ·█▪■■█ ●· ●· -●■●··  ▴  --▴
    ·♦█▴ ♦♦■ ▴▪●▪-  · -♦●♦ ■ · ■
    ■■▪---♦■·●▴▪-▪▴· ▪●● ·♦■ ▪♦▴
    ▴ -♦●■█·█   ● ♦▪●■- ·■♦-▪▴■▴
     ●-■● ···●█▴▪ -█·▪ ♦▴    ● ●
    ·█  █▴ ·♦---■▴·█■■▴ ▴■  -  █
    - ▪  ●█·▴♦▪    ■ ▪■ ■···   ▴
    ■ ♦♦- █▪♦-- ▴ ▴ ··█▴● ■♦    
    ■·■■▪▴-·█♦●■ ▴ ♦ ♦▴■♦  ■ ●♦▪
    ·█▪- ■●▴▪▴▪ ▪  ▴▪ ·   ▪▴▴··♦
      ▪█♦■   ·♦ ■▪ ♦ ▴·●█▪· ·▪▴ 
    · ■♦▪■ ▪■● ♦  ··· ·▪█■·  ▪■●
    ●▴▪ ·■● -█●█·▪■▴ ▴▴♦  ■  ■ ▴

       b:
    DV : 1
     

### Training

Let's train the model for 10 epochs (full passes through the training data), with a minibatch size of 100, using the training and validation sets we've defined. The validation set will make sure that we're not overfitting the model.
*)

let p = {Params.Default with 
            Epochs = 10; 
            Batch = Minibatch 100; 
            EarlyStopping = EarlyStopping.DefaultEarly}

n.Train(MNISTtrain01, MNISTvalid01, p)

(**
<pre>
[12/11/2015 20:21:12] --- Training started
[12/11/2015 20:21:12] Parameters     : 785
[12/11/2015 20:21:12] Iterations     : 1240
[12/11/2015 20:21:12] Epochs         : 10
[12/11/2015 20:21:12] Batches        : Minibatches of 100 (124 per epoch)
[12/11/2015 20:21:12] Training data  : 12465
[12/11/2015 20:21:12] Validation data: 200
[12/11/2015 20:21:12] Valid. interval: 10
[12/11/2015 20:21:12] Method         : Gradient descent
[12/11/2015 20:21:12] Learning rate  : RMSProp a0 = D 0.00100000005f, k = D 0.899999976f
[12/11/2015 20:21:12] Momentum       : None
[12/11/2015 20:21:12] Loss           : L2 norm
[12/11/2015 20:21:12] Regularizer    : L2 lambda = D 9.99999975e-05f
[12/11/2015 20:21:12] Gradient clip. : None
[12/11/2015 20:21:12] Early stopping : Stagnation thresh. = 750, overfit. thresh. = 10
[12/11/2015 20:21:12] Improv. thresh.: D 0.995000005f
[12/11/2015 20:21:12] Return best    : true
[12/11/2015 20:21:12]  1/10 | Batch   1/124 | D  4.748471e-001 [- ] | Valid D  4.866381e-001 [- ] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch  11/124 | D  2.772053e-001 [↓▼] | Valid D  3.013612e-001 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch  21/124 | D  2.178165e-001 [↓▼] | Valid D  2.304372e-001 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch  31/124 | D  2.009703e-001 [↓▼] | Valid D  1.799015e-001 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch  41/124 | D  1.352896e-001 [↓▼] | Valid D  1.405802e-001 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch  51/124 | D  1.182899e-001 [↓▼] | Valid D  1.108390e-001 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch  61/124 | D  1.124191e-001 [↓▼] | Valid D  8.995526e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch  71/124 | D  8.975799e-002 [↓▼] | Valid D  7.361954e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch  81/124 | D  5.031444e-002 [↓▼] | Valid D  5.941865e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch  91/124 | D  5.063754e-002 [↑ ] | Valid D  4.927430e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch 101/124 | D  3.842642e-002 [↓▼] | Valid D  4.095582e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch 111/124 | D  4.326219e-002 [↑ ] | Valid D  3.452797e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  1/10 | Batch 121/124 | D  2.585407e-002 [↓▼] | Valid D  2.788338e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  2/10 | Batch   1/124 | D  3.069563e-002 [↑ ] | Valid D  2.663207e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  2/10 | Batch  11/124 | D  1.765305e-002 [↓▼] | Valid D  2.332163e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  2/10 | Batch  21/124 | D  2.314118e-002 [↑ ] | Valid D  1.902804e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  2/10 | Batch  31/124 | D  3.177435e-002 [↑ ] | Valid D  1.691620e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  2/10 | Batch  41/124 | D  2.219648e-002 [↓ ] | Valid D  1.455527e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  2/10 | Batch  51/124 | D  1.205402e-002 [↓▼] | Valid D  1.240637e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  2/10 | Batch  61/124 | D  3.891717e-002 [↑ ] | Valid D  1.189688e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  2/10 | Batch  71/124 | D  2.114762e-002 [↓ ] | Valid D  1.083007e-002 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  2/10 | Batch  81/124 | D  5.075417e-003 [↓▼] | Valid D  9.630994e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:12]  2/10 | Batch  91/124 | D  1.343214e-002 [↑ ] | Valid D  8.666289e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  2/10 | Batch 101/124 | D  6.054885e-003 [↓ ] | Valid D  8.039203e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  2/10 | Batch 111/124 | D  1.964125e-002 [↑ ] | Valid D  7.339509e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  2/10 | Batch 121/124 | D  4.401092e-003 [↓▼] | Valid D  6.376633e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch   1/124 | D  7.068173e-003 [↑ ] | Valid D  6.426438e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch  11/124 | D  3.763680e-003 [↓▼] | Valid D  6.076077e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch  21/124 | D  9.855231e-003 [↑ ] | Valid D  5.091224e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch  31/124 | D  1.263964e-002 [↑ ] | Valid D  4.641499e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch  41/124 | D  1.205439e-002 [↓ ] | Valid D  4.599225e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch  51/124 | D  2.941387e-003 [↓▼] | Valid D  4.381890e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch  61/124 | D  2.546543e-002 [↑ ] | Valid D  4.439059e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch  71/124 | D  9.878366e-003 [↓ ] | Valid D  4.358966e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch  81/124 | D  1.868963e-003 [↓▼] | Valid D  3.960044e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch  91/124 | D  7.171181e-003 [↑ ] | Valid D  3.634899e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch 101/124 | D  2.681098e-003 [↓ ] | Valid D  3.636524e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch 111/124 | D  1.502046e-002 [↑ ] | Valid D  3.393996e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  3/10 | Batch 121/124 | D  2.381395e-003 [↓ ] | Valid D  3.178693e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  4/10 | Batch   1/124 | D  3.185510e-003 [↑ ] | Valid D  3.240891e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:13]  4/10 | Batch  11/124 | D  2.029225e-003 [↓ ] | Valid D  3.163968e-003 [↓ ] | Stag: 20 Ovfit: 0
[12/11/2015 20:21:13]  4/10 | Batch  21/124 | D  6.450378e-003 [↑ ] | Valid D  2.772849e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  4/10 | Batch  31/124 | D  7.448227e-003 [↑ ] | Valid D  2.572560e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:13]  4/10 | Batch  41/124 | D  9.700718e-003 [↑ ] | Valid D  2.693694e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:13]  4/10 | Batch  51/124 | D  1.799919e-003 [↓▼] | Valid D  2.737873e-003 [↑ ] | Stag: 20 Ovfit: 1
[12/11/2015 20:21:13]  4/10 | Batch  61/124 | D  1.919956e-002 [↑ ] | Valid D  2.778393e-003 [↑ ] | Stag: 30 Ovfit: 3
[12/11/2015 20:21:13]  4/10 | Batch  71/124 | D  5.462923e-003 [↓ ] | Valid D  2.870561e-003 [↑ ] | Stag: 40 Ovfit: 3
[12/11/2015 20:21:13]  4/10 | Batch  81/124 | D  1.455469e-003 [↓▼] | Valid D  2.632472e-003 [↓ ] | Stag: 50 Ovfit: 4
[12/11/2015 20:21:14]  4/10 | Batch  91/124 | D  5.270801e-003 [↑ ] | Valid D  2.455564e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:14]  4/10 | Batch 101/124 | D  2.057914e-003 [↓ ] | Valid D  2.511977e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:14]  4/10 | Batch 111/124 | D  1.314815e-002 [↑ ] | Valid D  2.393763e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:14]  4/10 | Batch 121/124 | D  2.033168e-003 [↓ ] | Valid D  2.358985e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:14]  5/10 | Batch   1/124 | D  2.199435e-003 [↑ ] | Valid D  2.389120e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:14]  5/10 | Batch  11/124 | D  1.668178e-003 [↓ ] | Valid D  2.356529e-003 [↓ ] | Stag: 20 Ovfit: 0
[12/11/2015 20:21:14]  5/10 | Batch  21/124 | D  5.649061e-003 [↑ ] | Valid D  2.151499e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:14]  5/10 | Batch  31/124 | D  5.264180e-003 [↓ ] | Valid D  2.038927e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:14]  5/10 | Batch  41/124 | D  8.416546e-003 [↑ ] | Valid D  2.145057e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:14]  5/10 | Batch  51/124 | D  1.564733e-003 [↓ ] | Valid D  2.208556e-003 [↑ ] | Stag: 20 Ovfit: 0
[12/11/2015 20:21:14]  5/10 | Batch  61/124 | D  1.581773e-002 [↑ ] | Valid D  2.233998e-003 [↑ ] | Stag: 30 Ovfit: 0
[12/11/2015 20:21:14]  5/10 | Batch  71/124 | D  3.898179e-003 [↓ ] | Valid D  2.347554e-003 [↑ ] | Stag: 40 Ovfit: 0
[12/11/2015 20:21:14]  5/10 | Batch  81/124 | D  1.395002e-003 [↓▼] | Valid D  2.182974e-003 [↓ ] | Stag: 50 Ovfit: 1
[12/11/2015 20:21:14]  5/10 | Batch  91/124 | D  4.450763e-003 [↑ ] | Valid D  2.069927e-003 [↓ ] | Stag: 60 Ovfit: 1
[12/11/2015 20:21:14]  5/10 | Batch 101/124 | D  1.927794e-003 [↓ ] | Valid D  2.129479e-003 [↑ ] | Stag: 70 Ovfit: 1
[12/11/2015 20:21:14]  5/10 | Batch 111/124 | D  1.238949e-002 [↑ ] | Valid D  2.059099e-003 [↓ ] | Stag: 80 Ovfit: 1
[12/11/2015 20:21:14]  5/10 | Batch 121/124 | D  1.969593e-003 [↓ ] | Valid D  2.072177e-003 [↑ ] | Stag: 90 Ovfit: 1
[12/11/2015 20:21:14]  6/10 | Batch   1/124 | D  1.885590e-003 [↓ ] | Valid D  2.087292e-003 [↑ ] | Stag:100 Ovfit: 1
[12/11/2015 20:21:14]  6/10 | Batch  11/124 | D  1.577425e-003 [↓ ] | Valid D  2.074389e-003 [↓ ] | Stag:110 Ovfit: 1
[12/11/2015 20:21:14]  6/10 | Batch  21/124 | D  5.410788e-003 [↑ ] | Valid D  1.943973e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:14]  6/10 | Batch  31/124 | D  4.188792e-003 [↓ ] | Valid D  1.863442e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:14]  6/10 | Batch  41/124 | D  7.516511e-003 [↑ ] | Valid D  1.951990e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:14]  6/10 | Batch  51/124 | D  1.510475e-003 [↓ ] | Valid D  2.003860e-003 [↑ ] | Stag: 20 Ovfit: 0
[12/11/2015 20:21:14]  6/10 | Batch  61/124 | D  1.375423e-002 [↑ ] | Valid D  2.020531e-003 [↑ ] | Stag: 30 Ovfit: 0
[12/11/2015 20:21:14]  6/10 | Batch  71/124 | D  3.260145e-003 [↓ ] | Valid D  2.129138e-003 [↑ ] | Stag: 40 Ovfit: 0
[12/11/2015 20:21:15]  6/10 | Batch  81/124 | D  1.402565e-003 [↓ ] | Valid D  2.002138e-003 [↓ ] | Stag: 50 Ovfit: 0
[12/11/2015 20:21:15]  6/10 | Batch  91/124 | D  3.999386e-003 [↑ ] | Valid D  1.920336e-003 [↓ ] | Stag: 60 Ovfit: 0
[12/11/2015 20:21:15]  6/10 | Batch 101/124 | D  1.929424e-003 [↓ ] | Valid D  1.976652e-003 [↑ ] | Stag: 70 Ovfit: 0
[12/11/2015 20:21:15]  6/10 | Batch 111/124 | D  1.205915e-002 [↑ ] | Valid D  1.926643e-003 [↓ ] | Stag: 80 Ovfit: 0
[12/11/2015 20:21:15]  6/10 | Batch 121/124 | D  1.978536e-003 [↓ ] | Valid D  1.951888e-003 [↑ ] | Stag: 90 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch   1/124 | D  1.769614e-003 [↓ ] | Valid D  1.959661e-003 [↑ ] | Stag:100 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch  11/124 | D  1.555518e-003 [↓ ] | Valid D  1.955613e-003 [↓ ] | Stag:110 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch  21/124 | D  5.217655e-003 [↑ ] | Valid D  1.861573e-003 [↓ ] | Stag:120 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch  31/124 | D  3.625835e-003 [↓ ] | Valid D  1.796666e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch  41/124 | D  6.929778e-003 [↑ ] | Valid D  1.872346e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch  51/124 | D  1.502809e-003 [↓ ] | Valid D  1.913079e-003 [↑ ] | Stag: 20 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch  61/124 | D  1.241405e-002 [↑ ] | Valid D  1.924762e-003 [↑ ] | Stag: 30 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch  71/124 | D  2.962820e-003 [↓ ] | Valid D  2.024504e-003 [↑ ] | Stag: 40 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch  81/124 | D  1.421725e-003 [↓ ] | Valid D  1.919308e-003 [↓ ] | Stag: 50 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch  91/124 | D  3.717377e-003 [↑ ] | Valid D  1.854433e-003 [↓ ] | Stag: 60 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch 101/124 | D  1.973184e-003 [↓ ] | Valid D  1.907719e-003 [↑ ] | Stag: 70 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch 111/124 | D  1.190252e-002 [↑ ] | Valid D  1.867085e-003 [↓ ] | Stag: 80 Ovfit: 0
[12/11/2015 20:21:15]  7/10 | Batch 121/124 | D  2.006255e-003 [↓ ] | Valid D  1.894716e-003 [↑ ] | Stag: 90 Ovfit: 0
[12/11/2015 20:21:15]  8/10 | Batch   1/124 | D  1.721533e-003 [↓ ] | Valid D  1.898627e-003 [↑ ] | Stag:100 Ovfit: 0
[12/11/2015 20:21:15]  8/10 | Batch  11/124 | D  1.553262e-003 [↓ ] | Valid D  1.897926e-003 [↓ ] | Stag:110 Ovfit: 0
[12/11/2015 20:21:15]  8/10 | Batch  21/124 | D  5.004487e-003 [↑ ] | Valid D  1.823838e-003 [↓ ] | Stag:120 Ovfit: 0
[12/11/2015 20:21:15]  8/10 | Batch  31/124 | D  3.308986e-003 [↓ ] | Valid D  1.768821e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:15]  8/10 | Batch  41/124 | D  6.563510e-003 [↑ ] | Valid D  1.835302e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:15]  8/10 | Batch  51/124 | D  1.507999e-003 [↓ ] | Valid D  1.868091e-003 [↑ ] | Stag: 20 Ovfit: 0
[12/11/2015 20:21:15]  8/10 | Batch  61/124 | D  1.148601e-002 [↑ ] | Valid D  1.876653e-003 [↑ ] | Stag: 30 Ovfit: 0
[12/11/2015 20:21:16]  8/10 | Batch  71/124 | D  2.807777e-003 [↓ ] | Valid D  1.968064e-003 [↑ ] | Stag: 40 Ovfit: 0
[12/11/2015 20:21:16]  8/10 | Batch  81/124 | D  1.440011e-003 [↓ ] | Valid D  1.876611e-003 [↓ ] | Stag: 50 Ovfit: 0
[12/11/2015 20:21:16]  8/10 | Batch  91/124 | D  3.522004e-003 [↑ ] | Valid D  1.821817e-003 [↓ ] | Stag: 60 Ovfit: 0
[12/11/2015 20:21:16]  8/10 | Batch 101/124 | D  2.031282e-003 [↓ ] | Valid D  1.872902e-003 [↑ ] | Stag: 70 Ovfit: 0
[12/11/2015 20:21:16]  8/10 | Batch 111/124 | D  1.182362e-002 [↑ ] | Valid D  1.836957e-003 [↓ ] | Stag: 80 Ovfit: 0
[12/11/2015 20:21:16]  8/10 | Batch 121/124 | D  2.035742e-003 [↓ ] | Valid D  1.864137e-003 [↑ ] | Stag: 90 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch   1/124 | D  1.699795e-003 [↓ ] | Valid D  1.865989e-003 [↑ ] | Stag:100 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch  11/124 | D  1.556397e-003 [↓ ] | Valid D  1.866347e-003 [↑ ] | Stag:110 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch  21/124 | D  4.788828e-003 [↑ ] | Valid D  1.804229e-003 [↓ ] | Stag:120 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch  31/124 | D  3.119682e-003 [↓ ] | Valid D  1.756223e-003 [↓▼] | Stag:  0 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch  41/124 | D  6.336636e-003 [↑ ] | Valid D  1.816257e-003 [↑ ] | Stag: 10 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch  51/124 | D  1.516153e-003 [↓ ] | Valid D  1.843593e-003 [↑ ] | Stag: 20 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch  61/124 | D  1.080968e-002 [↑ ] | Valid D  1.850113e-003 [↑ ] | Stag: 30 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch  71/124 | D  2.720124e-003 [↓ ] | Valid D  1.934669e-003 [↑ ] | Stag: 40 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch  81/124 | D  1.455176e-003 [↓ ] | Valid D  1.852409e-003 [↓ ] | Stag: 50 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch  91/124 | D  3.375944e-003 [↑ ] | Valid D  1.804057e-003 [↓ ] | Stag: 60 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch 101/124 | D  2.093168e-003 [↓ ] | Valid D  1.853583e-003 [↑ ] | Stag: 70 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch 111/124 | D  1.178356e-002 [↑ ] | Valid D  1.820183e-003 [↓ ] | Stag: 80 Ovfit: 0
[12/11/2015 20:21:16]  9/10 | Batch 121/124 | D  2.061530e-003 [↓ ] | Valid D  1.846045e-003 [↑ ] | Stag: 90 Ovfit: 0
[12/11/2015 20:21:16] 10/10 | Batch   1/124 | D  1.689459e-003 [↓ ] | Valid D  1.846794e-003 [↑ ] | Stag:100 Ovfit: 0
[12/11/2015 20:21:16] 10/10 | Batch  11/124 | D  1.560583e-003 [↓ ] | Valid D  1.847311e-003 [↑ ] | Stag:110 Ovfit: 0
[12/11/2015 20:21:16] 10/10 | Batch  21/124 | D  4.588457e-003 [↑ ] | Valid D  1.792883e-003 [↓ ] | Stag:120 Ovfit: 0
[12/11/2015 20:21:16] 10/10 | Batch  31/124 | D  3.001853e-003 [↓ ] | Valid D  1.750141e-003 [↓ ] | Stag:130 Ovfit: 0
[12/11/2015 20:21:16] 10/10 | Batch  41/124 | D  6.195725e-003 [↑ ] | Valid D  1.805622e-003 [↑ ] | Stag:140 Ovfit: 0
[12/11/2015 20:21:16] 10/10 | Batch  51/124 | D  1.524289e-003 [↓ ] | Valid D  1.829196e-003 [↑ ] | Stag:150 Ovfit: 0
[12/11/2015 20:21:17] 10/10 | Batch  61/124 | D  1.029841e-002 [↑ ] | Valid D  1.834366e-003 [↑ ] | Stag:160 Ovfit: 0
[12/11/2015 20:21:17] 10/10 | Batch  71/124 | D  2.667856e-003 [↓ ] | Valid D  1.913492e-003 [↑ ] | Stag:170 Ovfit: 0
[12/11/2015 20:21:17] 10/10 | Batch  81/124 | D  1.467351e-003 [↓ ] | Valid D  1.837669e-003 [↓ ] | Stag:180 Ovfit: 0
[12/11/2015 20:21:17] 10/10 | Batch  91/124 | D  3.261143e-003 [↑ ] | Valid D  1.793646e-003 [↓ ] | Stag:190 Ovfit: 0
[12/11/2015 20:21:17] 10/10 | Batch 101/124 | D  2.153974e-003 [↓ ] | Valid D  1.842048e-003 [↑ ] | Stag:200 Ovfit: 0
[12/11/2015 20:21:17] 10/10 | Batch 111/124 | D  1.176465e-002 [↑ ] | Valid D  1.810117e-003 [↓ ] | Stag:210 Ovfit: 0
[12/11/2015 20:21:17] 10/10 | Batch 121/124 | D  2.082179e-003 [↓ ] | Valid D  1.834467e-003 [↑ ] | Stag:220 Ovfit: 0
[12/11/2015 20:21:17] Duration       : 00:00:05.2093910
[12/11/2015 20:21:17] Loss initial   : D  4.748471e-001
[12/11/2015 20:21:17] Loss final     : D  1.395002e-003 (Best)
[12/11/2015 20:21:17] Loss change    : D -4.734521e-001 (-99.71 %)
[12/11/2015 20:21:17] Loss chg. / s  : D -9.088434e-002
[12/11/2015 20:21:17] Epochs / s     : 1.919610181
[12/11/2015 20:21:17] Epochs / min   : 115.1766109
[12/11/2015 20:21:17] --- Training finished

</pre>

After a 5-second training, we can see that the characteristics of the problem domain (distinguishing between the digits of 0 and 1) is captured in the model weights.

*)

let l = (n.[0] :?> Linear)
l.VisualizeWRowsAsImageGrid(28) |> printfn "%s"

(**
    [lang=cs]
    Hype.Neural.Linear
       784 -> 1
       Learnable parameters: 785
       Init: Standard
       W's rows reshaped to (28 x 28), presented in a (1 x 1) grid:
    DM : 28 x 28
    ----------------------------
    ----------------------------
    ------------▴▴▴▴▴-----------
    ---------▴--▴▴▴▴▴▴-▴--------
    --------▴▴▪▴▪▪▪▴-▴▴▴▪▪▪▴----
    ------▴-▴▴▴▴▪▪▴▴-·-▴▪▪▪▪▴---
    ------▴--▴▴-▴▴--···▴▪▴▴▴▴---
    ----▴---------▴▴-· ---------
    ---------··---▴▪--·-·····---
    -------······▴▪▪▪▴-······---
    ------·····  ▴●●●▴·     ·---
    -----·· ·    ▪♦■♦▪      ·---
    -----· ·     ●■■♦▴      ·---
    -----·      ·♦██♦·      ·---
    -----·      ▴■██●       ·---
    ----·       ▪██■▪       ·---
    ----·      -●█■♦-       ·---
    ----·      ▴♦█■●·     ···---
    ----·     ·▴♦♦♦●·   ····----
    ----·    ·▴▪●●●▪·· ····--▴--
    ----······-▴▪▪▪▴--------▴---
    -----▴▴----·--▴▴-▴▴-▴▴------
    -----▴▪▪▴-· ··--▴▴▪▴▴▴▴-----
    ----▴▪▪▪▪▴-· ·-▴▴▪▴▴▴▴------
    -----▴▪▴▴▴▴·---▴▴▴▴▴--------
    ------------▴▴▴▴------------
    ----------------------------
    ----------------------------

       b:
    DV : 1
     
### Classifier

You can create classifiers by instantiating types such as **LogisticClassifier** or **SoftmaxClassifier**, and passing the classification function in the contructor. Alternatively, you can directly pass the model we have just trained. Please see the [API reference](/reference/index.html) and the [source code](https://github.com/hypelib/Hype/blob/master/src/Hype/Classifier.fs) for a better understanding of how classifiers are implemented.

*)

let cc = LogisticClassifier(n)

(**
Let's test the class predictions for 10 random elements from the MNIST test set, which we've filtered to have only 0s and 1s.
*)

let pred = cc.Classify(MNISTtest01.X.[*,0..9]);;
let real = MNISTtest01.Y.[*, 0..9] |> DM.toDV |> DV.toArray |> Array.map (float32>>int)

(**
<pre>
val pred : int [] = [|1; 0; 1; 0; 1; 0; 0; 1; 1; 1|]
val real : int [] = [|1; 0; 1; 0; 1; 0; 0; 1; 1; 1|]
</pre>

The classifier seems to be working well. We can compute the classification error for a given dataset.
*)

let error = cc.ClassificationError(MNISTtest01);;

(**
<pre>
val error : float32 = 0.000472813234f
</pre>

The classification error is 0.047%.

Finally, this is how you would classify single digits.
*)

let cls = cc.Classify(MNISTtest01.X.[*,0]);;
MNISTtest01.X.[*,0] |> DV.visualizeAsDM 28 |> printfn "%s"

(**
    [lang=cs]
    val cls : int = 1

    DM : 28 x 28
                            
                            
                            
                            
                ♦               
                ●♦              
                 █              
                 ■·             
                ▪█-             
                ▴█-             
                 ■♦             
                 ♦█·            
                 -█▪            
                  █▪            
                  ●▪            
                  ▪█            
                  ▪█-           
                  ▪█▴           
                  ▪█■           
                   █■           
                   ██           
                   ▪█           
                    █▴          
                    █●          
                            
                            
And this is how you would classify many digits efficiently at the same time, by running them through the model together as the columns of an input matrix.
*)

let clss = cc.Classify(MNISTtest01.X.[*,5..9]);;
MNISTtest01.[5..9].VisualizeXColsAsImageGrid(28) |> printfn "%s"

(**
    [lang=cs]
    val clss : int [] = [|0; 0; 1; 1; 1|]

    Hype.Dataset
       X: 784 x 5
       Y: 1 x 5
    X's columns reshaped to (28 x 28), presented in a (2 x 3) grid:
    DM : 56 x 84
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                 ██·                                                                    
                ●███♦-                        ·████♦·                   -█▴             
                ██████■-                     ♦███████-                  ▪██·            
               ●███████■                  ♦███████████▴                 ●██·            
              ▪███● -███-                ♦█████████████■                ▴♦█·            
              ♦██▪   -██■                ████♦ ●●████████                ▪█·            
             ▪███    ·██■               ●█████·  ··██████                ▪█·            
             ■██▪    ·██■              ▪██████·    ██████                ▪█·            
            ·██■     ▪██■             -██████♦     ██████                ▪█·            
            ♦██▪     ■██■            ▪███████     ·█████♦                ●█·            
           ·███-     ■██■            ██████♦·    ♦██████·                ██·            
           ♦██■     ·███■            ██████-   ▪███████-                 ██·            
           ■██-     -███-            ██████   ●███████▴                  ██·            
           ■██·     ■██♦             ██████·♦████████■                   ██·            
           ■██·    ■███●             ████████████████                    ██             
           ■██●    ████              ██████████████-                     ██             
           ▴███· -■███-              ▴████████████▴                     ·██             
           ·█████████♦                ■████████■●                        ██             
            ▴███████♦                  ████████                         ·█♦·            
              ▪███♦·                    ●●■●●-                          ·██▴            
                                                                         █■             
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                      ·██                                                               
                      ███                       -██■                                    
                     ●██■                      ▪████-                                   
                     ♦██                       ♦███■                                    
                    ·███                      ·████■                                    
                    ■██♦                      ▴███■                                     
                   ▪██■                      -████●                                     
                   ███-                      ▴████▪                                     
                  ▪██■                      ·████♦                                      
                 -███▴                      █████·                                      
                 ♦██●                       ████♦                                       
                -███                       ▪████·                                       
                ███▴                       ■███■                                        
               -███                       ▴████·                                        
               ███●                       ■███■                                         
              ████                        ■███▪                                         
             ●███-                       -███♦                                          
            ▴███●                        ●███▪                                          
            ●██♦                         ████▪                                          
            ●██·                         ■███▴                                          
                                          ■■-                                           
                                                                                    
                                                                                    
                                                                                    
                                                                                                                
*)
