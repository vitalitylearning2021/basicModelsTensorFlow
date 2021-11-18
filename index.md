# Basic machine learning models with TensorFlow

## What is Machine Learning?

Machine Learning is one of the fields of artificial intelligence and identifies the development of automatic learning systems capable to improve their own performance in a given task through experience. The applications of Machine Learning are the most varied and regard, to cite only a few examples, speech recognition, computer vision including autonomous driving cars, the classificiation of new astronomical structures exploited, for example, by NASA to automatically classify the celestial bodies in the sky surveys and the data-mining. 

### Classification of the Machine Learning algorithms

Machine Learning algorithms are typically divided in three main categories, depending on the type of learning used to contruct a forecast model.

  - *Supervised Learning*: the computer is trained by a learning set in which each input element is made up by a set of attributes to which an output value, called label, is attached. The output value represents the result associated to the corresponding input element. Once performed the training phase and built up the forecast model, the performance of the approach can be assessed by a testing phase in which the algorithm is applied to test data and the forecast results are compared to the "ground-truth" results.
  - *Unsupervised Learning*: in this case, no label is associated to the input data and the only way the algorithm has to construct a model is analyzing the input data and organize them on the basis of common features. Such an approach can be used either to highlight models hidden within the data or to return forecasts.
  - *Reinforced Learning*: by this approach, the program interacts with a dynamic environment from which it receives feedbacks according to the correctness of the performed action choices. The system receives a prize if the choice is right, while it receives a penalty if the choice is wrong.

A different division can be performed depending on the desired output. For supervised learning, we have different approaches, among which:

  - *Classification*: in the case of classifiers, the inputs are subdivided into two or more classes and the algorithm must return a model capable to assign possible new elements to one of these classes.
  - *Regression*: in the case of regression, the outputs are numerical results, rather than classes.

## What is TensorFlow?

TensorFlow is an open source library used in machine learning.

It has been developed by Google in the framework of the Google Brain (AI) project and, in 2015, its code has been released with an open source license.
Today, TensorFlow is used in many areas of science and industry for machine learning applications and, more generally, for artificial intelligence. For example, Google uses TensorFlow for image recognition algorithms and in its own RankBrain algorithm developed and used by its search engine to interpret the meaning of a query. Moreover, TensorFlow is often employed for reading handwritten text and for the automatic recognition of objects or people. Generally speaking, TensorFlow is routinely adopted in commercial or research developments to create and distribute automatic learning modules. In few years, it has evolved from a simple library to a whole ecosystem for all types of machine learning.

The name TensorFlow is composed by the two words “tensor” and “flow”. The use of “tensor” is due to the fact that a tensor is an algebraic object that describes a multilinear relationship between sets of algebraic objects related to a vector space, that such algebraic objects are often described by multidimensional matrices and that TensorFlow is based on the manipulation of multidimensional matrices. The word “flow” evokes the operations flow.

TensorFlow is compatible with Windows, Linux, MaxOS, Raspbian and Android. It can be used under Python, C, Java, Go and Rust. At this [link](https://www.tensorflow.org/install), the installation instructions can be found along with the versions of the operating systems and of Python with which TensorFlow is compatible. TensorFlow is already installed under Google Colab and there is nothing simpler than using it under such a service.

TensorFlow  runs on both CPUs and GPUs. Nevertheless, it must be mentioned that, since 2016, Google has released a new Application Specific Integrated Circuit (ASIC) processor named Tensor Processing Unit (TPU) and purposely designed for AI applications that require TensorFlow. It is capable to accelerate machine learning processing and to execute TensorFlow operations much quicker as compared to a standard CPU. By Google Colab, it is possible to choose the desired computing platform among CPU, GPU and TPU.
Today, we have TensorFlow `2.6.1` which dramatically simplfies the coding as compared to TensorFlow `1.x`. 

### Lazy and eager evaluations

One of the most relevant differences between TensorFlow `1.x` and TensorFlow `2.x` is the default execution modality: TensorFlow `1.x` adopts the *eager* evaluation while TensorFlow `2.x` uses the *lazy* evaluation. But what is the difference between the two?

A programming language like Python implements the eager execution model. In other words, the operations are executed immediately as they are called. From the User's point of view, this has the advantage of simplfying the debugging since pieces of code can be easily integrated in tools for error check or simply the content of variables can be controlled in a direct way.

Opposite to that, by the lazy evaluation, the operations are not executed at the point in which they are invoked, but they are exploited to create a computational graph, as illustrated in the following Fig. [1](#computationalGraph)

<p align="center">
  <img src="computationalGraph.png" width="250" id="computationalGraph">
  <br>
     <em>Figure 1. Computational graph illustrating the <img src="https://render.githubusercontent.com/render/math?math=2 x_1 %2B x_2"> operation.</em>
</p>

This, obviously, complicates the debugging phase since, opposite to the eager modality, it is not possible, for example, to follow the content of the variables during the execution. The lazy evaluation has however the following advantages:

  - *Parallelism*. By the computational graph, spotting the parallelizable portions of the code is simpler.
  - *Distributed execution*. By the computational graph, automatically distributing the execution of portions of the code among different devices (CPUs, GPUs and TPUs in the cases of our interest), possibly installed on different machines, is simpler.
  - *Compilation*. The computational graph can be used to generate a faster code since it can lead to “simplifications” or fusions of adjacent operations.
  - *Portability*. The computational graph is language and platform-independent which favors portability.

Using lazy execution was cumbersome in TensorFlow `1.x` since it required the use of proper *sessions*. Now this is not necessary anymore, since, as mentioned, lazy execution is the default computational modality for TensorFlow `2.x`. Of course, it is possible to manually switch to eager computation in TensorFlow `2.x` too.

Once clarified the meaning of lazy and eager evaluations in TensorFlow, let us spend some words on *automatic differentiation* for the automatic computation of the derivatives of a multidimensional function. Automatic differentiation, indeed, is a crucial technique in many machine and deep learning applications.

### Automatic differentiation

To illustrate automatic differentiation in a simple way [\[1\]](#AUTODIFF), let us suppose to compute the partial derivatives of a function <img src="https://render.githubusercontent.com/render/math?math=f(x_1,x_2)">, where <img src="https://render.githubusercontent.com/render/math?math=x_1"> and <img src="https://render.githubusercontent.com/render/math?math=x_2"> are the independent variables and function <img src="https://render.githubusercontent.com/render/math?math=f">
could represent, for example, the output of a *cost function* as it will be clearer in the following. Our purpose is to “automatically” compute the partial derivatives of <img src="https://render.githubusercontent.com/render/math?math=f">, namely, <img src="https://render.githubusercontent.com/render/math?math=\partial f/\partial x_1"> and <img src="https://render.githubusercontent.com/render/math?math=\partial f/\partial x_2">. Many possibilities exist.

The first consists into resorting to a finite difference approximation of the derivatives, as done in [Problem solving with PyCUDA](https://vitalitylearning2021.github.io/problemSolvingPyCUDA/). However, finite differencing must be performed with care to avoid numerical errors which can easily become much relevant. 

The second is using a symbolic differentiation tool, as those available in Matlab or WolframAlpha. Nevertheless, also symbolic differentiation must be used with care since the size of the result can be much larger than the minimum required.

A third possibility is to regard the expression to differentiate as a series of elementary operations that can be implemented by any programming language. On applying the differentiation rules to each individual elementary operation, we can obtain a code enabling the numerical computation of the derivatives. To better clarify the idea, let us suppose that the expression of <img src="https://render.githubusercontent.com/render/math?math=f(x_1,x_2)"> is the following:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=f(x_1, x_2) = x_2 \cdot \sin \left( 2\cdot x_1 %2B x_2 \right)." id="functionToBeDifferentiated">       [1]
</p>

In this case, the elementary operations that can be isolated are:

<p align="center">
   <img src="equation_2.png" width="150" id="elementaryOperationsAutomaticDiff">     [2]
</p>

These operations correspond to the following graph:

<p align="center">
  <img src="computationalGraphAutoDiff.png" width="120" id="computationalGraphAutodiff">
  <br>
     <em>Figure 2. Computational graph for the automatic differentiation example.</em>
</p>

The first two operations are meant as assignment operations. The graph in Fig. [2](#computationalGraphAutodiff) is the same graph that TensorFlow would internally build up for the lazy execution.

Let us suppose now to compute <img src="https://render.githubusercontent.com/render/math?math=\partial f/\partial x_1"> using the so-called *forward mode automatic differentiation*. By such a technique, the calculations begin from the innermost function and proceed with the derivatives towards the outermost functions. In the case at hand, the forward mode automatic differentiation would perform the following computations paired with respect to those in [\[2\]](#elementaryOperationsAutomaticDiff):

  - the first operation would be to compute the derivative of <img src="https://render.githubusercontent.com/render/math?math=x_1"> with respect to <img src="https://render.githubusercontent.com/render/math?math=x_1">, namely <img src="https://render.githubusercontent.com/render/math?math=\partial x_1/\partial x_1">, which leads to <img src="https://render.githubusercontent.com/render/math?math=1"> as result;
  - the second operation would be to compute the derivative of <img src="https://render.githubusercontent.com/render/math?math=x_2"> with respect to <img src="https://render.githubusercontent.com/render/math?math=x_1">, namely, <img src="https://render.githubusercontent.com/render/math?math=\partial x_2/\partial x_1">, which leads to <img src="https://render.githubusercontent.com/render/math?math=0"> as result, being <img src="https://render.githubusercontent.com/render/math?math=x_2"> independent from <img src="https://render.githubusercontent.com/render/math?math=x_1">;
  - furthermore, we would need to compute the derivative of <img src="https://render.githubusercontent.com/render/math?math=z_1"> with respect to <img src="https://render.githubusercontent.com/render/math?math=x_1">, namely, <img src="https://render.githubusercontent.com/render/math?math=\partial z_1/\partial x_1">; the result would be <img src="https://render.githubusercontent.com/render/math?math=2"> since we are differentiating a constant multiplied by a function;
  - as fourth step, we would need to compute <img src="https://render.githubusercontent.com/render/math?math=\partial z_2/\partial x_1">; this corresponds to <img src="https://render.githubusercontent.com/render/math?math=\partial z_1/\partial x_1 %2B \partial x_2/\partial x_1"> which returns <img src="https://render.githubusercontent.com/render/math?math=2">, according to the previous steps;
  - as penultimate step, we would need to compute <img src="https://render.githubusercontent.com/render/math?math=\partial z_3/\partial x_1">; by the *chain rule*, we would obtain <img src="https://render.githubusercontent.com/render/math?math=\cos\left(z_2\right)\partial z_2/\partial x_1">; we could exploit the result at the previous step to evaluate <img src="https://render.githubusercontent.com/render/math?math=\partial z_2/\partial x_1">;
  - finally, the last step would consist of computing <img src="https://render.githubusercontent.com/render/math?math=\partial z_4/\partial x_1">; using the chain rule again, we would obtain <img src="https://render.githubusercontent.com/render/math?math=\partial x_2/\partial x_1 \cdot z_3 %2B x_2 \cdot \partial z_3/\partial x_1">.

As it can be seen, at each step, the computed derivatives depend only on the derivatives at the previous steps. By comparing the above operations with the graph in Fig. [2](#computationalGraphAutodiff), we can see that the operations corresponding to each node depend only on those of the nodes immediately upstream on the graph. Thanks to that, we can express the computation of the partial derivative with respect to <img src="https://render.githubusercontent.com/render/math?math=x_1"> as the following sequence of operations

<p align="center">
   <img src="equation_3.png" width="350" id="elementaryOperationsAutomaticDiffDerivative">     [3]
</p>

By forward substitution in the last equation of the eqs. [\[3\]](#elementaryOperationsAutomaticDiffDerivative), the derivative of interest remains computed. Given that, the procedure to compute <img src="https://render.githubusercontent.com/render/math?math=\partial f/\partial x_2"> is totally analogous.

It should be noticed that the forward mode automatic differentiation is effective when the functions of which computing the derivatives have few inputs and many outputs. Opposite to that, in machine learning applications, as it will be seen, the functions have tipically different inputs and only one output.

An alternative to forward mode automatic differentiation, which is used by almost all the machine learning and deep learning tools, is called *reverse mode automatic differentiation* and consists of traversing the graph in a reversed order. To clarify this point, let us consider again eqs. [\[2\]](#elementaryOperationsAutomaticDiff) and suppose that we want to compute <img src="https://render.githubusercontent.com/render/math?math=\partial z_4/\partial x_1">. Thanks to the chain rule, we can compute <img src="https://render.githubusercontent.com/render/math?math=\partial z_4/\partial x_1"> as

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial z_4}{\partial x_1}=\frac{\partial z_4}{\partial z_3}\cdot\frac{\partial z_3}{\partial z_2}\cdot \frac{\partial z_2}{\partial z_1}\cdot \frac{\partial z_1}{\partial x_1}." id="reverseModeAutoDiff">       [4]
</p>

Reconsidering eqs. [\[2\]](#elementaryOperationsAutomaticDiff) in a reverse way, we have

<p align="center">
   <img src="equation_5.png" width="180" id="elementaryOperationsAutomaticDiffDerivativeReverse">     [5]
</p>

By substituting eqs. [\[5\]](#elementaryOperationsAutomaticDiffDerivativeReverse) in [\[2\]](#elementaryOperationsAutomaticDiff), the partial derivative of interest remains computed.

## “Hello World” in TensorFlow

Let us present the first, simple, classical example employing TensorFlow:

``` python
import tensorflow as tf

print(tf.__version__)

message = tf.constant('Hello World')

print(message)

tf.print(message)
```

The Listing is much simple to be read. The first instruction performs the import of the TensorFlow library, while the second shows the version, `2.7.0` at the time of writing. Then, a constant tensor, namely, an immutable sensor, of string type, is defined which has `Hello World` as value. 

The last two instructions perform the printout of the tensor. The first `print` uses a Python function and the printout consists of just the properties of the `message` object. Indeed, the printout is

``` python
tf.Tensor(b'Hello World', shape=(), dtype=string)
```

It informs us that the `message` object is a TensorFlow tensor, having the value `Hello World`, of undefined `shape` and of `string` type. If we want to print out the only value of a tensor, then we need the TensorFlow primitive `tf.print`.

## Basic operations in TensorFlow

Internamente, Tensorflow rappresenta i tensor come array <img src="https://render.githubusercontent.com/render/math?math=N">-dimensionali di *datatypes* base (`int`, `string`, etc..). Il datatype di un tensore è sempre noto in qualsiasi momento dell'esecuzione del codice ed è condiviso da tutti gli elementi del tensore. Nella modalità di esecuzione lazy, la *shape* di un tensore, ossia il numero di dimensioni e la lunghezza di ogni dimensione, può invece essere anche solo parzialmente nota. Questo avviene perché le operazioni in un grafo producono tensori di dimensioni full-known solo se quelle degli input sono altrettanto conosciute. Dunque, spesso è possibile determinare la shape finale di un tensore solo al termine dell’esecuzione dei grafi. Il *rank* di un tensor è infine il numero di sue dimensioni. Datatype, shape and rank rappresentano le tre caratteristiche fondamentali di un tensore.

Nel seguito, mostreremo semplici esempi con difficoltà incrementale. The import of the TensorFlow library

``` python
import tensorflow as tf
```

will be always assumed and suppressed.

### Tensori monodimensionali costanti

Facciamo un primo semplice esempio di tensore monodimensionale costante il cui prototipo generale è

``` python
tf.constant(
   value, dtype=None, shape=None, name='Const'
)
```

Nell'esempio che segue, si creano due tensori monodimensionali costanti a partire da liste dei loro elementi. Il primo è un tensore di stringhe, mentre il secondo è un tensore di numeri razionali

``` python
instruments       = tf.constant(["Violin", "Piano"], tf.string)
rationalNumbers   = tf.constant([1.223, 2.131, -10.43], tf.float32)

print("`instruments` is a {}-d Tensor with shape: {}".format(tf.rank(instruments).numpy(), tf.shape(instruments)))
print("`rationalNumbers` is a {}-d Tensor with shape: {}".format(tf.rank(rationalNumbers).numpy(), tf.shape(rationalNumbers)))
```

### Tensori <img src="https://render.githubusercontent.com/render/math?math=N">-dimensionali costanti

In applicazioni reali, come vedremo, avremo bisogno anche di 4-d Tensor, con i quali rappresentare ad esempio immagini in task di image preprocessing e computer vision.

Consideriamo dunque un tensor pronto a gestire 10 immagini quadrate a colori di 256px nello spazio RGB: 10 x 256*256*3.

``` python
'''Define a 4-d Tensor.'''
# Use tf.zeros to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3.
# You can think of this as 10 images where each image is RGB 256 x 256.

#  Creates a tensor with all elements set to zero.
images = tf.constant(tf.zeros((10,256,256,3), tf.int32, "The name of the operation"))

assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"
```


A eccezione del `tf.Variable`, i tensori sono immutabili.

``` python
tf.Variable(
 initial_value=None, trainable=None, validate_shape=True, caching_device=None,
    name=None, variable_def=None, dtype=None, import_scope=None, constraint=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.compat.v1.VariableAggregation.NONE, shape=None
)
```






Un grafo è la rappresentazione, per mezzo di nodi, di operazioni eseguite sui tensori.



Per comprendere il flow dei dati e delle operazioni in Tensorflow, possiamo servirci dei grafi, rappresentazioni convenienti di computazioni.

Un grafo, o graph, e sarà quindi costituito da tensori, che gestiranno i dati, e dalle operazioni compiuti su di essi.

Vediamo una semplice operazione di somma: due tensori costanti e una sommatoria.

``` python
# Create the nodes in the graph, and initialize values
a = tf.constant(15)
b = tf.constant(61)

# Add them!
c1 = tf.add(a,b)
c2 = a + b # TensorFlow overrides the "+" operation so that it is able to act on Tensors
print(c1)
print(c2)

# Output:
# tf.Tensor(76, shape=(), dtype=int32)
# tf.Tensor(76, shape=(), dtype=int32)
```

Ora consideriamo un esempio più complesso:

Andiamo ora a costruire una funzione che riproduca le operazioni rappresentate nel grafo:

``` python
'''Defining Tensor computations''''

# Construct a simple computation function
def func(a,b):
  c = tf.add(a,b)
  d = tf.subtract(b,1)
  e = tf.multiply(c,d)
  return e
```

Quindi calcoliamo il risultato:

``` python
# Consider example values for a,b
a, b = 1.5, 2.5
# Execute the computation
e_out = func(a,b)
print(e_out)

# Output:
# tf.Tensor(6.0, shape=(), dtype=float32)
```

L’ouput è un semplice scalare, privo di dimensoni.

https://ichi.pro/it/la-guida-definitiva-per-principianti-a-tensorflow-72377596104903
https://ichi.pro/it/tutorial-su-tensorflow-una-guida-completa-all-apprendimento-approfondito-con-tensorflow-204595782052810
https://riptutorial.com/Download/tensorflow-it.pdf
https://vittoriomazzia.com/tensorflow-lite/
https://ichi.pro/it/guida-per-principianti-a-tensorflow-2-x-per-applicazioni-di-deep-learning-219815635385326

Variabili e costanti
https://www.it-swarm.it/it/python/variabili-e-costanti-tensorflow/833521263/

https://ichi.pro/it/padroneggiare-le-variabili-di-tensorflow-in-5-semplici-passaggi-100777216055126

https://www.it-swarm.it/it/python/perche-denominiamo-le-variabili-tensorflow/1056840775/

https://pretagteam.com/question/in-tensorflow-what-is-the-difference-between-a-variable-and-a-tensor

https://people.unica.it/diegoreforgiato/files/2012/04/TesiNicolaPes.pdf

https://ichi.pro/it/padroneggiare-i-tensori-tensorflow-in-5-semplici-passaggi-59313927797638

https://it.linkedin.com/pulse/tensorflow-what-why-how-when-mauro-minella

Placeholder

https://www.it-swarm.it/it/tensorflow/qual-e-la-differenza-tra-tf.placeholder-e-tf.variable/824943616/

https://amslaurea.unibo.it/14173/1/tesi_piscaglia.pdf

Regressione lineare?
https://www.andreaminini.com/ai/tensorflow/esempio-tutorial-tensorflow

https://learntutorials.net/it/tensorflow/topic/856/iniziare-con-tensorflow

Semplice rete neurale?
https://andreaprovino.it/start-tensorflow-2-esempio-semplice-tutorial/
https://medium.com/@cosimo.iaia/machine-learning-tensorflow-per-luomo-di-strada-2c71a948b4e3
https://ichi.pro/it/introduzione-a-tensorflow-2-0-275931290659758
https://andreaprovino.it/tensorflow-guida-italiano-primi-passi-con-tensorflow/

%\section{Linear regression}
%http://www.ecostat.unical.it/Didattica/Statistica/didattica/StatAziendale2/StatAz2_cap2.pdf\\
%https://www.germanorossi.it/mi/file/disp/Regression.pdf\\
%http://www.dima.unige.it/~rogantin/ls_stat/N_scheda_5.pdf\\
%https://sanjayasubedi.com.np/deeplearning/tensorflow-2-linear-regression-from-scratch/\\
%https://towardsdatascience.com/get-started-with-tensorflow-2-0-and-linear-regression-29b5dbd65977

# REFERENCES
<p align="center" id="AUTODIFF" >
</p>
[1] A.G. Baydin, B.A. Pearlmutter, A.A. Radul, J.M. Siskind, "Automatic differentiation in machine learning: a survey," J. Machine Learn. Res., vol. 18, pp. 1-43, 2018.


