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

Internamente, Tensorflow rappresenta i tensor come array <img src="https://render.githubusercontent.com/render/math?math=N">-dimensionali di *datatypes* base (`int`, `string`, etc..). Il datatype di un tensore è sempre noto in qualsiasi momento dell'esecuzione del codice ed è condiviso da tutti gli elementi del tensore. Nella modalità di esecuzione lazy, la *shape* di un tensore, ossia il numero di dimensioni e la lunghezza di ogni dimensione, può invece essere anche solo parzialmente nota. Questo avviene perché le operazioni in un grafo producono tensori di dimensioni full-known solo se quelle degli input sono altrettanto conosciute. Dunque, spesso è possibile determinare la shape finale di un tensore solo al termine dell’esecuzione dei grafi. Il *rank* di un tensor è infine il numero di sue dimensioni. Datatype, shape and rank rappresentano le tre caratteristiche fondamentali di un tensore. Tutti i tensori hanno anche una dimensione, che è il numero totale di elementi al loro interno. Come si vede, i tensori di TensorFlow sono simili agli array della libreria NumPy.

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

Nelle tipiche applicazioni di TensorFlow (e.g., image processing or computer vision), può essere necessario gestire anche immagini 2d o 3d o sequenze di immagini 2d o 3d. Da questo punto di vista, può essere necessario riuscire a gestire anche 4d tensor. Nell'esempio riportato di seguito, si definisce un 4d constant tensor doppia precisione per gestire 3 immagini 128 x 128 x 16. 

``` python
images = tf.constant(tf.zeros((3, 128, 128, 16), tf.float64, "4d constant tensor definition"))

assert isinstance(images, tf.Tensor), "Matrix must be a TensorFlow tensor object"
assert tf.rank(images).numpy() == 4, "Matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [3, 128, 128, 16], "Matrix has incorrect shape"
```

Per attivare gli assert, è necessario modificare o la natura dell'oggetto, o il rank oppure lo shape.

Un tensore <img src="https://render.githubusercontent.com/render/math?math=N">-dimensionale può anche essere costruito a partire da una lista <img src="https://render.githubusercontent.com/render/math?math=N">-dimensionale, come nell'esempio seguente

``` python
threeDimensionalList = [[[0, 1, 2], 
                         [3, 4, 5]], 
                        [[6, 7, 8], 
                         [9, 10, 11]]]
rank3Tensor = tf.constant(threeDimensionalList)
print(rank3Tensor)
print("The number tensor dimensions is", rank3Tensor.ndim)
print("The tensor shape is", rank3Tensor.shape)
print("The tensor data type is", rank3Tensor.dtype)
print("The tensor size is", tf.size(rank3Tensor).numpy())
```

Nell'esempio di sopra, vengono anche `print`-ed the number of dimensions, the shape, the data type and the size of the tensor. La size indica il numero totale di elementi di un tensore. Come si può vedere, non è possibile to `print` the size con un attributo dell'oggetto tensore. Invece, è necessario usare la funzione `tf.size()` e convertire il suo output con la funzione di istanza `.numpy()` per ottenere un risultato più leggibile.

### Indexing

TensorFlow segue anche le regole di indicizzazione Python standard.

``` python
aList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
aTensor = tf.constant(aList)

print("The first element is:", aTensor[0].numpy())
print("The last element is:", aTensor[-1].numpy())
print("Elements in between the first and the last:", aTensor[1 : -1].numpy())
```

Nell'esempio sopra riportato, si può notare che:

  - gli indici iniziano da zero (`0`);
  - il valore dell'indice negativo (`-n`) indica il conteggio all'indietro dalla fine;
  - the colon syntax (`:`) is used to slide: `start : stop : step`.
  - not shown in the example above, but the comma syntax (`,`) vengono utilizzate per raggiungere livelli più profondi.

### Basic operations

Nell'esempio che segue, si mostra come si possano eseguire operazioni matematiche di base su tensori come addizione, elementwise multiplication, matrix multiplication e determinare il massimo o l'indice di massimo di un tensore.

``` python
a = tf.constant([[3, 2], [-10, 7]], dtype=tf.float32)
b = tf.constant([[0, 2], [  6, 1]], dtype=tf.float32)

addingTensors         = tf.add(a, b)
multiplyingTensors    = tf.multiply(a, b)
matrixMultiplication  = tf.matmul(a, b)
tf.print(addingTensors)
tf.print(multiplyingTensors)
tf.print(matrixMultiplication)

print("The maximum value of b is:", tf.reduce_max(b).numpy())
print("The index position of the maximum element of b is:", tf.argmax(b).numpy())
```

### Reshaping

Proprio come negli array NumPy, è possibile effettuare il reshaping di oggetti TensorFlow. L'operazione `tf.reshape()` è molto veloce poiché i dati sottostanti non devono essere elaborati, ma solo i parametri che descrivono le dimensioni modificati. 

``` python
a = tf.constant([[1, 2, 3, 4, 5, 6]])
print('Initial shape:', a.shape)

b = tf.reshape(a, [6, 1])
print('First reshaping:', b.shape)

c = tf.reshape(a, [3, 2])
print('Second reshaping:', c.shape)

# --- Flattening
print('Flattening:', tf.reshape(a, [-1]))
```

### Operator overload

Quando eseguiamo operazioni tra tensori di dimensioni diverse, la differenza delle dimensioni può essere gestita automaticamente da TensorFlow tramite opportuni overload delle operazioni in questione, esattamente come in NumPy. Ad esempio, quando si tenta di moltiplicare un tensore scalare con un tensore di rango `2`, ogni elemento tensore di rango `2` viene moltiplicato per lo scalare, come nell'esempio che segue:

``` python
m = tf.constant([5])

n = tf.constant([[1, 2], [3, 4]])

tf.print(tf.multiply(m, n))
```

### Irregular tensors

Generalmente, i tensori di interesse hanno forma rettangolare. Tuttavia, TensorFlow supporta anche tipi di tensori irregolari come:

  - ragged tensors,
  - string tensors,
  - sparse tensors.

<p align="center">
  <img src="irregularTensors.png" width="120" id="irregularTensors">
  <br>
     <em>Figure 3. Irregular tensor.</em>
</p>

#### Ragged tensors

I ragged tensors sono tensori con un numero diverso di elementi lungo le varie dimensioni, come mostrato in Fig. [3](#irregularTensors). Un ragged tensor può essere costruito come segue:

``` python
raggedList = [[1, 2, 3], [4, 5], [6]]

raggedTensor = tf.ragged.constant(raggedList)

tf.print(raggedTensor)
```

#### String tensors

I tensori stringa sono tensori che memorizzano gli oggetti stringa. Possiamo costruire un tensore stringa come un normale oggetto tensore passando oggetti stringa come elementi al posto di oggetti numerici, come mostrato di seguito:

``` python
stringTensor = tf.constant(["I like", 
                            "TensorFlow", 
                            "very much"])

tf.print(stringTensor)
```

#### Sparse tensors

Quando molti degli elementi di un tensore sono nulli anzicché diversi da zero, è conveniente utilizzare tensori sparsi. Essi si costruiscono indicando solo gli elementi non nulli e la loro posizione all'interno del tensore:

``` python
sparseTensor = tf.sparse.SparseTensor(indices      = [[0, 0], [2, 2], [4, 4]], 
                                      values       = [25, 50, 100], 
                                      dense_shape  = [5, 5])

tf.print(sparseTensor)
tf.print(tf.sparse.to_dense(sparseTensor))
```

### Prossimo paragrafo

Stiamo vedendo questo

https://ichi.pro/it/la-guida-definitiva-per-principianti-a-tensorflow-72377596104903

QUESTO L'HO VISTO
https://ichi.pro/it/padroneggiare-i-tensori-tensorflow-in-5-semplici-passaggi-59313927797638



A eccezione del `tf.Variable`, i tensori sono immutabili.

``` python
tf.Variable(
 initial_value=None, trainable=None, validate_shape=True, caching_device=None,
    name=None, variable_def=None, dtype=None, import_scope=None, constraint=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.compat.v1.VariableAggregation.NONE, shape=None
)
```

### TensorBoard

Qui si può spiegare perché le variabili constanti apparentemente possono essere modificate

https://stackoverflow.com/questions/46786463/modifying-tensorflow-constant-tensor




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


