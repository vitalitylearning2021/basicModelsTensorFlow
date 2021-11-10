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
Today, TensorFlow is used in many areas of science and industry for machine learning applications and, more generally, for artificial intelligence. For example, Google uses TensorFlow for image recognition algorithms and in its own RankBrain algorithm developed and used by its search engine to interpret the meaning of a query. Moreover, TensorFlow is often employed for reading handwritten text and for the automatic recognition of objects or people.

The name TensorFlow is composed by the two words “tensor” and “flow”. The use of “tensor” is due to the fact that a tensor is an algebraic object that describes a multilinear relationship between sets of algebraic objects related to a vector space, that such algebraic objects are often described by multidimensional matrices and that TensorFlow is based on the manipulation of multidimensional matrices. The word “flow” evokes the operations flow.

TensorFlow is compatible with Windows, Linux, MaxOS, Raspbian and Android. It can be used under Python, C, Java, Go and Rust. At this [link](https://www.tensorflow.org/install), the installation instructions can be found along with the versions of the operating systems and of Python with which TensorFlow is compatible. TensorFlow is already installed under Google Colab and there is nothing simpler than using it under such a service.

TensorFlow  runs on both CPUs and GPUs. Nevertheless, it must be mentioned that, since 2016, Google has released a new Application Specific Integrated Circuit (ASIC) processor named Tensor Processing Unit (TPU) and purposely designed for AI applications that require TensorFlow. It is capable to accelerate machine learning processing and to execute TensorFlow operations much quicker as compared to a standard CPU. By Google Colab, it is possible to choose the desired computing platform among CPU, GPU and TPU.
Today, we have TensorFlow `2.6.1` which dramatically simplfies the coding as compared to TensorFlow `1.x`. 

### Lazy and eager evaluations

One of the most relevant differences between TensorFlow `1.x` and TensorFlow `2.x` is the default execution modality: TensorFlow `1.x` adopts the *eager* evaluation while TensorFlow `2.x` uses the *lazy* evaluation. But what is the difference between the two?

A programming language like Python implements the eager execution model. In other words, the operations are executed immediately as they are called. From the User's point of view, this has the advantage of simplfying the debugging since pieces of code can be easily integrated in tools for error check or simply the content of variables can be controlled in a direct way.

Opposite to that, by the lazy evaluation, the operations are not executed at the point in which they are invoked, but they are exploited to create a computational graph, as illustrated in the following Fig. [1](#computationalGraph)

<p align="center">
  <img src="computationalGraph.png" width="400" id="computationalGraph">
  <br>
     <em>Figure 1. Computational graph illustrating the <img src="https://render.githubusercontent.com/render/math?math=xy %2B 2"> operation.</em>
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
  <img src="https://render.githubusercontent.com/render/math?math=f(x_1, x_2) = x_2 \cdot \sin \left( 2\cdot x_1 + x_2 \right)." id="functionToBeDifferentiated">       [9]
</p>

In this case, the elementary operations that can be isolated are:



\begin{equation}
  \left\{
    \begin{array}{lc}
      x_1=\ldots\\
      x_2=\ldots\\
      z_1=2\cdot x_1 \\
      z_2 = z_1 + x_2 \\
      z_3 = \sin\left(z_2\right) \\
      z_4 = x_2\cdot z_3 \\
    \end{array}\right..
  \label{elementaryOperationsAutomaticDiff}
\end{equation}

Queste operazioni corrispondono al grafo illustrato nella seguente figura:

\begin{figure}[H]
%\sidecaption
\begin{center}
\includegraphics[scale=0.8]{Pictures/Chapter06/computationalGraphAutoDiff.png}
\caption{Computational graph for the automatic differentiation example.}
\label{computationalGraphAutodiff}
\end{center}
\end{figure}

Le prime due operazioni si intendono di assegnazione dei valori. Il grafo in Fig. \ref{elementaryOperationsAutomaticDiff} è esattamente il grafo che TensorFlow costruirebbe per la lazy execution. \\ 
Immaginiamo ora di dover calcolare $\partial f/\partial x_1$ utilizzando la cosiddetta \emph{forward mode automatic differentiation}. Tale tecnica parte dalla funzione più interna e calcola le derivate, utilizzando le opportune regole di derivazione, muovendosi verso le funzioni più esterne. Nel caso in esame, essa compirebbe le seguenti operazioni paired with respect to those in (\ref{elementaryOperationsAutomaticDiff}):

\begin{enumerate}
    \item La prima operazione sarebbe quella di calcolare la derivata di $x_1$ rispetto ad $x_1$, cioè $\partial x_1/\partial x_1$, avendo come risultato $1$.
    \item La seconda operazione sarebbe quella di calcolare la derivata di $x_2$ rispetto ad $x_1$, cioè $\partial x_1/\partial x_1$, ottenendo $0$, essendo $x_2$ indipendente da $x_1$.
    \item Ancora, bisognerebbe calcolare la derivata di $z_1$ rispetto a $x_1$, cioè $\partial z_1/\partial x_1$. Il risultato sarebbe ovviamente $2$ per la regola di derivazione di una costante per una funzione.
    \item Come quarto passo, bisognerebbe calcolare $\partial z_2/\partial x_1$. Per la regola di derivazione della somma, questa sarebbe uguale a $\partial z_1/\partial x_1+\partial x_2/\partial x_1$ che fornisce come risultato, in base ai passi precedenti, $2$.
    \item Al penultimo passo, bisogna calcolare $\partial z_3/\partial x_1$. Per la \emph{chain rule}, otteniamo $\cos\left(z_2\right)\partial z_2/\partial x_1$, dove può essere sfruttato il risultato al passo precedente per valutare $\partial z_2/\partial x_1$.
    \item Infine, nell'ultimo passo, bisogna calcolare $\partial z_4/\partial x_1$. Usando la regola della derivata della moltiplicazione otteniamo $\partial x_2/\partial x_1 \cdot z_3+x_2 \cdot \partial z_3/\partial x_1$.
\end{enumerate}

Come si può vedere, ad ogni passo, le derivate calcolate dipendono solo dalle derivate ai passi precedenti. Confrontando le operazioni di sopra con il grafo in Fig. \ref{computationalGraphAutodiff}, si può vedere che le operazioni di derivazione corrispondenti a ciascun nodo dipendono solo da quelle dei nodi immediatamente a monte sul grafo. Grazie a ciò, possiamo esprimere il calcolo della derivata parziale rispetto ad $x_1$ come la seguente sequenza di operazioni

\begin{equation}
  \left\{
    \begin{array}{lc}
      \partial x_1/\partial x_1=1\\
      \partial x_2/\partial x_1=0\\
      \partial z_1/\partial x_1=2 \\
      \partial z_2/\partial x_1= \partial z_1/\partial x_1 + \partial x_2/\partial x_1= \partial z_1/\partial x_1\\
      \partial z_3/\partial x_1= \partial z_2/\partial x_1\cos\left(z_2\right) \\
      \partial z_4/\partial x_1= \partial x_2/\partial x_1\cdot z_3 +x_2\partial z_3/\partial x_1\\
    \end{array}\right..
  \label{elementaryOperationsAutomaticDiffDerivative}
\end{equation}

Given that, il procedimento per il calcolo di $\partial f/\partial x_2$ è del tutto simile.\\
E' da notare che la forward mode automatic differentiation è efficiente quando le funzioni di cui calcolare le derivate hanno pochi ingressi e molte uscite. Al contrario, nelle applicazioni di machine learning, come si vedrà, le funzioni hanno tipicamente differenti ingressi ed una sola uscita.\\ Un'alternativa della forward mode automatic differentiation, che poi è quella usata da quasi tutti gli strumenti di machine learning o deep learning prende il nome di \emph{reverse mode automatic differentiation} e consiste nel percorrere il grafo al contrario. Per chiarire questo punto, riprendiamo in considerazione le (\ref{elementaryOperationsAutomaticDiff}) e supponiamo di voler calcolare $\partial z_4/\partial x_1$. Grazie alla chain rule, possiamo scrivere $\partial z_4/\partial x_1$ come

\begin{equation}
    \frac{\partial z_4}{\partial x_1}=\frac{\partial z_4}{\partial z_3}\cdot\frac{\partial z_3}{\partial z_2}\cdot \frac{\partial z_2}{\partial z_1}\cdot \frac{\partial z_1}{\partial x_1}
\end{equation}

Ripercorrendo le (\ref{elementaryOperationsAutomaticDiff}) al contrario, abbiamo

\begin{equation}
  \left\{
    \begin{array}{lc}
      \frac{\partial z_4}{\partial z_3}=x_2\\
      \frac{\partial z_3}{\partial z_2}=\cos\left(z_2\right)\\
      \frac{\partial z_2}{\partial z_1}=1 \\
      \frac{\partial z_1}{\partial x_1}=2 \\
    \end{array}\right..
  \label{elementaryOperationsAutomaticDiffDerivativeReverse}
\end{equation}

\\

Per rendere il tutorial più concreto possibile, supponiamo di avere questa espressione:
% https://iaml.it/blog/differenziazione-automatica-parte-1
% https://iaml.it/blog/differenziazione-automatica-parte-2

%https://gdcoder.com/tensorflow-2-0-in-5-minutes/\\


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


