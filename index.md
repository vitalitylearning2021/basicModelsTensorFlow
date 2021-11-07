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
TensorFlow is compatible with Windows, Linux, MaxOS, Raspbian and Android. It can be used under Python, C, Java, Go and Rust. At this [link](https://www.tensorflow.org/install), the installation instructions can be found along with the versions of the operating systems and of Python with which TensorFlow is compatible. TensorFlow is already installed under Google Colab and there is nothing simpler than using it under such a service.\\
TensorFlow  runs on both CPUs and GPUs. Nevertheless, it must be mentioned that, since 2016, Google has released a new Application Specific Integrated Circuit (ASIC) processor named Tensor Processing Unit (TPU) and purposely designed for AI applications that require TensorFlow. It is capable to accelerate machine learning processing and to execute TensorFlow operations much quicker as compared to a standard CPU. By Google Colab, it is possible to choose the desired computing platform among CPU, GPU and TPU.
Today, we have TensorFlow `2.6.1` which dramatically simplfies the coding as compared to TensorFlow `1.x`. 

### Lazy and eager evaluations

One of the most relevant differences between TensorFlow `1.x` and TensorFlow `2.x` is the default execution modality: TensorFlow `1.x` adopts the *lazy* evaluation while TensorFlow `2.x` uses the *eager* evaluation. But what is the difference between the two?
A programming language like Python implements the eager execution model. In other words, the operations are executed immediately as they are called. From the User's point of view, this has the advantage of simplfying the debugging since pieces of code can be easily integrated in tools for error check or simply the content of variables can be controlled in a direct way.
Opposite to that, by the lazy evaluation, the operations are not executed at the point in which they are invoked, but they are exploited to create a computational graph, as illustrated in the following Fig. [1](#computationalGraph)

\begin{figure}[H]
%\sidecaption
\begin{center}
\includegraphics[scale=1.1]{Pictures/Chapter06/computationalGraph.png}
\caption{Computational graph illustrating the $xy+2$ operation.}
\label{computationalGraph}
\end{center}
\end{figure}

Questo ovviamente complica il debugging in quanto, al contrario di prima, non è possibile, ad esempio, seguire i valori delle variabili nel tempo. Al contrario, la lazy execution ha i seguenti vantaggi:

\begin{itemize}
    \item \emph{Parallelism}. Con il computational graph, è più semplice individuare le porzioni parallelizzabili del codice.
    \item \emph{Distributed execution}. Con il computational graph, è più semplice distribuire automaticamente l'esecuzione di porzioni di codice tra diversi devices (CPUs, GPUs and TPUs nei casi di nostro interesse), eventualmente presenti su macchine differenti.
    \item \emph{Compilation}. Il computational graph può essere utilizzato anche per generare un codice più veloce in quanto può dar luogo a ``semplificazioni'' o fusioni di operazioni adiacenti.
    \item \emph{Portability}. Il computational graph è language e platform-independent, cosa che ne favorisce la portabilità.
\end{itemize}

Utilizzare la lazy execution era cumbersome in TensorFlow \lstinline{1.x} in quanto richiedeva l'utilizzo di apposite \lstinline{session}. Ora questo non è più necessario, essendo, come detto, la eager execution la modalità di default per TensorFlow \lstinline{2.x}. Naturalmente, è possibile manualmente switchare dalla eager alla lazy in TensorFlow \lstinline{2.x}.\\
Una volta chiarito il significato di lazy ed eager evaluation in TensorFlow, spendiamo qualche parola sulla tecnica della differenziazione automatica per il calcolo automatico delle derivate di una funzione multidimensionale. La differenziazione automatica, infatti, rappresenta una tecnica cruciale in molte applicazioni di machine learning e deep learning.

%\section{Linear regression}
%http://www.ecostat.unical.it/Didattica/Statistica/didattica/StatAziendale2/StatAz2_cap2.pdf\\
%https://www.germanorossi.it/mi/file/disp/Regression.pdf\\
%http://www.dima.unige.it/~rogantin/ls_stat/N_scheda_5.pdf\\
%https://sanjayasubedi.com.np/deeplearning/tensorflow-2-linear-regression-from-scratch/\\
%https://towardsdatascience.com/get-started-with-tensorflow-2-0-and-linear-regression-29b5dbd65977

%\section{Logistic regression}
