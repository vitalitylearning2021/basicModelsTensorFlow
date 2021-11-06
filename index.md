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
TensorFlow è compatibile con i sistemi operativi Windows, Linux, MacOS, Raspbian e Android. Esso può essere utilizzato in connessione con l'uso di Python, C, Java, Go e Rust. Alla pagina \url{https://www.tensorflow.org/install} sono disponibili le istruzioni per l'installazione di TensorFlow e le versioni dei sistemi operativi e di Python con cui TensorFlow è compatibile.  Tuttavia, esso è già installato su Google Colab e non c'è nulla di più facile che utilizzarlo tramite such a service.\\
TensorFlow  gira sia su CPU e GPU. Tuttavia, è da menzionare che, nel 2016, Google ha realizzato un nuovo Application Specific Integrated Circuit (ASIC) processor appositamente progettato per le applicazioni AI che usano TensorFlow, denominato Tensor Processing Unit (TPU). Esso è infatto in grado di accelerare processing di tipo machine learning e di svolgere le operazioni di TensorFlow molto più velocemente rispetto a una CPU standard. Con Google Colab è possibile scegliere la piattaforma desiderata tra CPU, GPU e TPU.\\
Oggi siamo alla versione 2.0 di TensorFlow che semplifica notevolmente la programmazione rispetto alla versione Tensorflow 1.x. 


%\section{Linear regression}
%http://www.ecostat.unical.it/Didattica/Statistica/didattica/StatAziendale2/StatAz2_cap2.pdf\\
%https://www.germanorossi.it/mi/file/disp/Regression.pdf\\
%http://www.dima.unige.it/~rogantin/ls_stat/N_scheda_5.pdf\\
%https://sanjayasubedi.com.np/deeplearning/tensorflow-2-linear-regression-from-scratch/\\
%https://towardsdatascience.com/get-started-with-tensorflow-2-0-and-linear-regression-29b5dbd65977

%\section{Logistic regression}
