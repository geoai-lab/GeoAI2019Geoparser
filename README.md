# Are We There Yet? Evaluating State-of-the-Art Deep Learning Geoparsers Using EUPEG as a Benchmarking Platform

### Introduction

This project restores three top ranked toponym resolution systems (geoparsers) reported on SemEval2019-Task12:

* DM NLP: 	ELMo + charBiLSTM + wordBiLSTM + CRF, Keras
* UniMelb: 	ELMo + wordBiLSTM + self-attention + softmax, Keras
* UArizona:	Glove + charLSTM + wordLSTM + CRF, Tensorflow

The source codes hosted on this repository are not the official source codes provided by task organization or task participants.
For each one of the three models, we restore the toponym detetcion part based on all information provided by papers. 
The toponym disambiguation is replaced with the Population heuristics.

### Repository organization

The whole repository contains codes for three parts:
* The toponym detection methods of three models;
* Population heuristics based toponym disambiguation;
* EUPEG corpus article examples;

### Geoparsing evaluation results

We test three restored models and eight other majorly used geoparsers using an Extensible and Unified Platform for Evaluating Geoparsers: EUPEG.
Here presents the performance tables of eight tested corpora:

<p align="center">
<img align="center" src="fig/TABLE1.png" width="520" height="300"/>
</p>
<p align="center">
<img align="center" src="fig/TABLE2.png" width="520" height="300"/>
</p>
<p align="center">
<img align="center" src="fig/TABLE3.png" width="520" height="300"/>
</p>
<p align="center">
<img align="center" src="fig/TABLE4.png" width="520" height="300"/>
</p>
<p align="center">
<img align="center" src="fig/TABLE5.png" width="520" height="300"/>
</p>
<p align="center">
<img align="center" src="fig/TABLE6.png" width="520" height="300"/>
</p>
<p align="center">
<img align="center" src="fig/TABLE7.png" width="520" height="300"/>
</p>
<p align="center">
<img align="center" src="fig/TABLE8.png" width="520" height="300"/>
</p>

