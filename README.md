# Are We There Yet? Evaluating State-of-the-Art Deep Learning Geoparsers Using EUPEG as a Benchmarking Platform

### Introduction

This project systematically evaluates the top three geoparsers from the SemEval2019-Task12 competition:

* DM NLP: 	ELMo + charBiLSTM + wordBiLSTM + CRF, Keras
* UniMelb: 	ELMo + wordBiLSTM + self-attention + softmax, Keras
* UArizona:	Glove + charLSTM + wordLSTM + CRF, Tensorflow

The source codes hosted on this repository are not the official codes provided by the SemEval2019-Task12 organizers but our own implementations based on the descriptions of the papers.
For each one of the three models, we implement the toponym detetcion part based on all the information provided by their papers. The toponym resolution part of all three geoparsers is based on the population heuristics.

### Repository organization

The whole repository contains codes for three parts:
* The toponym detection methods of three models;
* The population heuristics for toponym resolution;
* EUPEG corpus article examples;

### Geoparsing evaluation results

We test three restored models and eight other existing geoparsers using our benchmarking platform EUPEG. The source code of EUPEG can be found at: https://github.com/geoai-lab/EUPEG

The performance tables of all geoparsers on the eight tested corpora are provided as below:

<p align="center">
<img align="center" src="fig/TABLE11.png" width="520" height="300"/>
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
<img align="center" src="fig/TABLE88.png" width="520" height="300"/>
</p>

