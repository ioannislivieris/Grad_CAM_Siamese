# Explainability on Siamese Networks

This repository presents the implementation code for paper: **Explainable Image Similarity: Integrating Siamese Networks and
Grad-CAM**. The manuscript is published in *Journal of Imaging*.

**Abstract:** With the proliferation of image-based applications in various domains, the need for accurate and interpretable image similarity measures has become increasingly critical. Existing image similarity models often lack transparency, making it challenging to understand the reasons why two 
images are considered similar. In this paper, we propose the concept of explainable image similarity, where the goal is the development of a framework, which is capable of providing similarity scores along with visual intuitive explanations for its decisions. To achieve this, we present a new approach, which integrates Siamese Networks and Grad-CAM for providing explainable image similarity and discuss the potential benefits and challenges of adopting this approach. In addition, we provide a 
comprehensive discussion about factual and counterfactual explanations provided by the proposed framework for assisting decision making. The proposed approach has the potential to enhance the interpretability, trustworthiness and user acceptance of image-based systems in real-world image similarity applications.

**Keywords:** Explainable AI; Siamese networks; Grad-CAM; Interpretability.

**Cite:** Livieris IE, Pintelas E, Kiriakidou N, Pintelas P. [Explainable Image Similarity: Integrating Siamese Networks and Grad-CAM](https://doi.org/10.3390/jimaging9100224). Journal of Imaging. 2023; 9(10):224. 

<br/>

## Contents

<!--ts-->
   * [Getting started](#getting-started)
        * [Setting up environment](#setting-up-environment)
        * [Upload dataset](#upload-dataset)
        * [How to run](#how-to-run)
   * [Contact](#mailbox-contact)
<!--te-->

<br/>

## Getting started

## Setting up environment

The provided setup instructions assume that anaconda is already installed on the system. To set up the environment for this repository, run the following commands to create and activate an environment named 'pytorch_siamese'. (The command takes a while to run, so please keep patience):

```
conda env create -f environment.yml
conda activate pytorch_siamese
```

## Upload dataset

The expected format for both the training and validation dataset is the same. Image belonging to a single entity/class should be placed in a folder with the name of the class. The folders for every class are then to be placed within a common root directory (which will be passed to the trainined and evaluation scripts) located in directory: ```Data``. The folder structure is also explained below:


```
My_dataset
|── class 1
|   ├── Image 1
|   ├── Image 2
|   ├── Image 3
|   ├── ...
|── class 2
|   ├── Image 1
|   ├── Image 2
|   ├── Image 3
|   ├── ...
|── ...
|── class k
|   ├── Image 1
|   ├── Image 2
|   ├── Image 3
└── └── ...
```

The training similarity instances as follows: *for each image of each class, one image from the same class is randomly selected and this pair is assigned with label zero (0). Next, another image from a different class is randomly
selected and this pair is assigned with label one (1).*

### How to run

- Update `config.yml` 

- Run notebook: `01.Train_Siamese_model.ipynb` for training a Siamese network. 

- Run notebook: `02.Inferences.ipynb` for conducting a use on two images, for calculating the similarity scores as well as factual and counterfactual explanations.

<br/>

## :mailbox: Contact

Ioannis E. Livieris (livieris@upatras.gr)
