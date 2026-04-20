
# VAE with Weight-Based Character Decoding

This repository presents a PyTorch-based Variational Autoencoder (VAE) for grayscale image reconstruction together with an auxiliary experiment on character decoding from model weights.

The main objective of the project is to learn compact latent representations of images and reconstruct them accurately. In addition to the standard VAE pipeline, the project also investigates whether selected network weights can contain enough structured information to allow a small decoder network to predict a target ASCII character.

This makes the repository an experimental study at the intersection of image reconstruction, latent representation learning, and hidden symbolic information in neural parameters.

---

## Project Overview

The project consists of two connected parts:

1. **Variational Autoencoder (VAE)**  
   A convolutional VAE is trained on grayscale images to reconstruct the input image from a low-dimensional latent representation.

2. **Weight-Based Character Decoding**  
   During training, the weights of a selected layer are flattened and passed into a small decoder network. This decoder is trained to predict a target ASCII character from those weights.

The model is therefore optimized not only for image reconstruction, but also for preserving weight patterns that are useful for character prediction.

---

## Motivation

Neural networks are usually trained only for their main task, such as classification or reconstruction. In this project, the idea is extended by adding an auxiliary objective: encouraging selected parameters of the model to carry information that can be decoded as a symbolic signal.

Instead of explicitly storing a character in a separate variable or file, the experiment explores whether neural network weights themselves can become informative enough for a decoder to recover a predefined ASCII target.

This is a small but interesting experiment for studying:

- neural representation learning  
- hidden information in model parameters  
- auxiliary training objectives  
- interaction between reconstruction and symbolic decoding  

---

## Architecture

### 1. Variational Autoencoder

The VAE contains an encoder and a decoder.

#### Encoder
The encoder takes a grayscale image as input and processes it through convolutional layers to produce:

- a mean vector `mu`
- a log-variance vector `logvar`

A latent vector `z` is sampled using the reparameterization trick:

```python
z = mu + std * eps
````

This latent representation is then passed to the decoder.

#### Decoder

The decoder reconstructs the image from the latent vector using transposed convolution layers.

The reconstruction output is compared with the original image during training.

---

### 2. Payload Decoder

A second small neural network is used as a **payload decoder**.

Its role is to:

* take the flattened weights of a selected VAE layer
* process those weights as input
* predict a target ASCII character

This creates an auxiliary learning objective where selected parameters are encouraged to contain information useful for character decoding.

---

## Training Objective

The total loss combines three components:

### 1. Reconstruction Loss

Measures how close the reconstructed image is to the original image.

### 2. KL Divergence Loss

Regularizes the latent distribution so that it stays close to a standard normal distribution.

### 3. Payload Loss

Measures how well the payload decoder predicts the target ASCII character from the selected model weights.

The total training objective can be summarized as:

```python
total_loss = reconstruction_loss + kl_loss + payload_loss
```

This means the model is jointly trained for both:

* image reconstruction
* character prediction from weights

---

## What This Repository Demonstrates

This repository demonstrates that a VAE can be extended with an auxiliary decoding objective that operates on model parameters rather than directly on input or latent vectors.

In particular, it explores whether:

* image reconstruction quality can be maintained
* selected weights can preserve structured signals
* a small decoder can recover a predefined ASCII target from those weights

This should be viewed as an experimental representation-learning study rather than a finalized encoding system.

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

---

## Possible Use Cases

Although this repository is experimental, the ideas are related to broader topics such as:

* representation learning
* model interpretability
* auxiliary objectives in deep learning
* hidden information in neural parameters
* compact symbolic encoding experiments

---

## Future Work

Possible future extensions include:

* decoding multiple characters instead of a single target
* using different layers for weight-based decoding
* comparing which layer stores more useful information
* studying the trade-off between reconstruction quality and payload accuracy
* extending the method to larger datasets and deeper architectures
* testing robustness under pruning or quantization

---



## Notes

This repository does **not** present a complete communication or message-hiding framework.
Instead, it focuses on a smaller experimental question:

**Can selected neural network weights be trained so that a decoder can predict a target symbolic value while the main model still performs image reconstruction?**

That is the main research idea explored here.

---

## Author

**Emir Gençler**
Computer Engineering Student
Adana Alparslan Türkeş Science and Technology University


