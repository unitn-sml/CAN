# Constrained Adversarial Networks

## Welcome to the CAN repository

Check the [README](src/README.md) in the src directory for understanding on how to use this repository.

Constrained Adversarial Networks has been presented in the [paper](https://papers.nips.cc/paper/2020/hash/a87c11b9100c608b7f8e98cfa316ff7b-Abstract.html).


### Abstract

> Generative Adversarial Networks (GANs) struggle to generate structured objects like molecules and game maps. The issue is that structured objects must satisfy hard requirements (e.g., molecules must be chemically valid) that are difficult to acquire from examples alone. As a remedy, we propose Constrained Adversarial Networks (CANs), an extension of GANs in which the constraints are embedded into the model during training. This is achieved by penalizing the generator proportionally to the mass it allocates to invalid structures. In contrast to other generative models, CANs support efficient inference of valid structures (with high probability) and allows to turn on and off the learned constraints at inference time. CANs handle arbitrary logical constraints and leverage knowledge compilation techniques to efficiently evaluate the disagreement between the model and the constraints. Our setup is further extended to hybrid logical-neural constraints for capturing very complex constraints, like graph reachability. An extensive empirical analysis shows that CANs efficiently generate valid structures that are both high-quality and novel.


### Environment

We run out experiments in the following environment:
- Ubuntu 16.04 LTS
- Cuda 10.0
- libcudnn7=7.4.1.5-1
- libcudnn7-dev=7.4.1.5-1
- Nvidia Driver: 410.78
- GPU: GTX-1080 Ti
