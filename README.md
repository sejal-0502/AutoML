**Project Title :**
Bayesian Optimization and Regularized Evolution to optimize the architecture of CNN on NAS-bench-201 tabular benchmark.

**Project Description :**
The goal of the project is to optimize the arvhitecture of a CNN on the NAS-Bench-201 tabular benchmark.
For this above task, Bayesian optimization and Regularized Evolution has been used.
NAS-Bench-201 tabular benchmark is used to avoid training the architectures.
In NAS, the search is generally done on the small dataset (eg. CIFAR-10) and then found cells are transferred to a more expensive dataset (eg. ImageNet).
For an offline evaluation, NAS is used. The performance is computed on ImageNet of all incumbents.

***What is NAS-Bench-201 :***
*It defines a cell-based neural architectures as graphs with : 4 nodes, 6 edges.
Each edge has 5 possible operations - zeroize, skip conneections, 1x1 and 3x3 convolutions, and 3x3 average pooling. This collectively leading to 15,626 unique architectures.*

**Installed Dependencies :**
Python Libraries :
1. numpy
2. matplotlib
3. scipy
4. sklearn
5. pytorch
6. ConfigSpace 0.6.1

**Acknowledgments :**
The project was build using the python libraries listed above. 
Additionally, for package installations and environment management, 'Anaconda' has been used.
Special thanks to the open-source community for making such a great contributions and for making them available.
