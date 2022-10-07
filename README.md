# HumanAITeaming
Code base for NeurIPS 2022 publication titled Human-AI Collaborative Bayesian Optimsiation

We have conducted all our experiments in a server with the following specifications.
• RAM: 16 GB
• Processor: Intel (R) Xeon(R) W-2133 CPU @ 3.60GHz
• GPU: NVIDIA Quadro P 1000 4GB + 8GB Memory
• Operating System: Ubuntu 18.04 LTS Bionic
We recommend to install the following dependencies for the smooth execution of the
modules.
• Python - 3.6
• scipy - 1.2.1
• numpy - 1.16.2
• pandas - 0.24.2
• scikit-learn - 0.21.2
For running the experiments, please follow the steps mentioned below.
1. Navigate to directory named “HATBO”, containing the source files required
for conducting the experiments.
2. $ python ExpAI_KernelOpt_Wrapper.py -d <dataset/function>
-t <number_iterations> -t <>
<dataset/function> is the dataset or the function to be used (ackley, sonar,
wdbc, ecoli, · · · )
For example: $ python ExpAI_KernelOpt_Wrapper.py -d sonar
