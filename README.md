# DFDG: Federated GNN Learning for Traffic Forcasting 
by Muhammad Usman

DFDG is introduced as a pioneering fusion of
federated graph neural networks (GNNs) with dynamic graphs
and unsupervised learning for dynamic traffic forecasting in
urban settings. Utilizing self-organizing map clustering, speed
patterns are clustered at various time intervals, and resulting
clusters are assigned to Federated Learning participants. Within
the federated setup, participants use neural ordinary differential equations coupled with attention spatial-temporal graph
convolution networks to collectively form the DFDG architecture. During training, participants contribute local updates to
a global GNN model, with performance weightage reflecting
their reliability. This dynamic weighting system influences each
participant’s significance in subsequent communication rounds,
promoting adaptive learning across the federated network. We
group homogeneous participants based on their previous round
performance and data distribution. Grouping contributes to the
overall efficiency and effectiveness of federated learning systems,
enhancing model generalization, reducing communication overhead, preserving privacy by aggregating updates from homo-
geneous client groups, and allowing adaptability to local data
characteristics. Extensive benchmark testing validates DFDG’s
superiority in traffic forecasting, surpassing current state-of-the-
art methods. Our method not only advances traffic forecasting
but can be adapted for broader discourse on real-time dynamic
graph applications

# Datasets
1. MetrLA : A traffic forecasting dataset based on Los Angeles
    Metropolitan traffic conditions. The dataset contains traffic
    readings collected from 207 loop detectors on highways in Los Angeles
    County in aggregated 5 minute intervals for 4 months between March 2012
    to June 2012.

    For further details on the version of the sensor network and
    discretization see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" (https://arxiv.org/abs/1707.01926)`
   
2. PEMS-BAY: This traffic dataset is collected by California Transportation Agencies (CalTrans)
    Performance Measurement System (PeMS). It is represented by a network of 325 traffic sensors
    in the Bay Area with 6 months of traffic readings ranging from Jan 1st 2017 to May 31th 2017
    in 5 minute intervals.

    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" (https://arxiv.org/abs/1707.01926)`_
    """
# Requirements
-  torch
-  torchdifeq
-  FedLab 1.2
-  python3.9
-  numpy, scipy, pandas, matplotlib, seaborn, cuda 11

# Performance Results
![image](https://github.com/GreenPatterns/DFDG/assets/15605985/204d7178-c349-4b9b-b07b-8b150f2e1f31)

![image](https://github.com/GreenPatterns/DFDG/assets/15605985/d3db0642-70a5-4120-a4ab-3563ad667579)

# Forecasted Results
![image](https://github.com/GreenPatterns/DFDG/assets/15605985/c51f6046-5a36-4d4c-aa01-e52c34a5307b)


