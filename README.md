---
title: Aprendizaje federado jerárquico para la predicción energética
url: None
labels: [hierarchical federated learning, energy prediction]
dataset: [PLEIAData]
---

# PleiadesHVAC

**Paper:** Present in the `Paper` directory if you speak Spanish (`TFG_AlfredoMarquinaMeseguer.pdf`, plus the defence slides in `Presentacion.pdf`). Alternatively you can read the Extended Abstract section of this file.

**Author:** Alfredo Marquina Meseguer

**Abstract:** This bachelor's thesis implements hierarchical federated learning (HFL) for the prediction of power consumption of the PLEIADES building in the University of Murcia with the Flower framework. Since there is no proper way to implement the aggregator element needed for implementing HFL, the proposed architecture simulates an aggregator with a client and a server, each one present in a different federation, simulating a three-level HFL architecture.

## About this baseline

**What's implemented:** A 3-level HFL setup using the Flower app `pleiadesHVAC` (root directory) as the two higher levels and the Flower app `edge` (directory `pleiadesHVAC_edge/`) as the lower level.

**Datasets:** [PLEIAData](https://zenodo.org/records/7620136)

**Hardware Setup:** Desktop computer running Ubuntu with 32GB of RAM, CPU Ryzen 7 and GPU NVIDIA GeForce RTX 3060

**Contributors:** Juan de Dios Hernández Kakauridze (original owner of the codebase adapted to HFL)


## Experimental Setup

**Task:** Next-step energy consumption prediction from historic data (target variable `dif_cons_real`).

**Models:** Purpose-built lightweight models based on different technologies: transformers, CNN (ConvLSTM2D), LSTM and GRU. They can be found in `pleiadesHVAC/models/` (mirrored in `pleiadesHVAC_edge/edge/models/`).

**Dataset:** PLEIAData, a series of IoT temperature, consumption and HVAC measurements of the PLEIADES building in the Universidad de Murcia along all of 2021. The data from all devices has been simplified into a series of hour-by-hour readings for each building block, as well as meteorological data from a nearby weather station. The preprocessing notebook is `data/scripts/preprocesado.ipynb`; the resulting partitions live in `data/datasets/` (`buildingA-data`, `buildingB-data`, `buildingC-data` and `combined-data`).

**Training Hyperparameters:** Each of the five models was evaluated in both an FL and an HFL scenario. In HFL the three local clusters use 3, 7 and 5 nodes respectively (15 nodes in total, see `num_nodes_subfed` in `pleiadesHVAC/server_app.py`); the FL baseline uses a single federation of 15 nodes over `combined-data`. Common settings (`[tool.flwr.app.config]` in each `pyproject.toml`): history window of 12 hours, 2 local epochs, batch size 32, learning rate 0.005. Each Flower invocation performs a single server round and checkpoints the global model to `state/`, so `gather_metrics.sh` repeats every configuration 10 times to obtain the 10 global rounds reported below.


## Environment Setup
```bash
# Create the virtual environment
pyenv virtualenv 3.11.11 <name-of-your-baseline-env>

# Activate it
pyenv activate <name-of-your-baseline-env>

# Install the baseline
pip install -e .
pip install -e pleiadesHVAC_edge/
```
## Running the Experiments

```bash
# All of the experiments are encapsulated in the gather_metrics.sh file

pyenv activate <name-of-your-env>

# The script expects these directories to exist before the first run
mkdir -p state/results data/metrics

./gather_metrics.sh
```

Results are written to `metrics/<model>/<run>/{HFL,FL}/` (JSON metrics plus the model weights). Note that `state/`, `data/metrics/` and `metrics/` are git-ignored, so the raw outputs are not part of this repository. The plots and tables are produced from them with `data/scripts/metrics.ipynb`.


## Appendix: Extended Abstract

In the contemporary era of smart cities and sustainable development, optimization of energy consumption within buildings has emerged as a critical socio-economic and environmental priority. Consequently, the capacity to accurately predict energy demand patterns constitutes a foundational pillar in reducing environmental impacts and operational costs.

With the proliferation of Internet of Things (IoT) infrastructures, modern smart buildings are continuously monitored by heterogeneous wireless sensor networks. These devices capture continuous data streams obtaining all kinds of information. To transform this raw data into actionable intelligence, Deep Learning (DL) methodologies have been widely adopted due to their capability to model non-linear, complex spatio-temporal dependencies inherent to multi-variable time-series forecasting.

However, the traditional deployment of DL frameworks relies heavily on centralized cloud-computing paradigms where all raw data collected by localized edge sensors must be continuously transmitted over the network to a monolithic cloud repository where the model is trained. This operational blueprint introduces severe structural bottlenecks: data privacy, due to the highly sensitive nature of some data; cybersecurity risks, because an attack on the centralized server may expose all data as well as being susceptible to man-in-the-middle attacks; and bandwidth constraints, because the continuous flow of data may introduce excessive overhead in the network.

To definitively alleviate these limitations, Federated Learning (FL) has emerged as a disruptive decentralized machine learning paradigm. In standard FL, data remains strictly localized on the generating edge devices or clients. Instead of transmitting raw datasets, clients train a local instance of the model using their proprietary data and exclusively upload the resulting model parameters (weights and biases) to a centralized server. The server aggregates these local updates utilizing a so-called aggregation algorithm (such as Federated Averaging or FedAvg) to update a shared global model, which is subsequently redistributed back to the clients. While FL natively guarantees data privacy and drastically minimizes network bandwidth consumption, standard two-tier FL topologies (client-to-server) may face scalability and compatibility issues when applied to already existing infrastructures such as the three-layer topologies seen in many edge computing environments used nowadays, especially those used in medical settings.

To bridge this gap, Hierarchical Federated Learning (HFL) has been proposed in the literature as a natural architectural evolution. HFL introduces intermediate aggregation nodes, typically deployed at the edge layer, establishing a multi-tier hierarchical infrastructure. HFL achieves this goal with the addition of the aggregator node, an intermediate stackable element between servers and clients.

This work follows a current line of investigation by the University of Murcia on the application of FL to the prediction of energy consumption using the PLEIAData dataset, which compiles a wide range of climatization and energy consumption data gathered through the year 2021 in the smart building PLEIADES, also from the University of Murcia. This Bachelor's Thesis tries to give continuity to the work started by Hernández with the implementation and testing of a three-tier HFL federation. The project has been developed in Python using the most widely used FL framework, Flower, together with TensorFlow for the DL models.

This intermediate tier acts as a localized buffer and regularizer. Instead of immediately averaging highly divergent local updates at the global scale, edge aggregators consolidate the updates of logical or physical clusters (e.g., specific floors or independent building blocks). This allows the network to capture regional contextual commonalities before abstracting the model weights to the global cloud server. As a side effect, this architecture reduces the communication frequency between local edge facilities and distant cloud architectures, which tend to be more limited and costly.

Concurrently, selecting the optimal underlying neural network architecture for edge-oriented time-series forecasting remains a critical design challenge. Traditional Long Short-Term Memory (LSTM) networks have long been the gold standard for sequence modeling due to their gating mechanisms designed to mitigate the vanishing gradient problem: LSTMs utilize three distinct gating structures (input, forget, and output gates) fed by two distinct channels, one for traditional input and another one for memory information.

However, the search for greater efficiency has led to the creation of the Gated Recurrent Unit (GRU) architectures. A GRU couples the forget and input gates into a single update gate and merges the cell state and hidden state, operating with only two gates (update and reset gates) and fusing both channels into one. Mathematically, the reduction in internal gates translates into a significantly lower parameter count.
This architectural simplification directly implies reduced RAM utilization, lower energy consumption during local processing cycles, and faster execution times. Despite having fewer parameters, the literature demonstrates that GRUs maintain a competitive performance profile closely matching LSTMs in specific short-to-medium-term regression tasks.

Furthermore, highly sophisticated architectures like Convolutional LSTMs (ConvLSTM2D), which capture coupled spatial and temporal dynamics, and Transformer networks based on Self-Attention mechanisms, which offer state-of-the-art accuracy in centralized environments, have been tested to evaluate their performance in HFL environments, following the work of Hernández Kakauridze, who implemented these models.

The experimental validation of this research relies on PLEIAData, a comprehensive multi-variable dataset gathered throughout the year 2021 from the PLEIADES experimental smart building located at the University of Murcia. This dataset builds upon foundational frameworks from prior work, maintaining consistency in data definitions while adapting the workflow to optimized pipelines. PLEIAData records high-frequency readings across diverse architectural zones, fundamentally divided into three autonomous structural building segments: Block A, Block B, and Block C.

To guarantee stable weight optimization during local model updates, a rigorous data preprocessing pipeline was developed by Hernández Kakauridze in the previous work. After carefully selecting the included dataset variables, a global Min-Max Normalization was applied. Following normalization, the data is transformed into three-dimensional arrays suitable for recurrent operations. In previous iterations of this research, massive history windows (such as 168 hours) were enforced; to accommodate resource-constrained IoT requirements, this study reduced the temporal window size down to 12 hours. This reduction preserves immediate context while reducing the historical parameter steps, striking an optimal balance between context preservation and memory footprint. This data was already horizontally segmented across clients representing localized blocks in the original PLEIAData dataset, each segment being used as a dataset for a local cluster.

A primary milestone of this thesis consists of the complete migration and re-implementation of the legacy simulation code into the modern Flower framework (version 1.30). However, Flower's native simulation engine operates strictly on a traditional two-tier hub-and-spoke layout. It does not inherently support intermediate aggregation rings. To successfully bypass this operational restriction without introducing networking conflicts, a decoupled execution flow was engineered.

The three-tier Hierarchical Federated Learning (HFL) layout is instantiated by configuring intermediate ClientApp and ServerApp routines to act as proxy edge aggregators for specific building blocks. In this workflow, local clients train directly within their local cluster boundaries (their own block). Once local updates settle, the intermediate proxy aggregators consolidate their cluster-specific weights using local instances of the Federated Averaging (FedAvg) algorithm. These block models are then sent to the root cloud server for final global averaging. This design effectively overrides Flower's default constraints, establishing an isolated, hierarchical paradigm tailored to the physical building subdivisions.

To accurately benchmark the performance gains of structural model optimization, the newly proposed models are directly compared against three established baselines inherited from previous research lines:

- Long Short-Term Memory (LSTM): A classic sequence model utilizing input, forget, and output gates to track temporal consumption states across sequences. It serves as the primary predictive standard.
- Convolutional LSTM (ConvLSTM2D): A hybrid model that stacks spatial convolutional operations inside recurrent cell transitions, designed to isolate inter-variable dependencies along the sequence path.
- Transformer Networks: An architecture built entirely around Multi-Head Self-Attention layers.

The integration of Gated Recurrent Unit (GRU) alternatives directly addresses the necessity of maximizing edge execution efficiency within resource-constrained IoT settings. As previously explained, GRU cells reduce their internal complexity compared to their LSTM counterparts. Two specific GRU variants were designed, compiled via TensorFlow/Keras, and integrated into the HFL workflow:

- Standard GRU Architecture: Mirroring the inherited LSTM architecture, this variant simply substitutes the LSTM layers with GRU layers. The final product consists of two GRU layers followed by two dense layers and a linear output layer. Regularization was applied using LayerNormalization and Dropout to prevent overfitting.
- Simplified GRU Architecture: This configuration consists of only two GRU layers between the input and the output layer. This pruning optimizes the model as much as possible.

The comprehensive evaluation of the decentralized framework was conducted through controlled simulations consisting of 10 synchronized global training rounds, with local edge training configured at 2 internal epochs per round. For each model presented, two distinct executions were made: the first one for the evaluation of the model in an HFL paradigm and the second one for the evaluation of the same model in an FL paradigm for comparison. For the HFL experiment three distinct local clusters were created to mirror the real-world organization of the PLEIADES building into blocks, while the FL experiment used a combined dataset of all three blocks. Performance was benchmarked using statistical regression metrics, primarily the Coefficient of Determination (R2 Score), Mean Absolute Error (MAE), and Mean Squared Error (MSE).

In the first experimental phase, each one of the local clusters presented its own performance alongside the aggregated model's performance. Local cluster performance seemed to be ordered by the quality of their local dataset, which consistently was: A first, B a close second and C last. All models showed a solid learning curve, barring the ConvLSTM2D model, which showed signs of divergence even though its metrics were similar (MSE = 0.028) to those obtained in previous works (MSE = 0.03) where it showed a healthy learning curve.

The second experimental phase compared all the different aggregated models during their training. The outcomes obtained by this experiment resulted in a clear win for the simplified GRU architecture. This architecture obtained the best results in most metrics measured (R2 = 0.75, MSE = 0.004 and MAE = 0.038). On the same note, the rest of the RNN architectures (LSTM and GRU), as well as the attention model (Transformer), showed intertwined learning curves obtaining a similar performance. Finally, ConvLSTM2D continued with its already described behavior.

Finally, the third experimental phase compared the data obtained with these HFL models with their FL counterparts. This phase showed that the structural implementation of the three-tier HFL paradigm outperformed the FL configuration in this scenario, whereas the expected result was a slight performance degradation inherent to aggregation functions like the one used, FedAvg. HFL models started with a higher error rate but quickly caught up with their FL counterparts, up to the point of overcoming them. The only exception to this phenomenon is again ConvLSTM2D, which had a close but higher degree of divergence in the HFL paradigm. This behavior can be explained by the reduction of its input dimensionality, as similar results were obtained in both paradigms.

It is hypothesized that in standard FL the worse performance comes from the variations between blocks, which introduce a level of high statistical heterogeneity (non-IID data) across clients, inducing model shaking or gradient cancellations. Under the proposed HFL layout, the intermediate proxy edge aggregators act as a structural localized regularizer, allowing each local cluster to approximate a local model before aggregating into the global one.

Finally, through the measurement of the Max Error of the models, it was found that the models obtained catastrophic errors even while maintaining their high performance in other metrics. It is hypothesized that this behavior is caused by the existence of a few non-IID evaluation examples that could be caused by the processing of the dataset or by the inherent nature of climate data. Either way, further research would be required to reach a satisfactory conclusion.

In conclusion, this Bachelor's Thesis demonstrates the viability and technical advantages of implementing a three-tier HFL architecture for short-term energy prediction within the smart building of PLEIADES. The evaluation validates that lightweight recurrent configurations, specifically the proposed simplified GRU, offer an optimal design pattern for resource-constrained environments.

Future research paths could focus on approaching a real implementation through more realistic simulations (for example, with Docker containers deployed in a local network) or on the implementation of more features, either aimed at the improvement of the IoT environment (like online training) or at the HFL architecture (through the implementation of heterogeneous hierarchical federated learning, where a given server or aggregator may have both aggregator and edge nodes as clients).
