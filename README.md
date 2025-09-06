# machine-learning-projects
Collection of my machine learning projects and algorithms, covering both classical methods and modern deep learning approaches.  

⚠️ Note: Projects from my full-time professional work experience are **not included here** due to confidentiality.  

**Highlights**: Projects span **LLMs, recommendation systems, natural language processing (NLP), computer vision (CV), reinforcement learning (RL), and graph neural networks (GNNs)**, demonstrating a broad skill set across applied machine learning domains.

## Project 1: Building LLM SMS Spam Classifier from scratch
Code in: [llm/llms-from-scratch/](./llm/llms-from-scratch/)

Run with:  
```bash
python llm/llms-from-scratch/funingtuning-for-text-classification.py
```

**Build LLMs model structure from scratch** and then **fine-tune** a pretrained **GPT-2** model with a classification head to detect **spam SMS messages** using the UCI SMS Spam Collection dataset.


## Highlights
- Dataset: UCI **SMS Spam Collection** (~5.5k messages)  
- Model: **GPT-2** frozen + trainable classification head  
- Tokenization: GPT-2 BPE (`tiktoken`)  
- Training: PyTorch `AdamW`, accuracy/loss tracking  
- Output: Classify text as **spam** or **ham**  


## Result
before finetuning:
<img width="221" height="54" alt="image" src="https://github.com/user-attachments/assets/76f9d8d8-cca0-4e5a-99ab-30a1eea1ceff" />
after finetuning:
<img width="254" height="61" alt="image" src="https://github.com/user-attachments/assets/b3526eb2-89ef-4b5d-bb23-86e7f141b2ef" />


## Project 2: Instruction-Tuned LLM for Text Generation
Code in: [llm/llms-from-scratch/](./llm/llms-from-scratch/)

Run with:  
```bash
python llm/llms-from-scratch/finetuning-to-follow-Instructions.py
```

This project explores **instruction tuning** for large language models (LLMs).  
We start with a pretrained GPT-2 model and fine-tune it on a dataset of human-written instructions (1.1k) and responses to align the model with task-following behavior.


## Dataset
- **Source**: [instruction-data.json](https://github.com/lixinglong806/machine-learning-projects/blob/main/llm/llms-from-scratch/instruction-data.json)  
- **Size**: ~1,000+ instruction–response pairs  
- **Structure**:
  - `instruction`: task description  
  - `input`: optional context  
  - `output`: ground-truth response  

Example:
```json
{
  "instruction": "Translate 'good morning' to French.",
  "input": "",
  "output": "Bonjour"
}
```

## Model
- **Base model**: GPT-2 (124M / 355M / 774M / 1558M)  
- **Tokenizer**: GPT-2 BPE (`tiktoken`)  
- **Objective**: Fine-tune GPT-2 with causal LM loss, supervised by *(instruction + input → output)* pairs  
- **Training setup**:  
  - Optimizer: AdamW  
  - Context length: 1024  
  - Epochs: 2–3  
  - Batch size: 8  

## Tech Stack
- **Framework**: PyTorch  
- **Data handling**: `torch.utils.data.Dataset`, `DataLoader`  
- **Tokenizer**: `tiktoken`  
- **Pretrained weights**: HuggingFace GPT-2 (via helper `download_and_load_gpt2`)

## Evaluation (LLM-as-a-Judge via Ollama)

- **Judge model:** `llama3.2` (running locally with **Ollama**)
- **Protocol:** rubric-based scoring of model responses on the **test set** (deterministic settings: `seed=123`, `temperature=0`, `num_ctx=2048`)
- **Result: <img width="230" height="38" alt="image" src="https://github.com/user-attachments/assets/54595b65-fd47-4793-ad47-b7794562ba0f" />





## Project 3: Building an Industrial-Scale E-commerce Recommendation System from Scratch
code in: [e-commerce-rec-sys-from-scratch/code](./e-commerce-rec-sys-from-scratch/code)

**Introduction**  
Designed and implemented the end-to-end recommendation workflow for a newly launched e-commerce platform, supporting product discovery and personalization at scale.

**System Design**  
- **Data Layer**: Integrated product catalog (~330K items) and user activity logs (clicks, exposures) into a centralized data warehouse.  
- **Processing Layer**:  
  - Offline: Built item profiles & user profiles, content-based similarity, collaborative filtering, CTR/conversion prediction models.  
  - Online: Real-time user behavior analysis for cold-start and session-based recommendations.  
- **Recommendation Layer**:  
  - Early stage: Item cold start handled via content-based similarity; user cold start mitigated by real-time behavioral signals and popular/new item strategies.  
  - Later stage: Shifted to hybrid offline + online recommendation using large-scale behavior logs for personalization.  

**Tech Stack**  
- **Data Infrastructure**: Hadoop (HDFS), Kafka, Spark  
- **Modeling**: Collaborative Filtering, Content-based filtering, CTR prediction (Logistic Regression, Tree-based models)  
- **Engineering**: Python, SQL, PySpark, Scikit-learn  
- **Serving**: Real-time recommendation service with log streaming + offline batch updates  

**Impact**  
Established a scalable recommendation foundation that evolved from cold-start strategies to personalized ranking, improving product discovery and user engagement.



## Project 4: Personalized Recommendation for Advertising
code in: [Personalized-Ad-RecSys](./Personalized-Ad-RecSys)

**Introduction**  
Built an end-to-end **advertising recommendation system** based on JD.com CTR prediction dataset (26M ad impression logs, 1M users, 800K ads).  
The system integrates **offline recall, CTR prediction, and real-time recommendation** to deliver personalized ads.

**System Design**  
- **Data Layer**: Integrated user profiles, ad features, behavior logs (~700M records) into HDFS.  
- **Processing Layer**:  
  - **Offline**: 
    - Trained ALS collaborative filtering model for user–category preferences.  
    - Built CTR prediction models with Logistic Regression.  
    - Cached recall sets (~500 items/user) and features in Redis.  
  - **Real-time**:  
    - Collected user logs via Kafka.  
    - Updated user features and recall sets dynamically.  
    - Combined with CTR model for real-time top-N recommendation.  
- **Recommendation Layer**:  
  - Solved **cold start** with profile-based recall and hot/new ads.  
  - Provided **personalized CTR-based ranking** for relevance.  

**Tech Stack**  
- Data Infrastructure: **HDFS, Kafka, Flume**  
- Data Processing: **Spark SQL, Spark ML**  
- Models: **ALS Collaborative Filtering, Logistic Regression for CTR**  
- Serving & Caching: **Redis**  
- Programming: **Python, Scala**  

**Results**  
- Established a scalable pipeline from log collection → offline/online processing → ad recommendation.  
- Demonstrated improvement in ad CTR prediction and personalized targeting compared to non-personalized baselines.


## Project 5: Language Models for Music Generation 🎶
code in: [RNN_Music_Generation.ipynb](./RNN_Music_Generation.ipynb)


**Description**: This project explores **sequence modeling with Recurrent Neural Networks (RNNs)** for **music generation**.  A character-based RNN was trained on symbolic music sequences to generate new compositions.

The project code demonstrates:

- Preparing and preprocessing music sequence data.  
- Building an RNN with Keras/TensorFlow to model sequential dependencies.  
- Training the RNN to generate new symbolic music.  
- Sampling and interpreting generated sequences.



**Skill sets involved in the project:**

- **Deep Learning for Sequences**: Implemented an RNN for generative modeling.  
- **Data Preprocessing**: Encoded symbolic music as sequences for training.  
- **Model Training**: Tuned hyperparameters, monitored loss, and improved sequence generation quality.  
- **Generative AI**: Demonstrated sequence generation from learned distributions.  
- **Frameworks**: Python, TensorFlow/Keras, NumPy, Matplotlib.  



**Example Result:**

The model generates short symbolic sequences that resemble simple music patterns.  
Future work could extend this to LSTM/GRU architectures, or generate MIDI for richer compositions.



## Project 6: Image Classification on CIFAR-10 with a Convolutional Neural Network (CNN) from Scratch
code in: [CNN-scratch-pytorch.ipynb](./CNN-scratch-pytorch.ipynb)

**Description**:This project implements a **Convolutional Neural Network (CNN)** in **PyTorch**, trained and tested on  image classification datasets. The goal was to build a CNN from the ground up and understand  how each layer contributes to feature extraction and classification.



- Built a CNN architecture from scratch in PyTorch.  
- Trained the model on image datasets (e.g., MNIST / CIFAR-10).  
- Monitored training and validation accuracy across epochs.  
- Visualized loss curves and sample predictions.  



**Skill Sets:**

- **Deep Learning Fundamentals**: CNN architecture (Conv, Pooling, Fully Connected layers).  
- **PyTorch Implementation**: Defined custom layers, training loops, optimizers.  
- **Regularization**: Applied dropout and weight decay to reduce overfitting.  
- **Evaluation**: Accuracy, confusion matrix, misclassified samples.  
- **Visualization**: Training/validation curves, feature maps.  
- **Tools & Libraries**: Python, PyTorch, NumPy, Matplotlib, Seaborn.  



Example Results

- Achieved high accuracy on MNIST / CIFAR-10 classification task.  
- Training curves showed smooth convergence with minimal overfitting.  
- Visualized CNN feature maps revealed hierarchical pattern learning.





## **Project 7 with detailed report: Data Pre-processing and Non-Parametric Classification Algorithms**
code in: [ECE657A-projects/A1](./ECE657A-projects/A1)
report: [a1-submitted.pdf](./ECE657A-projects/A1/a1-submitted.pdf)

**Description**: Implemented data preprocessing, feature engineering, and classification algorithms (KNN, Decision Trees, Random Forests) on Wine Quality and Abalone datasets, with comparative evaluation and visualization of results.



**Skill sets involved in the project** :



- **Data Preprocessing**: handled missing values, detected outliers, applied normalization (Z-score, MinMax), and balanced imbalanced datasets using SMOTE.
- **Feature Engineering**: statistical analysis (mean, variance, skewness, kurtosis), feature selection, and categorical encoding (one-hot vs. label encoding).
- **Machine Learning Algorithms**: implemented and tuned **K-Nearest Neighbors (KNN)**, **Decision Trees**, and **Random Forests** using scikit-learn.
- **Model Evaluation**: performed 5-fold cross-validation, hyperparameter tuning with GridSearchCV, and compared accuracy across models/datasets.
- **Visualization & Reporting**: created pair plots, accuracy-vs-parameter plots, heatmaps, and summary tables to present experimental results clearly.
- **Tools & Libraries**: Python, Jupyter Notebook, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn.



## **Project 8 with detailed report: Representation Learning and Regression Models**

code in: [ECE657A-projects/A2](./ECE657A-projects/A2)
report: [a2-submitted.pdf](./ECE657A-projects/A2/a2-submitted.pdf)

**Description**: Applied dimensionality reduction techniques (t-SNE, PCA, Isomap) and regression algorithms (KNN Regressor, Random Forest Regressor, Gradient Boosting Regressor) on Forest Fires, Wine Quality, and Abalone datasets, with comparative evaluation of RMSE performance and visualization of feature representations.



**Skill sets involved in the project**:



- **Feature Engineering**: one-hot encoding of categorical features, sparse element binning, feature removal (e.g., low variance or outlier-dominated features), and Z-score normalization.
- **Representation Learning**: implemented **t-SNE** for visualization, **PCA** for linear dimensionality reduction, and **Isomap** for non-linear manifold learning.
- **Regression Algorithms**: trained and tuned **KNN Regressor**, **Random Forest Regressor**, and **Gradient Boosting Regressor** using scikit-learn.
- **Model Evaluation**: performed 5-fold cross-validation, hyperparameter tuning with GridSearchCV, and compared RMSE across datasets and feature spaces.
- **Visualization & Reporting**: created t-SNE and Isomap 2D plots, PCA scree plots, RMSE vs. number of components graphs, and comparative tables to summarize best results.
- **Tools & Libraries**: Python, Jupyter Notebook, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn.





## **Project 9 with detailed report: Clustering and Classification with Deep Learning on “FashionMNIST with a Twist!”**
code in: [ECE657A-projects/A3](./ECE657A-projects/A3)
report: [a3-submitted.pdf](./ECE657A-projects/A3/a3-submitted.pdf)


**Description**: Built and evaluated multiple deep learning models on the **Fashion MNIST dataset**, exploring optimization strategies, deeper architectures, convolutional layers, dropout, residual connections, and data augmentation. The best model achieved **98.63% test accuracy**, with additional analysis on feature representations through clustering and visualization.



**Skill sets involved in the project**:



- **Data Preprocessing**: normalized pixel values (0–255 → 0–1), created training/validation/test splits, and ensured balanced datasets.

- **Model Development**: implemented and trained with **GPU** a series of deep learning architectures including:

	

	- Base fully connected model
	- Optimized model with Adam optimizer & early stopping
	- Deeper fully connected network (pyramid-like structure)
	- CNNs with additional convolution/pooling layers
	- Dropout-regularized CNNs to mitigate overfitting
	- Residual network block experiments
	- Classical CNNs (GoogleNet, ResNet) for comparison

	

- **Regularization & Optimization**: applied dropout, early stopping, and experimented with L1/L2 regularization.

- **Data Augmentation**: tested rotation, shift, and zoom transformations to enhance model robustness.

- **Clustering & Representation Learning**: extracted latent features from the best CNN, applied **KMeans** and **DBSCAN**, and visualized feature spaces with PCA and t-SNE to reveal meaningful subcategories.

- **Model Evaluation**: compared models using accuracy, loss curves, runtime performance, and cluster interpretability.

- **Tools & Libraries**: Python, Jupyter Notebook, NumPy, Pandas, TensorFlow/Keras, Matplotlib, Seaborn, scikit-learn.







## Project 10: Variational Autoencoder (VAE) for Anime Face Generation
code in: [VAE.ipynb](./VAE.ipynb)

This project implements a **Variational Autoencoder (VAE)** using **PyTorch**.  The VAE is trained on image data to learn compressed latent representations and generate new samples, demonstrating the power of **deep generative models**.

Description

- Implemented the encoder and decoder networks with reparameterization trick.  
- Used latent variable sampling to generate realistic outputs.  
- Trained the model on dataset (e.g., MNIST / Fashion-MNIST) for unsupervised representation learning.  
- Visualized reconstructed images and samples drawn from the latent space.  

Skill Sets

- **Deep Generative Models**: Variational Autoencoder (VAE) with reparameterization trick.  
- **Unsupervised Learning**: Latent feature extraction and reconstruction.  
- **Neural Network Implementation**: Encoder, decoder, and loss function (reconstruction + KL divergence).  
- **PyTorch**: Custom training loops, optimizers, loss monitoring.  
- **Visualization**: Original vs reconstructed images, latent space sampling.  

Example Results

- Achieved high-quality reconstructions on MNIST/Fashion-MNIST.  
- Latent space visualization shows meaningful clustering of digit/image classes.  
- Generated new synthetic samples from the learned distribution.



## Project 11: Multi-Armed Bandit with Q-Learning
code in: [bandit_Q.ipynb](./bandit_Q.ipynb)

**Description**: This project explores **Reinforcement Learning (RL)** through the **multi-armed bandit problem**,  implemented with **Q-learning** in Python.  The goal is to balance **exploration vs exploitation** while maximizing cumulative reward.



- Implemented the **multi-armed bandit environment** with different arms and reward distributions.  
- Applied **Q-learning** to estimate action values and improve decision making over time.  
- Compared different exploration strategies (ε-greedy, softmax, etc.).  
- Visualized reward curves and convergence of action-value estimates.  



Skill Sets

- **Reinforcement Learning Fundamentals**: Exploration-exploitation tradeoff, reward maximization.  
- **Q-Learning Implementation**: Update rules, learning rate tuning, convergence behavior.  
- **Experimentation**: Tested ε-greedy vs other exploration strategies.  
- **Visualization**: Plotted average reward per timestep and action-value convergence.  
- **Tools & Libraries**: Python, NumPy, Matplotlib, Jupyter Notebook.  



Example Results

- Q-learning converges to optimal arm selection with sufficient exploration.  
- ε-greedy policy balances exploration vs exploitation effectively.  
- Reward curves demonstrate performance differences between exploration strategies.



## Project 12: Movie Rating Prediction
code in: [MovieRatingPrediction](./MovieRatingPrediction)




## Project 13: Stock Prediction with RNN
code in: [StockPrediction](./StockPrediction)

This project applies **Recurrent Neural Networks (RNNs)** to forecast stock prices from historical time-series data.  The model captures sequential dependencies in daily price movements to predict future stock trends. Build and train an RNN-based model to predict future stock prices using historical data (open, high, low, close, volume).

**Approach:**

- **Data Preprocessing**:  
  - Collected stock price data (daily OHLCV).  
  - Applied normalization for numerical stability.  
  - Created sliding windows for time-series sequences.  
- **Modeling**:  
  - Implemented RNN (and optionally LSTM/GRU for comparison).  
  - Trained the network to minimize prediction error (MSE/MAE).  
- **Evaluation**:  
  - Compared predicted vs actual stock prices.  
  - Reported RMSE/MAE performance on test data.  
- **Visualization**:  
  - Training loss curves.  
  - Predicted vs actual price comparison.  



**Skill Sets:**

- **Time-Series Forecasting**: Sequence windowing, lag features.  
- **Deep Learning**: RNN/LSTM/GRU for sequential data.  
- **Data Engineering**: Normalization, train/test split for time-series.  
- **Model Evaluation**: RMSE, MAE, and trend alignment analysis.  
- **Tools & Libraries**: Python, TensorFlow/Keras (or PyTorch), Pandas, NumPy, Matplotlib, Seaborn.  



**Example Results**

- The RNN successfully learned short-term stock price patterns.  
- LSTM/GRU models showed improved stability and lower RMSE compared to vanilla RNN.  
- Predictions followed the general trend of stock prices, with stronger performance in short-term horizons.  



## Project 14: Graph Convolutional Network (GCN) on Dolphins Social Network 🐬
code in: [NewZealandDolphinsEmbedding-GCN](./NewZealandDolphinsEmbedding-GCN)

This project applies a **Graph Convolutional Network (GCN)** to the famous **Dolphins social network dataset**,  which captures associations between 62 dolphins in Doubtful Sound, New Zealand.

**Objective**

- Learn low-dimensional embeddings for dolphins (nodes) using GCN.  
- Visualize community structures and compare with traditional methods.  
- Demonstrate the power of GCNs on small social networks.

**Dataset**

- **Nodes**: 62 dolphins (individual animals).  
- **Edges**: Social interactions (association in groups).  
- **Reference**: Lusseau, D. (2003). *The emergent properties of a dolphin social network*. Proc. R. Soc. Lond. B.

**Approach**

- Built the graph with NetworkX / PyTorch Geometric.  
- Implemented a 2-layer GCN to aggregate neighbor information.  
- Trained embeddings for unsupervised visualization / community separation.  
- Compared GCN embeddings with Node2Vec and DeepWalk baselines.  

**Results**

- GCN embeddings separated dolphins into two clear communities.  
- Visualization showed distinct clusters corresponding to observed subgroups.  
- Demonstrated that GCN captures both structure and features (when available).  

**Skill Sets**

- Graph Neural Networks (GNNs), GCN architecture.  
- Graph representation learning (embeddings).  
- Unsupervised learning and visualization (PCA, t-SNE).  
- Tools: Python, PyTorch (or PyTorch Geometric / DGL), NetworkX, scikit-learn, Matplotlib.



# Kaggle Machine Learning Competition – ECE657: Stroke Risk Prediction
code and report in: [submit-ece657-kaggle.ipynb](./submit-ece657-kaggle.ipynb)

This project contains my submission for the **ECE657 Kaggle competition**. The goal was to build and optimize machine learning models for predictive accuracy on the provided dataset.

## Description
- Explored and cleaned the dataset (missing values, normalization, encoding).  
- Engineered new features to improve model performance.  
- Implemented and compared multiple models (e.g., Logistic Regression, Random Forest, Gradient Boosting, Neural Networks).  
- Tuned hyperparameters using cross-validation and grid/random search.  
- Submitted predictions to Kaggle and evaluated leaderboard performance.  

## Skill Sets
- **Data Preprocessing**: Handling missing data, categorical encoding, feature scaling.  
- **Feature Engineering**: Created derived features to improve model accuracy.  
- **Modeling**: Implemented ML algorithms with scikit-learn / XGBoost / PyTorch.  
- **Model Selection**: Cross-validation, hyperparameter tuning, and ensemble methods.  
- **Competition Workflow**: Prepared final predictions for Kaggle submission.  
- **Tools & Libraries**: Python, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn.  

## Results
- Achieved strong performance on the Kaggle leaderboard.  
- Demonstrated end-to-end ML pipeline: data → features → model → submission.
