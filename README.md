# nlp-classification-lstm-bert
McGill COMP 551 - Mini-Project 4: Comparative study of LSTM (implemented from scratch) vs. BERT fine-tuning for scientific abstract classification. Includes custom LSTM cells, GloVe embeddings, and t-SNE visualization of feature spaces.

* Gabriel Caballero (261108565)
* Adam Dufour (261193949)

## Prerequisites
* Python 3.8+ (we are using python 3.12)
* Packages in requirements.txt
```bash
pip install -r requirements.txt
```

## Data Setup
- WebOfScience Dataset: download and place the folder `WebOfScienceDataset` in the project root
- GloVe Embeddings: download `glove.6B.zip`, extract it, and place `glove.6B.300d.txt` inside a folder named `glove.6B/`.

## Running the Code
Notebook is structured in 3 main sections:
- Processing the data
- Implementation of LSTM and BERT models
- Running of Experiments

Launch the Jupyter Notebook / JupyterLab from your terminal (or use PyCharm/VSCode) and click run, all the tests will start running automatically
