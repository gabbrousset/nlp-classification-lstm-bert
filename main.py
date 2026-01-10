import os
import re
import time
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

seed = 551
np.random.seed(seed)
torch.manual_seed(seed)

def getDevice():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")
    return device

DEVICE = getDevice()

"""## Task 1: Acquire and Pre-process the Web of Science Dataset"""

DATA_DIR = "WebOfScienceDataset/WOS11967"
GLOVE_PATH = "glove.6B/glove.6B.300d.txt"

MAX_SEQ_LEN = 200 # dynamically set later
EMBED_DIM = 300     # we are using GloVe 300d
MAX_VOCAB = 10000
BATCH_SIZE = 32

def clean_text(text):
    """
    simple text cleaning:
    - lowercase
    - remove non-alphanumeric (keep spaces)
    - collapse multiple spaces
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data(data_dir):
    """
    loads X, Y, and YL1 from the directory
    """
    with open(os.path.join(data_dir, 'X.txt'), 'r', encoding='utf-8') as f:
        texts = [clean_text(line) for line in f.readlines()]

    with open(os.path.join(data_dir, 'Y.txt'), 'r', encoding='utf-8') as f:
        y_sub = [int(line.strip()) for line in f.readlines()]

    with open(os.path.join(data_dir, 'YL1.txt'), 'r', encoding='utf-8') as f:
        y_domain = [int(line.strip()) for line in f.readlines()]

    return texts, y_sub, y_domain

def get_dynamic_max_len(texts, percentile=95):
    """
    justification for MAX_SEQ_LEN:
    calculates the length covering 'percentile'% of all documents
    avoids wasting memory on padding outliers
    """
    lengths = [len(t.split()) for t in texts]
    limit = int(np.percentile(lengths, percentile))
    print(f"95th percentile length is {limit}. Mean is {int(np.mean(lengths))}.")
    return limit

def build_vocab(texts, max_words=MAX_VOCAB, min_freq=2):
    """
    builds a dictionary mapping words to integer indices
    """
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    # special tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}

    # add words that meet criteria
    for word, count in word_counts.most_common(max_words - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab

def encode_texts(texts, vocab, max_len=MAX_SEQ_LEN):
    """
    converts list of text strings to a Tensor of integer sequences
    """
    tensor_data = []
    unk_idx = vocab["<UNK>"]
    pad_idx = vocab["<PAD>"]

    for text in texts:
        tokens = text.split()
        # convert to indices with UNK fallback
        seq = [vocab.get(t, unk_idx) for t in tokens]

        if len(seq) < max_len:
            # pad with 0
            seq = seq + [pad_idx] * (max_len - len(seq))
        else:
            # truncate
            seq = seq[:max_len]

        tensor_data.append(seq)

    return torch.tensor(tensor_data, dtype=torch.long)

def load_glove_matrix(path, vocab, embed_dim=EMBED_DIM):
    """
    parse GloVe text file and returns a weight matrix for the specific vocab
    uses memory efficient loading: checks vocab while reading the file
    """
    weights = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim))

    if "<PAD>" in vocab:
        weights[vocab["<PAD>"]] = 0

    hits = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]

            if word in vocab:
                # only parse if we need this word
                vector = np.array(values[1:], dtype=float)

                if len(vector) == embed_dim:
                    weights[vocab[word]] = vector
                    hits += 1

    print(f"GloVe loaded. Found {hits} / {len(vocab)} words.")
    return torch.tensor(weights, dtype=torch.float32)

def prepare_data():
    texts, y_sub, y_domain = load_data(DATA_DIR)

    dynamic_max_len = get_dynamic_max_len(texts)

    vocab = build_vocab(texts)

    # encode X (Inputs)
    X_tensor = encode_texts(texts, vocab, max_len=dynamic_max_len)

    # encode Y (Targets)
    Y_sub_tensor = torch.tensor(y_sub, dtype=torch.long)
    Y_domain_tensor = torch.tensor(y_domain, dtype=torch.long)

    embedding_weights = load_glove_matrix(GLOVE_PATH, vocab)

    # create TensorDataset
    # since we have everything in RAM, TensorDataset is fastest
    dataset = TensorDataset(X_tensor, Y_domain_tensor, Y_sub_tensor)

    # Train/Val/Test Split (64/16/20)
    test_size = int(0.2 * len(dataset))
    remaining_size = len(dataset) - test_size
    val_size = int(0.2 * remaining_size)
    train_size = remaining_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_ldr = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_ldr = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_ldr = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_ldr, val_ldr, test_ldr, embedding_weights, vocab, train_dataset, val_dataset, test_dataset

"""## Task 2: Implement LSTM and BERT models"""

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, init_type):
        super(CustomLSTMCell, self).__init__()

        self.hidden_size = hidden_size

        # we need weights for input (x) and hidden state (h) for all 4 gates:
        # i (input), f (forget), g (cell), o (output)
        # we can combine them into one big Linear layer for efficiency: 4 * hidden_size

        # input: input_size -> Output: 4 * hidden_size
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        # input: hidden_size -> Output: 4 * hidden_size
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)

        # initialize weights (from what I researched Xavier is standard for Tanh activations)
        if init_type == 'xavier':
            nn.init.xavier_uniform_(self.weight_ih.weight)
            nn.init.xavier_uniform_(self.weight_hh.weight)
        elif init_type == 'zero':
            nn.init.zeros_(self.weight_ih.weight)
            nn.init.zeros_(self.weight_hh.weight)
        elif init_type == 'random':
            nn.init.normal_(self.weight_ih.weight, mean=0, std=0.01)
            nn.init.normal_(self.weight_hh.weight, mean=0, std=0.01)

    def forward(self, x, state):
        """
        x: (batch_size, input_size)
        state: tuple (h_prev, c_prev)
        """
        h_prev, c_prev = state

        # computing all gates at once
        gates = self.weight_ih(x) + self.weight_hh(h_prev)

        # splitting into 4 chunks (input, forget, cell, output)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        # applying activations
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)

        # updating memory cell (c_t)
        # "forget what is old, add what is new"
        c_next = (f * c_prev) + (i * g)

        # updating hidden state (h_t)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class CustomLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, hidden_size=512, embedding_matrix=None, dropout=0.5, init_type='xavier'):
        super(CustomLSTMModel, self).__init__()
        self.device = getDevice()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix)
            # freezing embeddings for a couple reasons: its faster, and might want to compare different pre-trained embeddings
            # self.embedding.weight.requires_grad = False

        self.lstm_cell = CustomLSTMCell(embed_dim, hidden_size, init_type)

        # classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

        self.to(self.device)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # embed: [batch_size, seq_len, embed_dim]
        x_emb = self.embedding(x)

        # init states (zeros)
        h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(self.device)

        # loop through sequence
        for t in range(seq_len):
            x_t = x_emb[:, t, :] # [batch_size, embed_dim]
            h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))

        # classify using last hidden state
        out = self.dropout(h_t)
        logits = self.fc(out)
        return logits

    def fit(self, train_ldr, val_ldr, epochs=10, lr=0.001, l1_lambda=0.0, weight_decay=0.0):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        print(f"\ntraining LSTM (output dim: {self.output_dim}) for {epochs} epochs")

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            correct = 0
            total = 0

            for X_batch, Y_domain, Y_sub in train_ldr:
                X_batch = X_batch.to(self.device)
                # target based on model output dimension
                target = Y_domain.to(self.device) if self.output_dim == 7 else Y_sub.to(self.device)

                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, target)

                # l1 regularization
                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in self.parameters())
                    loss += l1_lambda * l1_norm

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            val_acc = self.evaluate_acc(val_ldr)
            val_loss = self.evaluate_loss(val_ldr, criterion)

            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)
            print(f"epoch {epoch+1}/{epochs} | loss: {total_loss/len(train_ldr):.4f} | val acc: {val_acc:.2f}%")

        return history

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self(X)
            predictions = torch.argmax(outputs, dim=1)
        return predictions

    def evaluate_acc(self, data_loader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, Y_domain, Y_sub in data_loader:
                X_batch = X_batch.to(self.device)
                target = Y_domain.to(self.device) if self.output_dim == 7 else Y_sub.to(self.device)

                preds = self.predict(X_batch)

                correct += (preds == target).sum().item()
                total += target.size(0)

        return 100 * correct / total

    def evaluate_loss(self, data_loader, criterion):
        self.eval()
        total_loss = 0
        total = 0

        with torch.no_grad():
             for X_batch, Y_domain, Y_sub in data_loader:
                X_batch = X_batch.to(self.device)
                target = Y_domain.to(self.device) if self.output_dim == 7 else Y_sub.to(self.device)

                outputs = self(X_batch)

                loss = criterion(outputs, target)
                total_loss += loss.item() * target.size(0)
                total += target.size(0)

        return total_loss / total

class BERTClassifier(nn.Module):
    """
    BERT-based text classifier using specific pre-trained weights
    includes functionality for fine-tuning and attention visualization
    """
    def __init__(self, output_dim, model_name = 'bert-base-uncased', dropout = 0.3):
        super().__init__()
        self.device = getDevice()

        # load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name, output_attentions=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        # BERT outputs (last_hidden_state, pooler_output, hidden_states, attentions)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # use the pooler_output (embedding of [CLS] token) for classification
        pooler_output = outputs.pooler_output

        pooled_output = self.dropout(pooler_output)
        logits = self.fc(pooled_output)

        return logits

    def get_attention_maps(self, input_ids, attention_mask):
        """
        extracts attention maps for visualization
        """
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids.to(self.device),
                                attention_mask=attention_mask.to(self.device))
        return outputs.attentions

    def fit(self, train_ldr, val_ldr,
            lr = 2e-5,
            epochs = 3,
            patience = 2,
            weight_decay = 0.0,
            l1_lambda = 0.0):
        """
        fine-tunes BERT
        - requires input_ids and attention_mask
        - data loaders must yield (input_ids, attention_mask, labels)
        """
        # AdamW handles weight_decay
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_ldr) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"starting BERT fine-tuning")

        for epoch in range(epochs):
            self.train()
            total_train_loss = 0
            correct_train = 0
            total_samples = 0
            start_time = time.time()

            for batch in train_ldr:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.zero_grad()
                logits = self.forward(b_input_ids, b_input_mask)
                loss = criterion(logits, b_labels)

                if l1_lambda > 0:
                     l1_norm = sum(p.abs().sum() for p in self.parameters())
                     loss += l1_lambda * l1_norm

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # clips gradients to prevent explosion
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item() * b_input_ids.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_train += (preds == b_labels).sum().item()
                total_samples += b_input_ids.size(0)

            avg_train_loss = total_train_loss / total_samples
            train_acc = correct_train / total_samples

            # validation
            val_loss, val_acc = self.evaluate(val_ldr, criterion)

            epoch_time = time.time() - start_time
            print(f"epoch {epoch+1}/{epochs} [{epoch_time:.1f}s] | "
                  f"train loss: {avg_train_loss:.4f} acc: {train_acc:.4f} | "
                  f"val loss: {val_loss:.4f} acc: {val_acc:.4f}")

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), "best_bert_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        self.load_state_dict(torch.load("best_bert_model.pth"))
        return history

    def evaluate(self, val_ldr, criterion):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_ldr:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                logits = self.forward(b_input_ids, b_input_mask)
                loss = criterion(logits, b_labels)

                total_loss += loss.item() * b_input_ids.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == b_labels).sum().item()
                total += b_input_ids.size(0)

        return total_loss / total, correct / total

"""## Task 3: Run experiments"""

def evaluate_lstm(train_ds, val_ds, vocab_size, output_dim, batch_size=BATCH_SIZE, embedding_matrix=None, epochs=10, hidden_size=512, dropout=0.5, l1_lambda=0.0, weight_decay=0.0, lr=0.001, init_type='xavier'):

    # DataLoader inside to handle batch_size variation
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # LSTM Classifier
    model = CustomLSTMModel(
        vocab_size,
        embedding_matrix.shape[1],
        output_dim,
        hidden_size=hidden_size,
        dropout=dropout,
        embedding_matrix=embedding_matrix,
        init_type=init_type
    )

    history = model.fit(
        train_ldr, val_ldr,
        lr=lr,
        epochs=epochs,
        l1_lambda=l1_lambda,
        weight_decay=weight_decay
    )

    # best validation accuracy from history
    return max(history.get("val_acc", [0]))

def evaluate_bert(dropout, l1_lambda, weight_decay, batch_size, lr,
                  train_ds, val_ds, output_dim):

    # DataLoader inside to handle batch_size variation
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # BERT Classifier (Note: re-init model each time is computationally heavy)
    model = BERTClassifier(output_dim, dropout=dropout).to(DEVICE)

    history = model.fit(
        train_ldr, val_ldr,
        lr=lr,
        epochs=2, # low epochs for search speed
        patience=1,
        weight_decay=weight_decay,
        l1_lambda=l1_lambda
    )

    # best validation accuracy from history
    return max(history.get("val_acc", [0]))

def get_bert_test_data(all_texts, all_labels, test_indices, max_length=256):
    """
    prepares a TensorDataset for the BERT test set using the shared indices
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_texts = [all_texts[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]

    # Encode test data
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    test_dataset = TensorDataset(
        test_encodings['input_ids'],
        test_encodings['attention_mask'],
        torch.tensor(test_labels)
    )
    test_ldr = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return test_ldr, test_texts, test_labels, tokenizer

def get_sample_texts(model, data_loader, raw_texts):
    """
    identifies one correctly predicted and one incorrectly predicted document index
    """
    model.eval()
    correct_idx = None
    incorrect_idx = None

    with torch.no_grad():
        for batch in data_loader:
            b_input_ids, b_input_mask, b_labels = [b.to(model.device) for b in batch]

            # predict
            logits = model(b_input_ids, b_input_mask)
            preds = torch.argmax(logits, dim=1)

            # find indices in the current batch where prediction matches/mismatches label
            correct_mask = (preds == b_labels).cpu().numpy()
            incorrect_mask = (preds != b_labels).cpu().numpy()

            if correct_idx is None and correct_mask.any():
                # get index within the batch, map to dataset index
                local_idx = np.where(correct_mask)[0][0]
                global_idx = data_loader.dataset.indices[data_loader.batch_size * data_loader.dataloader.batch_sampler.last_index + local_idx]
                correct_idx = global_idx

            if incorrect_idx is None and incorrect_mask.any():
                local_idx = np.where(incorrect_mask)[0][0]
                global_idx = data_loader.dataset.indices[data_loader.batch_size * data_loader.dataloader.batch_sampler.last_index + local_idx]
                incorrect_idx = global_idx

            if correct_idx is not None and incorrect_idx is not None:
                break

    if correct_idx is None or incorrect_idx is None:
        # fallback if the full data isn't used or model is too accurate/inaccurate
        return raw_texts[0], raw_texts[-1], 0, 0

    return raw_texts[correct_idx], raw_texts[incorrect_idx], correct_idx, incorrect_idx

def plot_combined_results(values, model_1_scores, model_2_scores, title, xlabel, label_model_1="LSTM Val Acc", label_model_2="BERT Val Acc", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots()

    ax.plot(values, model_1_scores, label=label_model_1, marker='o')
    ax.plot(values, model_2_scores, label=label_model_2, marker='x')

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    clean_title = title.replace(" ", "_").replace(":", "")
    filename = f"{clean_title}.png"
    filepath = os.path.join(save_dir, filename)

    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"saved plot to: {filepath}")

def plot_single_result(values, scores, title, xlabel, ylabel="Validation Accuracy", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()

    ax.plot(values, scores, marker='o', label='LSTM')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    clean_title = title.replace(" ", "_").replace(":", "")
    filename = f"{clean_title}.png"
    filepath = os.path.join(save_dir, filename)

    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"saved plot to: {filepath}")

print("loading data")
train_loader, val_loader, test_loader, embedding_matrix, vocab, train_ds, val_ds, test_ds = prepare_data()

# reload raw texts for BERT (needed since LSTM used indices)
all_texts, y_sub_raw, y_domain_raw = load_data(DATA_DIR)

vocab_size = len(vocab)
embedding_dim = embedding_matrix.shape[1]
output_dim_domain = 7
output_dim_sub = 33

# indices to accurately map labels to raw text (for BERT's tokenizer)
train_indices = train_ds.indices
val_indices = val_ds.indices
test_indices = test_ds.indices

bert_train_labels = [y_domain_raw[i] for i in train_indices]
bert_val_labels = [y_domain_raw[i] for i in val_indices]

bert_sub_train_labels = [y_sub_raw[i] for i in train_indices]
bert_sub_val_labels = [y_sub_raw[i] for i in val_indices]

# tokenise the raw text for BERT separately
bert_train_texts = [all_texts[i] for i in train_indices]
bert_val_texts = [all_texts[i] for i in val_indices]
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 1) test dropout for LSTM (both classification tasks)
dropout_list = [0.1, 0.3, 0.5]
lstm_res_domain = []
lstm_res_sub = []

for d in dropout_list:
    print(f"testing dropout: {d}")

    l_acc_domain = evaluate_lstm(train_ds, val_ds, vocab_size, output_dim_domain, embedding_matrix=embedding_matrix, dropout=d)
    lstm_res_domain.append(l_acc_domain)

    l_acc_sub = evaluate_lstm(train_ds, val_ds, vocab_size, output_dim_sub, embedding_matrix=embedding_matrix, dropout=d, epochs=20)
    lstm_res_sub.append(l_acc_sub)

plot_combined_results(dropout_list, lstm_res_domain, lstm_res_sub, "Effect of Dropout", "Dropout Rate", label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub")

best_lstm_dropout_index_domain = np.argmax(lstm_res_domain)
best_lstm_dropout_index_sub = np.argmax(lstm_res_sub)

BEST_DROPOUT_LSTM_DOMAIN = dropout_list[best_lstm_dropout_index_domain]
BEST_DROPOUT_LSTM_SUB = dropout_list[best_lstm_dropout_index_sub]

# 2) test learning rate for LSTM (both classification tasks)
lr_list = [1e-2, 1e-3, 1e-4]
lstm_res_domain_lr = []
lstm_res_sub_lr = []

for lr in lr_list:
    print(f"testing learning rate: {lr}")

    l_acc_domain = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_domain,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_DOMAIN,
        lr=lr
    )
    lstm_res_domain_lr.append(l_acc_domain)

    l_acc_sub = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_sub,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_SUB,
        lr=lr,
        epochs=20
    )
    lstm_res_sub_lr.append(l_acc_sub)

plot_combined_results(lr_list, lstm_res_domain_lr, lstm_res_sub_lr, "Effect of Learning Rate", "Learning Rate", label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub")

best_lstm_lr_index_domain = np.argmax(lstm_res_domain_lr)
best_lstm_lr_index_sub = np.argmax(lstm_res_sub_lr)

BEST_LR_LSTM_DOMAIN = lr_list[best_lstm_lr_index_domain]
BEST_LR_LSTM_SUB = lr_list[best_lstm_lr_index_sub]

# 3) test hidden size for LSTM (both classification tasks)
hidden_size_list = [128, 256, 512]
lstm_res_domain_hidden = []
lstm_res_sub_hidden = []

for hs in hidden_size_list:
    print(f"testing hidden size: {hs}")

    l_acc_domain = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_domain,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_DOMAIN,
        lr=BEST_LR_LSTM_DOMAIN,
        hidden_size=hs
    )
    lstm_res_domain_hidden.append(l_acc_domain)

    l_acc_sub = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_sub,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_SUB,
        lr=BEST_LR_LSTM_SUB,
        hidden_size=hs,
        epochs=20
    )
    lstm_res_sub_hidden.append(l_acc_sub)

plot_combined_results(hidden_size_list, lstm_res_domain_hidden, lstm_res_sub_hidden, "Effect of Hidden Size", "Hidden Size", label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub")

best_lstm_hs_index_domain = np.argmax(lstm_res_domain_hidden)
best_lstm_hs_index_sub = np.argmax(lstm_res_sub_hidden)

BEST_HS_LSTM_DOMAIN = hidden_size_list[best_lstm_hs_index_domain]
BEST_HS_LSTM_SUB = hidden_size_list[best_lstm_hs_index_sub]

# 4) test init type for LSTM
init_types = ['xavier', 'random', 'zero']
init_results = []

for hs in init_types:
    print(f"testing initialization type: {hs}")

    l_acc_domain = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_domain,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_DOMAIN,
        lr=BEST_LR_LSTM_DOMAIN,
        hidden_size=BEST_HS_LSTM_DOMAIN,
    )
    init_results.append(l_acc_domain)

plot_single_result(init_types, init_results, "Effect of Initialization", "Init Type")

# 5) Test L1 Regularization for LSTM (Domain Task Only)
# Testing: None (0.0), Small (1e-5), Moderate (1e-3)
l1_values = [0.0, 1e-5, 1e-3]
l1_results = []

for l1_val in l1_values:
    print(f"testing L1 lambda: {l1_val}")

    l_acc_domain = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_domain,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_DOMAIN,
        lr=BEST_LR_LSTM_DOMAIN,
        hidden_size=BEST_HS_LSTM_DOMAIN,
        l1_lambda=l1_val
    )
    l1_results.append(l_acc_domain)

l1_labels = [str(v) for v in l1_values]
plot_single_result(l1_labels, l1_results, "Effect of L1 Regularization", "L1 Lambda")

# 6) Embedding Dimension Comparison (50d vs 100d vs 200d vs 300d)
print("GloVe dimension comparison")
GLOVE_DIR = "glove.6B"
glove_dims = [50, 100, 200, 300]
dim_results = []

for dim in glove_dims:
    filename = f"glove.6B.{dim}d.txt"
    path = os.path.join(GLOVE_DIR, filename)

    print(f"\ntesting GloVe dimension: {dim}d")

    # load specific matrix for this dimension
    # pass 'dim' so the parser knows the vector size
    current_matrix = load_glove_matrix(path, vocab, embed_dim=dim)

    acc = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_domain,
        embedding_matrix=current_matrix,
        dropout=BEST_DROPOUT_LSTM_DOMAIN,
        lr=BEST_LR_LSTM_DOMAIN,
        hidden_size=BEST_HS_LSTM_DOMAIN,
        epochs=10
    )
    dim_results.append(acc)

dim_labels = [str(d) for d in glove_dims]
plot_single_result(dim_labels, dim_results, "Effect of Embedding Dimension", "GloVe Dimension")

# TASK 3: REQUIRED CLASSIFICATION EXPERIMENTS
# 1) custom LSTM (comain classification - 7 classes)
print("\n--- EXP 1/4: Custom LSTM (Domain Classification) ---")
lstm_domain = CustomLSTMModel(
    vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_size=BEST_HS_LSTM_DOMAIN, output_dim=output_dim_domain,
    embedding_matrix=embedding_matrix, dropout=BEST_DROPOUT_LSTM_DOMAIN
)
lstm_domain.fit(train_loader, val_loader, epochs=10)
test_acc_domain_lstm = lstm_domain.evaluate_acc(test_loader)
print(f"Final LSTM Domain Test Accuracy: {test_acc_domain_lstm:.2f}%")

# 2) BERT Fine-Tuning (Domain Classification - 7 classes)
print("\n--- EXP 2/4: BERT Fine-Tuning (Domain Classification) ---")
bert_domain_model = BERTClassifier(output_dim=output_dim_domain)

# create BERT loaders using full raw text data
bert_domain_train_ldr = DataLoader(TensorDataset(
    bert_tokenizer(bert_train_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')['input_ids'],
    bert_tokenizer(bert_train_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')['attention_mask'],
    torch.tensor(bert_train_labels)
), batch_size=BATCH_SIZE, shuffle=True)

bert_domain_val_ldr = DataLoader(TensorDataset(
    bert_tokenizer(bert_val_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')['input_ids'],
    bert_tokenizer(bert_val_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')['attention_mask'],
    torch.tensor(bert_val_labels)
), batch_size=BATCH_SIZE)

bert_domain_model.fit(bert_domain_train_ldr, bert_domain_val_ldr, epochs=3)

# setup BERT test loader
bert_test_loader, _, _, _ = get_bert_test_data(all_texts, y_domain_raw, test_indices)
bert_test_loss, bert_test_acc = bert_domain_model.evaluate(bert_test_loader, nn.CrossEntropyLoss())
test_acc_domain_bert = bert_test_acc * 100
print(f"Final BERT Domain Test Accuracy: {test_acc_domain_bert:.2f}%")

# 3) custom LSTM (sub-field classification - 33 classes)
print("\n--- EXP 3/4: Custom LSTM (Sub-field Classification) ---")
lstm_sub = CustomLSTMModel(
    vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_size=BEST_HS_LSTM_SUB, output_dim=output_dim_sub,
    embedding_matrix=embedding_matrix, dropout=BEST_DROPOUT_LSTM_SUB
)
lstm_sub.fit(train_loader, val_loader, epochs=25)
test_acc_sub_lstm = lstm_sub.evaluate_acc(test_loader)
print(f"Final LSTM Sub-field Test Accuracy: {test_acc_sub_lstm:.2f}%")

# 4) BERT Fine-Tuning (sub-field classification - 33 classes)
print("\n--- EXP 4/4: BERT Fine-Tuning (Sub-field Classification) ---")
bert_sub_model = BERTClassifier(output_dim=output_dim_sub)

# create BERT sub-field loaders
bert_sub_train_ldr = DataLoader(TensorDataset(
    bert_tokenizer(bert_train_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')['input_ids'],
    bert_tokenizer(bert_train_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')['attention_mask'],
    torch.tensor(bert_sub_train_labels)
), batch_size=BATCH_SIZE, shuffle=True)

bert_sub_val_ldr = DataLoader(TensorDataset(
    bert_tokenizer(bert_val_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')['input_ids'],
    bert_tokenizer(bert_val_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')['attention_mask'],
    torch.tensor(bert_sub_val_labels)
), batch_size=BATCH_SIZE)

bert_sub_model.fit(bert_sub_train_ldr, bert_sub_val_ldr, epochs=3)

# setup BERT test loader (Sub-field)
bert_sub_test_loader, _, _, _ = get_bert_test_data(all_texts, y_sub_raw, test_indices)
bert_sub_test_loss, bert_sub_test_acc = bert_sub_model.evaluate(bert_sub_test_loader, nn.CrossEntropyLoss())
test_acc_sub_bert = bert_sub_test_acc * 100
print(f"Final BERT Sub-field Test Accuracy: {test_acc_sub_bert:.2f}%")

def get_sample_data(model, data_loader):
    """
    Returns the input_ids, attention_mask, and tokenizer for one correct
    and one incorrect prediction directly from the tensors.
    """
    model.eval()
    correct_data = None
    incorrect_data = None

    # We need a tokenizer to decode the IDs back to text later
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with torch.no_grad():
        for batch in data_loader:
            b_input_ids, b_input_mask, b_labels = [b.to(model.device) for b in batch]

            logits = model(b_input_ids, b_input_mask)
            preds = torch.argmax(logits, dim=1)

            correct_mask = preds == b_labels
            incorrect_mask = preds != b_labels

            # Capture one correct example
            if correct_data is None and correct_mask.any():
                idx = torch.where(correct_mask)[0][0]
                # Store tuple: (input_ids, attention_mask)
                # unsqueeze(0) keeps the batch dimension [1, seq_len]
                correct_data = (b_input_ids[idx].unsqueeze(0), b_input_mask[idx].unsqueeze(0))

            # Capture one incorrect example
            if incorrect_data is None and incorrect_mask.any():
                idx = torch.where(incorrect_mask)[0][0]
                incorrect_data = (b_input_ids[idx].unsqueeze(0), b_input_mask[idx].unsqueeze(0))

            if correct_data is not None and incorrect_data is not None:
                break

    return correct_data, incorrect_data, tokenizer

def visualize_attention(model, tokenizer, input_ids, attention_mask, title_prefix, device, layer_idx=11, head_idx=0, save_dir="plots"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    attentions = model.get_attention_maps(input_ids, attention_mask)
    attention_matrix = attentions[layer_idx][0, head_idx, :, :].cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Clean tokens
    seq_len = attention_mask.sum().item()
    attention_matrix = attention_matrix[:seq_len, :seq_len]
    tokens = tokens[:seq_len]

    # --- ROBUST PLOTTING ---
    # Create explicit Figure and Axes objects
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis', ax=ax)

    title = f"{title_prefix} - Layer {layer_idx+1}, Head {head_idx+1}"
    ax.set_title(title)
    ax.set_xlabel("Key (Attended To)")
    ax.set_ylabel("Query (Attending From)")
    plt.xticks(rotation=90)

    clean_title = title.replace(" ", "_").replace(":", "").replace(",", "")
    filename = f"{clean_title}.png"
    filepath = os.path.join(save_dir, filename)

    # Save the specific figure object, not the global state
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig) # Close memory
    print(f"saved attention map to: {filepath}")

def visualize_token_importance(model, tokenizer, input_ids, attention_mask, title_prefix, device, layer_idx=11, head_idx=0, save_dir="plots"):
    """
    Plots a bar chart of the top 15 tokens that the [CLS] token attends to.
    This solves the "purple grid" issue by auto-scaling the importance scores.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Get attention maps
    attentions = model.get_attention_maps(input_ids, attention_mask)

    # Select specific layer and head
    # Shape: [batch, num_heads, seq_len, seq_len]
    # We want the attention FROM [CLS] (index 0) TO every other token
    cls_attention = attentions[layer_idx][0, head_idx, 0, :].cpu().detach().numpy()

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Filter out [CLS], [SEP], [PAD] to see the actual content words
    valid_tokens = []
    valid_scores = []

    for token, score in zip(tokens, cls_attention):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            valid_tokens.append(token)
            valid_scores.append(score)

    # Sort by score descending
    sorted_indices = np.argsort(valid_scores)[::-1]
    top_n = 15 # Only show top 15 for readability

    top_tokens = [valid_tokens[i] for i in sorted_indices[:top_n]]
    top_scores = [valid_scores[i] for i in sorted_indices[:top_n]]

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal bar chart
    y_pos = np.arange(len(top_tokens))
    ax.barh(y_pos, top_scores, align='center', color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_tokens)
    ax.invert_yaxis()  # labels read top-to-bottom

    title = f"{title_prefix} Top Tokens - Layer {layer_idx+1} Head {head_idx+1}"
    ax.set_xlabel('Attention Score')
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Save
    clean_title = title.replace(" ", "_").replace(":", "").replace("-", "_")
    filename = f"{clean_title}.png"
    filepath = os.path.join(save_dir, filename)

    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved token importance plot to: {filepath}")

# TASK 3, REQUIREMENT 2: ATTENTION ANALYSIS
# BERT domain model for attention analysis
bert_test_loader_domain, _, _, _ = get_bert_test_data(all_texts, y_domain_raw, test_indices)

correct_data, incorrect_data, tokenizer = get_sample_data(bert_domain_model, bert_test_loader_domain)

visualize_attention(bert_domain_model, tokenizer, correct_data[0], correct_data[1], "Correct", DEVICE)
visualize_attention(bert_domain_model, tokenizer, incorrect_data[0], incorrect_data[1], "Incorrect", DEVICE)

visualize_token_importance(
    bert_domain_model, tokenizer, correct_data[0], correct_data[1],
    "Correct", DEVICE, layer_idx=11, head_idx=0
)
visualize_token_importance(
    bert_domain_model, tokenizer, incorrect_data[0], incorrect_data[1],
    "Incorrect", DEVICE, layer_idx=11, head_idx=0
)

print("\n--- RESULTS SUMMARY TABLE ---")
print("| Model | Task | Test Accuracy | Winner? |")
print("|---|---|---|---|")
print(f"| Custom LSTM (GloVe) | Domain (7 Classes) | {test_acc_domain_lstm:.2f}% | {'<--' if test_acc_domain_lstm > test_acc_domain_bert else ''} |")
print(f"| BERT Classifier | Domain (7 Classes) | {test_acc_domain_bert:.2f}% | {'<--' if test_acc_domain_bert > test_acc_domain_lstm else ''} |")
print(f"| Custom LSTM (GloVe) | Sub-field (33 Classes) | {test_acc_sub_lstm:.2f}% | {'<--' if test_acc_sub_lstm > test_acc_sub_bert else ''} |")
print(f"| BERT Classifier | Sub-field (33 Classes) | {test_acc_sub_bert:.2f}% | {'<--' if test_acc_sub_bert > test_acc_sub_lstm else ''} |")

def plot_model_comparison(test_acc_domain_lstm, test_acc_domain_bert,
                          test_acc_sub_lstm, test_acc_sub_bert,
                          save_path="plots/model_comparison.png"):

    # Data preparation
    tasks = ['Domain (7 Classes)', 'Sub-field (33 Classes)']
    lstm_scores = [test_acc_domain_lstm, test_acc_sub_lstm]
    bert_scores = [test_acc_domain_bert, test_acc_sub_bert]

    x = np.arange(len(tasks))  # label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting the bars
    rects1 = ax.bar(x - width/2, lstm_scores, width, label='Custom LSTM (GloVe)', color='skyblue')
    rects2 = ax.bar(x + width/2, bert_scores, width, label='BERT Classifier', color='salmon')

    # Formatting
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 100)  # Set y-axis to 0-100% for context
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Helper function to put text labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to {save_path}")

plot_model_comparison(test_acc_domain_lstm, test_acc_domain_bert, test_acc_sub_lstm, test_acc_sub_bert)
