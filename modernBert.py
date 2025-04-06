import torch
import ast
import pandas as pd
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from typing import List, Dict
from tqdm import tqdm
from torch.optim import AdamW
from torch import nn
import logging
logger = logging.getLogger("modern Bert")

def load_data():
    train_df = pd.read_csv("../data/train_data.csv", header=0)
    eval_df = pd.read_csv("../data/eval_data.csv", header=0)
    test_df = pd.read_csv("../data/test_data.csv", header=0)

    train_graphs = [ast.literal_eval(triples) for triples in train_df["body"]]
    eval_graphs = [ast.literal_eval(triples)for triples in eval_df["body"]]
    test_graphs = [ast.literal_eval(triples)for triples in test_df["body"]]

    graph_dict= {"train" : train_graphs, "eval":eval_graphs, "test":test_graphs}

    label_2_index = {"Inconsistent" : 1, "Consistent" : 0}
    label_dict= {"train" : train_df["consistency"].map(label_2_index), "eval" : eval_df["consistency"].map(label_2_index), "test" : test_df["consistency"].map(label_2_index)}
    return graph_dict, label_dict, label_2_index

def graph_to_t5_input(triples) -> str:
    return " ".join([f"{s} {p} {o}." for s, p, o in triples])

def graphs_to_t5_format(triples , labels: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "input_ids": graph_to_t5_input(graph),
            "label": label
        }
        for graph, label in tqdm(zip(triples, labels), total=len(triples))
    ]

def tokenize_data(tokenizer, data, max_input_length=4096):
    input_texts = [d["input_ids"] for d in data]
    target_labels = [d["label"] for d in data]

    # Initialize lists to store the tokenized sequences
    model_inputs = {
        "input_ids": [],
        "attention_mask": []
    }
    for text in tqdm(input_texts, total=len(input_texts)):
        encoding = tokenizer(
            text,
            padding="max_length",  # Ensure all sequences are padded to max_input_length
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt"
        )
        model_inputs["input_ids"].append(encoding.input_ids)
        model_inputs["attention_mask"].append(encoding.attention_mask)

    labels = torch.tensor(target_labels, dtype=torch.long)
    model_inputs["labels"] = labels
    return model_inputs



def run_train_epoch(model, data, criterion, optimizer, batch_size, gradient_accumulation_steps, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    data_points = int(len(data["input_ids"]))
    step = 0
    for i in tqdm(range(0, data_points)):
        step += 1
        input_ids = data["inputs_ids"][i]
        attention_mask = data["attention_mask"][i]
        labels = data["label"][i]

        output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        logits = output.logits
        loss = output.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = logits.argmax().item()
        accuracy = (preds == labels).float().mean().item()

        total_loss += loss.item()
        total_accuracy += accuracy

    # Averages
    avg_loss = total_loss / step
    avg_accuracy = total_accuracy / step

    return avg_loss, avg_accuracy




def main(batch_size, device):

    graphs, labels, _ = load_data()

    data =  {
        "train" : graphs_to_t5_format(graphs["train"],  labels["train"]),
        "eval" : graphs_to_t5_format(graphs["eval"], labels["eval"]),
        "test" : graphs_to_t5_format(graphs["test"], labels["test"])
        }
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    train_data = tokenize_data(tokenizer, data["train"])
    eval_data = tokenize_data(tokenizer, data["eval"])
    test_data = tokenize_data(tokenizer, data["test"])

    model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=2)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_epoch = 0 
    best_eval_accuracy = 0
    best_eval_loss = float('inf')
    best_eval_recall = 0
    best_eval_precision = 0


    print("TRAINING")
    for epoch in range(50):
        train_loss, train_accuracy = run_train_epoch(model=model, data=train_data, optimizer=optimizer, batch_size=batch_size, gradient_accumulation_steps=1, device=device)
        logging.info(f'train - {epoch = } # {train_loss = :.2f} # {train_accuracy = :.2f}')



if __name__ == "__main__": 
    batch_size = 1
    device = "cpu" 
    main(batch_size, device)

