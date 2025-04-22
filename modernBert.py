import torch
import ast
import pandas as pd
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from typing import List, Dict
from tqdm import tqdm
from torch.optim import AdamW
from torch import nn
import logging
from sklearn.metrics import precision_score, recall_score
import wandb
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
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
        model_inputs["input_ids"].append(encoding.input_ids.squeeze())
        model_inputs["attention_mask"].append(encoding.attention_mask.squeeze())

    labels = torch.tensor(target_labels, dtype=torch.long)
    model_inputs["labels"] = labels
    return model_inputs



def run_train_epoch(model, data, batch_size, optimizer,  gradient_accumulation_steps, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    data_points = int(len(data["input_ids"]))
    step = 0

    input_ids = torch.stack(data["input_ids"]) if isinstance(data["input_ids"][0], torch.Tensor) else torch.tensor(data["input_ids"])
    attention_mask = torch.stack(data["attention_mask"]) if isinstance(data["attention_mask"][0], torch.Tensor) else torch.tensor(data["attention_mask"])
    labels = torch.tensor(data["labels"])

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in tqdm(dataloader):
#    for i in tqdm(range(50)):
        step += 1
#        input_ids = data["input_ids"][i].unsqueeze(0).to(device)
#        attention_mask = data["attention_mask"][i].unsqueeze(0).to(device)
#        labels = data["labels"][i].unsqueeze(0).to(device)

        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        assert input_ids.dim() ==2 , f"Input Ids mask shape is {input_ids.shape}"
        assert attention_mask.dim() == 2, f"Attention mask shape is {attention_mask.shape}"
        assert labels.dim() ==1, f"Labels mask shape is {labels.shape}"

        output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        logits = output.logits
        loss = output.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()

        total_loss += loss.item()
        total_accuracy += accuracy

    # Averages
    avg_loss = total_loss / step
    avg_accuracy = total_accuracy / step

    return avg_loss, avg_accuracy

def run_eval_epoch(model, data, gradient_accumulation_steps, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    total_accuracy = 0
    num_batches = len(data["input_ids"])

#    for i in tqdm(range(20)):
    for i in tqdm(range(num_batches)):
        # Get the input data for the current batch (batch size is 1)
        input_ids = data["input_ids"][i].unsqueeze(0).to(device)  # Adding a batch dimension
        attention_mask = data["attention_mask"][i].unsqueeze(0).to(device)  # Adding a batch dimension
        labels = data["labels"][i].unsqueeze(0).to(device)  # Adding a batch dimension

        # Ensure correct dimensions (although the above should ensure this)
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        if len(attention_mask.shape) == 1:
            attention_mask = attention_mask.unsqueeze(0)
        labels = labels.unsqueeze(0)

        # No gradient calculation during evaluation
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            logits = output.logits

        # Get predictions (for batch size 1, we just pick the index of the highest logit)
        preds = logits.argmax().item()  # Get the index of the highest logit as a scalar

        # Calculate accuracy for this batch (since batch size = 1, it's binary)
        accuracy = (preds == labels.item())  # Check if the prediction matches the label

        # Accumulate total loss and accuracy
        total_loss += loss.item()
        total_accuracy += accuracy

        # Append the predictions and labels for precision/recall calculation
        all_preds.append(preds)
        all_labels.append(labels.item())

    # Move to CPU for precision and recall calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate overall precision and recall using scikit-learn
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy, precision, recall


def main(batch_size, device, lr, wd):

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

    model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_epoch = 0 
    best_eval_accuracy = 0
    best_eval_loss = float('inf')
    best_eval_recall = 0
    best_eval_precision = 0

    stopped_early = False
    last_epoch = 0

    print("TRAINING")
    train_start = datetime.now()
    for epoch in range(50):
        print("Train Step...")
        train_loss, train_accuracy = run_train_epoch(model=model, data=train_data, batch_size=batch_size, optimizer=optimizer,  gradient_accumulation_steps=1, device=device)
        logging.info(f'train - {epoch = } # {train_loss = :.2f} # {train_accuracy = :.2f}')
        print("Val Step...")
        eval_loss, eval_accuracy, eval_precision, eval_recall = run_eval_epoch(model=model, data=eval_data,  gradient_accumulation_steps=1, device=device)
        logging.info(f'dev   - {epoch = } # {eval_loss = :.2f} # {eval_accuracy = :.2f}')

        # get test scores
        #test_loss, test_accuracy, test_precision, test_recall = run_eval_epoch(model=model, data=test_data,  gradient_accumulation_steps=1, device=device)
        #logging.info(f'test  - {epoch = } # {test_loss = :.2f} # {test_accuracy = :.2f}')

        if eval_loss < best_eval_loss:
            best_epoch = epoch
            best_eval_accuracy = eval_accuracy
            best_eval_loss = eval_loss
            best_eval_precision = eval_precision
            best_eval_recall = eval_recall

        wandb.log(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss, 
                "eval/accuracy": eval_accuracy, "eval/precision" : eval_precision, "eval/recall" : eval_recall, "eval/loss": eval_loss, 
                "eval/best_accuracy": best_eval_accuracy, "eval/best_precision":best_eval_precision, "eval/best_recall" : best_eval_recall, "eval/best_loss": best_eval_loss,
         #       "test/accuracy": test_accuracy, "test/precision" : test_precision, "test/recall" : test_recall, "test/loss": test_loss, 
            }
        )

        last_epoch = epoch
        if epoch - best_epoch >= 5:
            logging.info(f'stopped early at epoch {epoch}')
            stopped_early = True
            break
        train_end = datetime.now()
    for epoch in range(last_epoch+1, 50):
        wandb.log(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss, 
                "eval/accuracy": eval_accuracy, "eval/precision" : eval_precision, "eval/recall" : eval_recall, "eval/loss": eval_loss, 
                "eval/best_accuracy": best_eval_accuracy, "eval/best_precision":best_eval_precision, "eval/best_recall" : best_eval_recall, "eval/best_loss": best_eval_loss,
          #      "test/accuracy": test_accuracy, "test/precision" : test_precision, "test/recall" : test_recall, "test/loss": test_loss, 
            }
        )
        inf_start = datetime.now()
        test_loss, test_accuracy, test_precision, test_recall = run_eval_epoch(model=model, data=test_data,  gradient_accumulation_steps=1, device=device)
        inf_end = datetime.now()
        wandb.log(
            {"time/inference": str(inf_end - inf_start), "time/training": str(train_end - train_start)})
if __name__ == "__main__": 
    batch_size = 4
    device = "cuda:3" 


    lr = 1e-5
    wd = 1e-4
    run_id = 0

    for i in range(1):
        name = f"ModernBert-lr{lr}-wd{wd}"
        wandb_run = wandb.init(
        project="modernBert",
        name=name,
            config={
                "learning_rate": lr,
                "weight_decay": wd,
            }
        )
        logging.info(f"Running {name}")
        main(batch_size, device, lr, wd)  # You can make batch_size and device configurable too
        logging.info("Done with main")
        wandb_run.finish()
        run_id += 1
