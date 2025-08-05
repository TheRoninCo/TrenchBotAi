import torch
import polars as pl
import wandb

def train():
    # Init tracking
    wandb.init(project="whale-classifier")
    
    # Load data
    df = pl.read_csv("whale_metrics.csv")
    X = torch.tensor(df.drop("personality").to_numpy(), dtype=torch.float32)
    y = torch.tensor(df["personality"].to_numpy(), dtype=torch.long)
    
    # Model
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 6)  # 6 personality classes
    )
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(100):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        wandb.log({"loss": loss.item()})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train()