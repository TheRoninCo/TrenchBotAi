import pandas as pd
from pathlib import Path

def migrate_old_format(old_path: Path, new_path: Path):
    df = pd.read_parquet(old_path)
    
    # Transform old schema to new
    df_new = df.rename(columns={
        "wallet": "address",
        "type": "personality" 
    }).assign(
        timestamp=pd.to_datetime(df["time"]),
        holding_period=df["hold_hours"]
    )
    
    df_new.to_csv(new_path, index=False)

if __name__ == "__main__":
    migrate_old_format(
        Path("data/old/whales.parquet"),
        Path("data/new/whale_metrics.csv")
    )