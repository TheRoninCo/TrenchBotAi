(* ::Package:: *)

(* Import both datasets *)
whales = Import["s3://your-bucket/sonar/whales_*.parquet", "Parquet"];
bundles = Import["s3://your-bucket/mev/bundles_*.parquet", "Parquet"];

(* Join by slot and wallet address *)
joined = JoinAcross[
  whales[All, {"slot", "wallet", "personality", "velocity"}],
  bundles[All, {"slot", "tip_lamports", "cu_limit", "success"}],
  "slot"
];

(* Feature engineering *)
dataset = Dataset@Map[{
  #personality /. {
    "Gorilla" -> 1, "Shark" -> 2, "Kraken" -> 3, _ -> 0
  },
  #velocity,
  #tip_lamports,
  #cu_limit,
  Boole@#success
} &, joined];

(* Train combined model *)
model = Predict[
  dataset[[All, ;;4]] -> dataset[[All,5]],
  Method -> "GradientBoostedTrees",
  PerformanceGoal -> "Quality"
];

(* Export for Rust *)
Export["~/models/joint_model.wlnet", model];

(* Generate Prometheus metrics *)
Export["/tmp/joint_metrics.prom", StringTemplate@"
  # TYPE mev_whale_success gauge
  mev_whale_success{personality=\"gorilla\"} ``1``
  mev_whale_success{personality=\"shark\"} ``2``
"[
  model[<|"personality"->1, "velocity"->500, "tip_lamports"->50000, "cu_limit"->200000|>],
  model[<|"personality"->2, "velocity"->1000, "tip_lamports"->100000, "cu_limit"->500000|>]
]];