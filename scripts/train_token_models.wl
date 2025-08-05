(* ::Package:: *)

(* Import token-labeled data *)
data = Import["s3://your-bucket/mev/token_bundles.parquet"];

(* Group by token mint *)
tokenGroups = GroupBy[data, #accounts[[1]] &];

(* Train per-token model *)
models = AssociationMap[
  Function[{token, group},
    Predict[
      group[[All, {"personality", "velocity", "imbalance", "tip", "cu"}]] -> 
      group[[All, "profit"]],
      Method -> "NeuralNetwork"
    ]
  ],
  Keys[tokenGroups]
];

(* Export models *)
Export["~/models/token_models.wxf", models];

(* Generate calibration report *)
calibrationReport = Table[
  <|
    "token" -> token,
    "mae" -> ModelMeasurements[models[token], "MeanAbsoluteError"],
    "r2" -> ModelMeasurements[models[token], "RSquared"]
  |>,
  {token, Keys[models]}
];

Export["~/reports/token_calibration.json", calibrationReport];