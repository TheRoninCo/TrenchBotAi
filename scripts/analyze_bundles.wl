(* Load Rust-exported data *)
data = Import["~/mev_data/bundles_1234567890.csv"];

(* Train tip success model *)
tipModel = Predict[
  data[[All, {"tip_lamports", "cu_limit", "cu_price"}] -> data[[All, "success"]],
  Method -> "GradientBoostedTrees"
];

(* Export optimal tip curve *)
optimalTips = Table[
  {cu, tipModel[<|"cu_limit" -> cu, "cu_price" -> price|>]},
  {cu, 100000, 1000000, 100000}, {price, 0, 1000, 50}
];
Export["~/mev_data/optimal_tips.json", optimalTips];