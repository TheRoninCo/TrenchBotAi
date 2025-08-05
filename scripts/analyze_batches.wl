(* Process compressed batches *)
batches = FileNames["~/mev_data/batches/*.parquet"];
data = Dataset@Flatten[Import[#, "Parquet"] & /@ batches;

(* Train tip optimization model *)
tipModel = Predict[
  data[All, {"tip_lamports", "cu_limit"} -> "success"],
  Method -> "RandomForest"
];

(* Export model for Rust *)
Export["~/mev_data/tip_model.wxf", tipModel];