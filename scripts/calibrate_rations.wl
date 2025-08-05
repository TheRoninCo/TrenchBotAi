(* ::Package:: *)

(* Import recent profit data *)
profits = Import["s3://your-bucket/mev/profit_history.parquet"];

(* Calculate volatility-adjusted thresholds *)
volatility = StandardDeviation[profits[[All, "profit"]];
meanProfit = Mean[profits[[All, "profit"]];

(* Dynamic tier calculation *)
newThresholds = If[volatility > 5000, 
  (* High volatility: wider bins *)
  {0, 250, 1000, 4000, 15000, 50000, 150000, Infinity},
  (* Normal market: tighter bins *)
  {0, 100, 500, 2000, 5000, 10000, 50000, 100000, Infinity}
];

(* Save for Rust *)
Export["~/configs/ration_thresholds.json", newThresholds];

(* Generate calibration report *)
calibrationReport = <|
  "timestamp" -> Now,
  "volatility" -> volatility,
  "mean_profit" -> meanProfit,
  "thresholds" -> newThresholds,
  "recommendation" -> If[volatility > 5000, "WidenTiers", "NormalTiers"]
|>;

Export["~/reports/ration_calibration_" <> DateString[Now, "ISODateTime"] <> ".json", calibrationReport];