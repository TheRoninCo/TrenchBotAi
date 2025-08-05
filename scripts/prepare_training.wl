CombatRationLabel[profit_] := Switch[
  Abs[profit],
  _? (# < 100 &),   "KetchupPacket",
  _? (# < 500 &),   "ShrimpSnack",
  _? (# < 2000 &),  "HappyMeal",
  _? (# < 5000 &),  "DoubleRation",
  _? (# < 10000 &), "FieldSteak",
  _? (# < 50000 &), "MessHall",
  _? (# < 100000 &),"AdmiralsFeast",
  _,                "KrakenHarvest"
]

(* Apply to dataset *)
labeledData = Map[
  Append[#, "profitClass" -> CombatRationLabel[#profit]] &,
  rawData
];