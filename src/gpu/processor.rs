// src/gpu/processor.rs
use trenchie_core::killfeed::NormalizedEvent;

pub struct GpuProcessor {
    model: ort::Session, // ONNX runtime
}

impl GpuProcessor {
    pub fn new() -> Self {
        let model = ort::Session::load("model.onnx").unwrap();
        Self { model }
    }

    pub fn predict(&self, event: NormalizedEvent) -> f32 {
        // Convert to tensor and run inference
        let inputs = vec![
            ort::Value::from_array(event.features).unwrap()
        ];
        let outputs = self.model.run(inputs).unwrap();
        outputs[0].get_float().unwrap()
    }
}