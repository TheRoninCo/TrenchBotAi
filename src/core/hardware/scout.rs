//! RECON TEAM - System health checks
pub struct ReconTeam {
    probes: Vec<Box<dyn Probe>>,
}

impl ReconTeam {
    pub fn scout() -> Self {
        Self {
            probes: vec![
                Box::new(GpuProbe),
                Box::new(RpcProbe),
                Box::new(MemoryProbe),
            ],
        }
    }

    pub fn run_check(&self) -> ReconReport {
        let mut report = ReconReport::new();
        self.probes.iter().for_each(|p| p.probe(&mut report));
        report
    }
}