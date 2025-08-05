impl MevProcessor {
    pub async fn should_submit(
        &self,
        bundle: &Bundle,
        whale: Option<&Whale>,
        token: &str,  // Primary token mint in bundle
    ) -> anyhow::Result<bool> {
        let imbalance = self.pool_analyzer.get_imbalance(token).await?;
        let personality = whale.map(|w| w.personality.as_score()).unwrap_or(0.0);
        let velocity = whale.map(|w| w.velocity).unwrap_or(0.0);

        let profit = self.token_predictor.predict(
            token,
            personality,
            velocity,
            imbalance,
            bundle.tip_lamports as f32,
            bundle.cu_limit as f32,
        )?;

        Ok(profit > self.config.min_profit)
    }
}