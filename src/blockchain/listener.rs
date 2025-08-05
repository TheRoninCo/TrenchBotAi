// src/blockchain/listener.rs
async fn handle_tx(tx: Transaction) {
    omni_logger::log_blockchain_event(
        tx, 
        LogPriority::COMBAT_CRITICAL
    ).await;
}