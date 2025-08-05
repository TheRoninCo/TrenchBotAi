#[test]
fn verify_cuda_alignment() {
    assert_eq!(std::mem::size_of::<Transaction>(), 40);
    assert_eq!(std::mem::align_of::<Transaction>(), 4);
}