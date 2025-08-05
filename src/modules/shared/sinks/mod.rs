pub mod mongo;
pub mod s3;
pub mod kafka;
pub mod local;

pub use mongo::MongoSink;
pub use s3::S3Sink;
pub use kafka::KafkaSink;
pub use local::LocalSink;
