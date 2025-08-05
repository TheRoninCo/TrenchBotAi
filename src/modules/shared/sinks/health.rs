use aws_sdk_s3::Client as S3Client;
use mongodb::Collection;

impl SinkHealth {
    pub async fn check_s3(client: &S3Client, bucket: &str) -> bool {
        client.head_bucket()
            .bucket(bucket)
            .send()
            .await
            .is_ok()
    }

    pub async fn check_mongo(collection: &Collection) -> bool {
        collection.count_documents(None, None)
            .await
            .is_ok()
    }
}