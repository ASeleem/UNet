"""Configuration classes for U-Net training and Kubeflow connection."""
from pydantic_settings import BaseSettings
from pydantic import Field

class UNetTrainRunParameters(BaseSettings):
    """Parameters for running the U-Net training pipeline.
    """
    experiment_name: str = Field(default="unet", alias="EXPERIMENT_NAME")
    mlflow_tracking_uri: str = Field(..., alias="MLFLOW_TRACKING_URI")
    minio_endpoint: str = Field(..., alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(..., alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(..., alias="MINIO_SECRET_KEY")
    minio_bucket_name: str = Field(default="datasets", alias="MINIO_BUCKET_NAME")
    dataset: str = Field(default="carvana.zip", alias="DATASET")
    data_dir: str = Field(default="/Data/datasets/carvana", alias="DATA_DIR")
    epochs: int = Field(default=5, alias="EPOCHS")
    batch_size: int = Field(default=8, alias="BATCH_SIZE")
    learning_rate: float = Field(default=0.0003, alias="LEARNING_RATE")
    acceptance_dc: float = Field(default=0.90, alias="ACCEPTANCE_DC")

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "case_sensitive": False
    }

class KubeflowConfig(BaseSettings):
    """Kubeflow connection configuration.
    """
    kubeflow_host: str = Field(..., alias="KUBEFLOW_HOST")
    kubeflow_user_namespace: str = Field(..., alias="KUBEFLOW_USER_NAMESPACE")
    kubeflow_user_token: str = Field(..., alias="KUBEFLOW_USER_TOKEN")
    unet_image_name: str = Field(default="aseleem/unet-train:latest", alias="UNET_IMAGE_NAME")

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "case_sensitive": False
    }
