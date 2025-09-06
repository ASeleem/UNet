import logging
from kfp import compiler
from kfp.client import Client

from unet_pipeline import unet_pipeline
from configs import UNetTrainRunParameters, KubeflowConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UNetPipelineRunner:
    """Class to handle UNet pipeline compilation and execution."""

    def __init__(self, config_file: str = ".env"):
        """Initialize the pipeline runner with configuration."""
        self.config_file = config_file
        self.kubeflow_config = None
        self.train_params = None
        self.client = None
        self.compiled_pipeline_path = "unet.yaml"

    def load_configuration(self) -> None:
        """Load configuration from environment variables."""
        try:
            self.kubeflow_config = KubeflowConfig()
            self.train_params = UNetTrainRunParameters()
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.error("Please ensure all required environment variables are set.")
            raise

    def compile_pipeline(self) -> None:
        """Compile the UNet pipeline to YAML."""
        logger.info("Compiling pipeline...")
        try:
            compiler.Compiler().compile(unet_pipeline, self.compiled_pipeline_path)
            logger.info("Pipeline compiled successfully.")
        except Exception as e:
            logger.error(f"Error compiling pipeline: {e}")
            raise

    def create_client(self) -> None:
        """Create Kubeflow Pipelines client."""
        if not self.kubeflow_config:
            raise ValueError("Configuration not loaded. Call load_configuration() first.")

        try:
            self.client = Client(
                host=self.kubeflow_config.kubeflow_host,
                namespace=self.kubeflow_config.kubeflow_user_namespace,
                verify_ssl=False,
                cookies=f"oauth2_proxy_kubeflow={self.kubeflow_config.kubeflow_user_token}"
            )
            logger.info("Kubeflow client created successfully.")
        except Exception as e:
            logger.error(f"Error creating Kubeflow client: {e}")
            raise

    def upload_pipeline(self, pipeline_name: str = "unet-pipeline") -> None:
        """Upload pipeline to Kubeflow."""
        if not self.client:
            raise ValueError("Client not created. Call create_client() first.")

        logger.info("Uploading pipeline...")
        try:
            self.client.upload_pipeline(
                pipeline_package_path=self.compiled_pipeline_path,
                pipeline_name=pipeline_name,
                description='A pipeline to train U-Net model on Carvana dataset'
            )
            logger.info("Pipeline uploaded successfully.")
        except Exception as e:
            if hasattr(e, 'status') and e.status == 409:
                logger.info("Pipeline already exists, skipping upload.")
            else:
                logger.error(f"Error uploading pipeline: {e}")
                raise

    def create_run(self) -> str:
        """Create and start a pipeline run."""
        if not self.client or not self.train_params:
            raise ValueError("Client or parameters not initialized.")

        logger.info("Creating pipeline run...")
        try:
            run = self.client.create_run_from_pipeline_package(
                self.compiled_pipeline_path,
                arguments={
                    'mlflow_tracking_uri': self.train_params.mlflow_tracking_uri,
                    'minio_endpoint': self.train_params.minio_endpoint,
                    'minio_access_key': self.train_params.minio_access_key,
                    'minio_secret_key': self.train_params.minio_secret_key,
                    'bucket_name': self.train_params.minio_bucket_name,
                    'dataset': self.train_params.dataset,
                    'data_dir': self.train_params.data_dir,
                    'epochs': self.train_params.epochs,
                    'batch_size': self.train_params.batch_size,
                    'learning_rate': self.train_params.learning_rate,
                    'acceptance_dc': self.train_params.acceptance_dc
                },
                experiment_name=self.train_params.experiment_name,
                namespace=self.kubeflow_config.kubeflow_user_namespace
            )
            logger.info(f"Pipeline run created successfully: {run}")
            return run
        except Exception as e:
            logger.error(f"Error creating pipeline run: {e}")
            raise

    def run_pipeline(self, pipeline_name: str = "unet-pipeline") -> str:
        """Execute the complete pipeline workflow."""
        self.load_configuration()
        self.compile_pipeline()
        self.create_client()
        self.upload_pipeline(pipeline_name)
        return self.create_run()


def main():
    """Main function to run the pipeline."""
    runner = UNetPipelineRunner()
    try:
        run_id = runner.run_pipeline()
        logger.info(f"Pipeline execution started successfully with run ID: {run_id}")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
