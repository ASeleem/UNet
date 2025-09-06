"""UNet Image Segmentation Pipeline using Kubeflow Pipelines (KFP)"""
from kfp import dsl, kubernetes
from kfp.dsl import component

from configs import KubeflowConfig

## -- load kubeflow config to get image name -- ##
try:
    kubeflow_config = KubeflowConfig()
except Exception as e:
    print(f"Error loading configuration: {e}")
    print("Please ensure all required environment variables are set.")
    raise

## -- Prepare Data Component -- ##
@component(
    base_image="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel",
    target_image=f"{kubeflow_config.unet_image_name}",
    packages_to_install=[
        "minio>=7.1.0",
        "kfp>=2.14.0",
        "kfp-kubernetes>=2.14.0"
    ],
    install_kfp_package=True
)
def prepare_data(minio_endpoint: str = "minio-service.kubeflow:9000",
                 minio_access_key: str = "",
                 minio_secret_key: str = "",
                 bucket_name: str = "datasets",
                 dataset: str = "carvana.zip",
                 data_dir: str = "/Data/datasets/carvana",
    ):
    import os
    import zipfile

    from minio import Minio

    # Connect to MinIO
    client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )
    ## -- Download and extract dataset -- ##
    # Ensure the bucket exists
    if not client.bucket_exists(bucket_name):
        raise ValueError(f"Bucket {bucket_name} does not exist in MinIO.")
    # Download the dataset tar file from MinIO
    tar_path = os.path.join(data_dir, "carvana.zip")
    os.makedirs(data_dir, exist_ok=True)
    client.fget_object(bucket_name, dataset, tar_path)

    # Extract the tar file
    with zipfile.ZipFile(tar_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    # Remove the tar file after extraction to save space
    os.remove(tar_path)

    # Extract train and masks zip files
    with zipfile.ZipFile(f"{data_dir}/train.zip", "r") as zip_ref:
        zip_ref.extractall(data_dir)
    with zipfile.ZipFile(f"{data_dir}/train_masks.zip", "r") as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(f"{data_dir}/train.zip")
    os.remove(f"{data_dir}/train_masks.zip")



## -- Train and Test U-Net Model Component -- ##
@component(
    base_image="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel",
    target_image=f"{kubeflow_config.unet_image_name}",
    packages_to_install=[
        "torch==2.6.0+cu126",
        "torchvision==0.21.0+cu126",
        "numpy>=1.24.0",
        "mlflow>=2.6.0",
        "onnx>=1.19.0",
        "onnxruntime>=1.19.0",
        "tqdm>=4.65.0",
        "pillow>=10.0.0",
        "kfp>=2.14.0",
        "kfp-kubernetes>=2.14.0"
    ],
    install_kfp_package=True
)
def train_test_unet(
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    data_dir: str = "/Data/datasets/carvana",
    acceptance_dc: float = 0.90,
    mlflow_tracking_uri: str = "http://mlflow-tracking.mlflow:80",
    experiment_name: str = "unet-experiment"
):
    import json

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
    import torch.optim as optim
    import numpy as np
    from tqdm import tqdm
    import mlflow
    from mlflow.models.signature import ModelSignature
    from mlflow.types import Schema, TensorSpec
    import onnx

    from unet.UNet import UNet
    from data.CarvanaDataset import CarvanaDataset
    from unet.utils import dice_coefficient

    ## -- Init MLflow -- ##
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    ## -- Prepare DataLoaders -- ##
    train_dataset = CarvanaDataset(data_dir)
    generator = torch.Generator().manual_seed(25)

    # Split the dataset into train and test sets
    train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)
    # Split the test set into validation and test sets
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)
    # Log dataset sizes
    mlflow.log_param("Train Size", len(train_dataset))
    mlflow.log_param("Validation Size", len(val_dataset))
    mlflow.log_param("Test Size", len(test_dataset))
    mlflow.log_param("Batch Size", batch_size)
    mlflow.log_param("Epochs", epochs)
    mlflow.log_param("Learning Rate", learning_rate)
    mlflow.log_param("Acceptance Dice Coefficient", acceptance_dc)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_workers = 0
    # Uncomment below to use multiple workers
    # if device == "cuda":
    #     num_workers = torch.cuda.device_count() * 4

    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=num_workers,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=False)
    val_dataloader = DataLoader(dataset=val_dataset,
                                num_workers=num_workers,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=False)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 num_workers=num_workers,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=False)

    ## -- Initialize Model, Loss, Optimizer -- ##
    model = UNet(in_channels=3, num_classes=1)

    # Uncomment below to use DataParallel for multi-GPU training
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_val_dc = 0.0
    ## -- Training Loop -- ##
    for epoch in tqdm(range(epochs)):
        model.train()
        training_running_loss = 0.0
        training_dice_score = 0.0

        for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            dc = dice_coefficient(y_pred, mask)
            loss = criterion(y_pred, mask)

            training_running_loss += loss.item()
            training_dice_score += dc.item()

            loss.backward()
            optimizer.step()
        train_loss = training_running_loss / (idx + 1)
        train_dc = training_dice_score / (idx + 1)
        mlflow.log_metric("Train Loss", train_loss, step=epoch)
        mlflow.log_metric("Train Dice Coefficient", train_dc, step=epoch)

        model.eval()
        val_running_loss = 0.0
        val_dice_score = 0.0

        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)

                dc = dice_coefficient(y_pred, mask)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()
                val_dice_score += dc.item()
            val_loss = val_running_loss / (idx + 1)
            val_dc = val_dice_score / (idx + 1)
            mlflow.log_metric("Validation Loss", val_loss, step=epoch)
            mlflow.log_metric("Validation Dice Coefficient", val_dc, step=epoch)

        # Save model checkpoint if validation dice coefficient improves
        if val_dc > best_val_dc:
            best_val_dc = val_dc
            torch.save(model.state_dict(), "/Data/best_model.pth")
            mlflow.log_artifact("/Data/best_model.pth", artifact_path="unet_checkpoints")

    ## -- End of Training -- ##

    # Log the final model
    torch.save(model.state_dict(), "/Data/final_model.pth")
    mlflow.log_artifact("/Data/final_model.pth", artifact_path="unet_checkpoints")
    # Save the best validation dice coefficient to a JSON file
    with open("/Data/best_val_dc.json", "w") as f:
        json.dump({"best_val_dc": best_val_dc}, f)
    mlflow.log_artifact("/Data/best_val_dc.json", artifact_path="unet_metrics")

    ## -- Testing Loop -- ##
    model.load_state_dict(torch.load("/Data/best_model.pth"))
    model.eval()

    test_running_loss = 0
    test_running_dc = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(test_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)

            dc = dice_coefficient(y_pred, mask)
            loss = criterion(y_pred, mask)

            test_running_loss += loss.item()
            test_running_dc += dc.item()
        test_loss = test_running_loss / (idx + 1)
        test_dc = test_running_dc / (idx + 1)
        mlflow.log_metric("Test Loss", test_loss)
        mlflow.log_metric("Test Dice Coefficient", test_dc)


        ## -- Register Model if Acceptance Criteria Met -- ##
        if test_dc >= acceptance_dc:
            # Input Schema
            input_schema = Schema([TensorSpec(np.dtype("float32"), (-1, -1, -1, -1), name="images")])
            output_schema = Schema([TensorSpec(np.dtype("float32"), (-1, 1, -1, -1), name="masks")])
            signature = ModelSignature(input_schema, output_schema)

            # Export the model as ONNX
            dummy_images = torch.randn(3, 3, 512, 512).to(device)
            onnx_path = "/Data/unet.onnx"
            torch.onnx.export(model,
                              dummy_images,
                              onnx_path,
                              input_names=["images"],
                              output_names=["masks"],
                              dynamic_axes={"images": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
                                            "masks": {0: "batch_size", 1: "channels", 2: "height", 3: "width"}},
                              opset_version=17)
            
            # Log the ONNX model with MLflow
            onnx_model = onnx.load(onnx_path)
            mlflow.onnx.log_model(onnx_model,
                                  name="unet",
                                  signature=signature,
                                  registered_model_name="unet")


# -- Define the Pipeline -- #
@dsl.pipeline(
    name="U-Net Pipeline",
    description="A pipeline that trains and evaluates a U-Net model for image segmentation using the Carvana dataset."
)
def unet_pipeline(
    mlflow_tracking_uri: str = "http://mlflow-tracking.mlflow:80",
    experiment_name: str = "unet",
    minio_endpoint: str = "minio-service.kubeflow:9000",
    minio_access_key: str = "",
    minio_secret_key: str = "",
    bucket_name: str = "datasets",
    dataset: str = "carvana.zip",
    data_dir: str = "/Data/datasets/carvana",
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    acceptance_dc: float = 0.90,
):
    """U-Net Image Segmentation Pipeline."""
    # Create a Persistent Volume Claim (PVC) for data sharing between components
    pvc = kubernetes.CreatePVC(
        pvc_name_suffix='-unet-pvc',
        access_modes=['ReadWriteMany'],
        # If your storage provisioner does not support ReadWriteMany, use ReadWriteOnce
        # access_modes=['ReadWriteOnce'],
        size='100Gi'
    )

    # -- Prepare Data -- #
    prepare_data_op = prepare_data(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        dataset=dataset,
        data_dir=data_dir
    )
    kubernetes.mount_pvc(prepare_data_op, pvc_name=pvc.outputs['name'], mount_path='/Data')
    # If your cluster storage provisioner does not support ReadWriteMany, you may need to add node selector
    # to ensure that the pod runs on a specific node that has access to the PVC.
    # Example below assumes a node with hostname 'cluster-01-worker3' has access to the PVC.
    # kubernetes.add_node_selector(prepare_data_op, 'kubernetes.io/hostname', 'cluster-01-worker3')
    prepare_data_op.set_caching_options(False)

    # -- Train and Test U-Net Model -- #
    train_test_unet_op = train_test_unet(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        data_dir=data_dir,
        acceptance_dc=acceptance_dc,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name
    )
    kubernetes.mount_pvc(train_test_unet_op, pvc_name=pvc.outputs['name'], mount_path='/Data')
    train_test_unet_op.after(prepare_data_op)
    train_test_unet_op.set_caching_options(False)

    # If your cluster storage provisioner does not support ReadWriteMany, you may need to add node selector
    # to ensure that the pod runs on a specific node that has access to the PVC.
    # Example below assumes a node with hostname 'cluster-01-worker3' has access to the PVC.
    # kubernetes.add_node_selector(prepare_data_op, 'kubernetes.io/hostname', 'cluster-01-worker3')

    # Set resource requests and limits as needed
    # train_test_unet_op.set_gpu_request("4")
    # train_test_unet_op.set_memory_request("16Gi")
    # train_test_unet_op.set_memory_limit("32Gi")
    # train_test_unet_op.set_cpu_request("16")
    # train_test_unet_op.set_cpu_limit("32")
    # train_test_unet_op.set_accelerator_type("hopper")
    