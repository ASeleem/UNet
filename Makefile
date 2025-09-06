# Makefile for UNet Pipeline

VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

.PHONY: help train serve setup build run_pipeline clean

# Default target is help
help:
	@echo "Available targets:"
	@echo "  train       - Setup, build, and run the training pipeline"
	@echo "  serve       - TODO: Serve the trained model"
	@echo "  setup       - Create virtual environment and install dependencies"
	@echo "  build       - Build KFP components and push images"
	@echo "  run_pipeline- Execute the pipeline"
	@echo "  clean       - Remove virtual environment and generated files"
	@echo "  help        - Show this help message"

# Train target: setup, build, run pipeline, then clean
train: setup build run_pipeline clean
	@echo "Training pipeline completed."

# Serve target: TODO
serve:
	@echo "TODO: Implement model serving functionality"

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install kfp>=2.14.0 kfp-kubernetes>=2.14.0 docker pydantic pydantic_settings python-dotenv
	@echo "Virtual environment setup complete."

# Build KFP components and push images
build:
	@echo "Building KFP components..."
	. $(VENV_DIR)/bin/activate && kfp component build . --component-filepattern "unet_pipeline.py" --push-image
	@echo "Component build complete."

# Run the pipeline
run_pipeline:
	@echo "Running UNet pipeline..."
	$(PYTHON) run_unet_pipeline.py
	@echo "Pipeline execution started."

# Clean up virtual environment
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR) component_metadata
	rm -f unet.yaml kfp_config.ini runtime-requirements.txt .dockerignore Dockerfile =2.14.0
	@echo "Cleanup complete."
