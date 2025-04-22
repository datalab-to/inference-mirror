import yaml
import click

@click.command()
@click.argument('output_file_path')
@click.option('--gpus', default=1, show_default=True, help='Number of GPUs to use.')
@click.option('--workers_per_gpu', default=1, show_default=True, help='Number of workers per GPU.')
def generate_compose(output_file_path, gpus, workers_per_gpu):
    """Generate a docker-compose file with GPU-bound worker services."""

    # Worker base template
    worker_template = {
        "image": "us-central1-docker.pkg.dev/inference-build/inference-images/worker:latest",
        "depends_on": {
            "rabbitmq": {"condition": "service_healthy"},
            "server": {"condition": "service_started"}
        },
        "volumes": [
            "${DATALAB_INFERENCE_DATA_VOLUME}:/data",
            "${DATALAB_INFERENCE_OUTPUT_VOLUME}:/output"
        ],
        "environment": {
            "RABBITMQ_HOST": "rabbitmq",
            "DATA_DIR": "/data",
            "OUTPUT_DIR": "/output",
            "COMPILE_MODELS": "${DATALAB_COMPILE_MODELS}",
            "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
            "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-log",
            "CHUNK_SIZE": "${DATALAB_INFERENCE_CHUNK_SIZE}",
            "RECOGNITION_BATCH_SIZE": "${DATALAB_INFERENCE_RECOGNITION_BATCH_SIZE}",
            "DETECTION_BATCH_SIZE": "${DATALAB_INFERENCE_DETECTION_BATCH_SIZE}",
            "TEXIFY_BATCH_SIZE": "${DATALAB_INFERENCE_TEXIFY_BATCH_SIZE}",
            "TABLE_REC_BATCH_SIZE": "${DATALAB_INFERENCE_TABLE_REC_BATCH_SIZE}",
            "LAYOUT_BATCH_SIZE": "${DATALAB_INFERENCE_LAYOUT_BATCH_SIZE}",
            "OCR_ERROR_BATCH_SIZE": "${DATALAB_INFERENCE_OCR_ERROR_BATCH_SIZE}"
        },
        "restart": "unless-stopped"
    }

    # Core services
    services = {
        "rabbitmq": {
            "image": "rabbitmq:management",
            "restart": "always",
            "ports": ["${DATALAB_RABBITMQ_MANAGEMENT_PORT}:15672"],
            "healthcheck": {
                "test": ["CMD", "rabbitmqctl", "status"],
                "interval": "10s",
                "retries": 5,
                "timeout": "5s",
                "start_period": "30s"
            }
        },
        "server": {
            "image": "us-central1-docker.pkg.dev/inference-build/inference-images/server:latest",
            "ports": ["8510:8000"],
            "depends_on": {
                "rabbitmq": {"condition": "service_healthy"}
            },
            "volumes": [
                "${DATALAB_INFERENCE_DATA_VOLUME}:/data",
                "${DATALAB_INFERENCE_OUTPUT_VOLUME}:/output"
            ],
            "environment": {
                "RABBITMQ_HOST": "rabbitmq",
                "DATA_DIR": "/data"
            },
            "restart": "unless-stopped"
        }
    }

    # Dynamically generate workers
    for gpu_id in range(gpus):
        worker_name = f"gpu{gpu_id}-worker"
        worker_config = dict(worker_template)
        worker_config["deploy"] = {
            "resources": {
                "reservations": {
                    "devices": [{
                        "driver": "nvidia",
                        "device_ids": [str(gpu_id)],
                        "capabilities": ["gpu"]
                    }]
                }
            },
            "replicas": workers_per_gpu
        }
        services[worker_name] = worker_config
        worker_config["volumes"].extend([f"/tmp/nvidia-mps-{gpu_id}:/tmp/nvidia-mps", f"/tmp/nvidia-log-{gpu_id}:/tmp/nvidia-log"])

    compose = {
        "services": services
    }

    with open(output_file_path, "w") as f:
        yaml.dump(compose, f, sort_keys=False)

    click.echo(f"Generated {output_file_path} with {gpus * workers_per_gpu} workers across {gpus} GPU(s).")

if __name__ == '__main__':
    generate_compose()