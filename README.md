# Datalab Inference Service

Containerized inference service for [marker](https://github.com/VikParuchuri/marker). 

# Features

- Upload and validate PDF files
- Queue files for asynchronous processing across multiple workers
- Automatically chunks large PDFs across multiple workers
- Retrieve file status or download results

# Setup

This will run a single container on a single GPU, and will run enough parallel marker workers to saturate the GPU.

```bash
export IMAGE_TAG=us-central1-docker.pkg.dev/inference-build/inference-images/combined:latest
docker pull $IMAGE_TAG
docker run --gpus device=0 -p 8000:8000 $IMAGE_TAG # Container can only handle one GPU
```

## CPU

If you want to run the combined container on CPU, you will need to set these environment variables when you run the Docker container (the variables must be set inside the container, and available to the entrypoint script):

```json
{
    "TORCH_NUM_THREADS": "NUM_PHYSICAL_CORES", // Equal to psutil.cpu_count(logical=False)
    "OPENBLAS_NUM_THREADS": 4,
    "OMP_NUM_THREADS": 4,
    "RECOGNITION_BATCH_SIZE": 16,
    "DETECTION_BATCH_SIZE": 4,
    "TABLE_REC_BATCH_SIZE": 6,
    "LAYOUT_BATCH_SIZE": 6,
    "OCR_ERROR_BATCH_SIZE": 6,
    "DETECTOR_POSTPROCESSING_CPU_WORKERS": 4,
    "CHUNK_SIZE": 6
}
```

Then you just run the container without mounting a GPU:

```bash
docker run -p 8000:8000 $IMAGE_TAG # Container can only handle one GPU
```

# Recommended Configurations
Here are a few recommended configurations that have been tested on a few different GPUs, to help set the number of workers and batch sizes
- **1xH100 GPU 80GB** (30 CPUs and 200GB RAM)
```
10 PDFs; 840 pages   ->    29.42s (28.552 pages/s)     

with `format_lines` enabled
10 PDFs; 840 pages   ->    109.42s (9.31 pages/s)
```

# API Description and Endpoints

## `GET /health_check`

**Description:**  
Check if the service is up and running.

**Response:**  
```json
{ "status": "healthy" }
```

**Python Example:**
```python
import requests

res = requests.get("http://localhost:8000/health_check")
print(res.json())
```

---

## `POST /marker/inference`

**Description:**  
Upload a PDF and queue it for processing.

**Form Data:**

- `file` (UploadFile, required): The PDF file to process.
- `config` (str, optional): A JSON string containing configuration options.

**Response:**
```json
{ "file_id": "<file_id>" }
```

**Python Example:**
```python
import requests

files = {'file': open('example.pdf', 'rb')}
data = {'config': '{"some_setting": true}'}
res = requests.post("http://localhost:8000/marker/inference", files=files, data=data)
print(res.json())
```

---

## `GET /marker/results`

**Description:**  
Check the status of a file or download the results once processing is done.

**Query Parameters:**

- `file_id` (str, required): The ID returned from the `/marker/inference` endpoint.
- `download` (bool, optional): If `true`, returns merged output and image URLs.

**Response (examples):**

**If processing is still ongoing:**
```json
{ "file_id": "<file_id>", "status": "processing" }
```

**If failed:**
```json
{ "file_id": "<file_id>", "status": "failed", "error": "Reason for failure" }
```

**If done:**
```json
{
  "file_id": "<file_id>",
  "status": "done",
  "result": "...",
  "images": ["https://.../image1.png", "..."]
}
```

**Python Example:**
```python
import requests

params = {"file_id": "your-file-id", "download": True}
res = requests.get("http://localhost:8000/marker/results", params=params)
print(res.json())
```

## `POST /marker/clear`

**Description:**
Clear the results of a file that has been processed.

**Data:**
- `file_id` (str, required): The ID of the file to clear.

**Response:**
```json
{ "status": "cleared", "file_id": "<file_id>" }
```

**Python Example:**
```python
import requests
data = {'file_id': 'your-file-id'}
res = requests.post("http://localhost:8000/marker/clear", json=data)
print(res.json())
```

# Integration Testing

You can run a small performance test with this script:

```bash
CUDA_VISIBLE_DEVICES=0 python integration_test.py --pdf_dir PATH/TO/PDFs --build
```