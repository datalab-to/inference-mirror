# Datalab Inference Service

Containerized inference service for [marker](https://github.com/VikParuchuri/marker). 

# Features

- Upload and validate PDF files
- Queue files for asynchronous processing across multiple workers
- Automatically chunks large PDFs across multiple workers
- Retrieve file status or download results

# Setup
Generate the docker compose file using
```bash
python generate_compose.py $FNAME --gpus $NUM_GPUS --workers_per_gpu $NUM_WORKERS_PER_GPU
```
A sample docker compose file has been provided in `docker-compose.yaml`
Before starting the inference service, enable NVIDIA MPS Server. This speeds up multiple processes sharing a single GPU. You will be prompted for `sudo` access
```bash
# For example, on a system with 3 GPUs - sudo ./start_mps.sh 0 1 2 
./start_mps.sh <GPU_ID_LIST>
```
A few environment variables must be set in a `.env` file, as detailed in the `.env.example` file. Copy over the defaults and modify them as required.
Finally, start the service using
```bash
cp .env.example .env
docker compose up
```

# Recommended Configurations
Here are a few recommended configurations that have been tested on a few different GPUs, to help set the number of workers and batch sizes
- **1xH100 GPU 80GB** (30 CPUs and 200GB RAM)
```
NUM_WORKERS_PER_GPU=8
10 PDFs; 840 pages   ->    29.42s (28.552 pages/s)     

with `format_lines` enabled
10 PDFs; 840 pages   ->    109.42s (7.677 pages/s)
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