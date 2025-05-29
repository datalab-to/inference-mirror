#!/usr/bin/env python3
"""
Integration test script for the combined Docker container.
Tests PDF processing performance by measuring processing time and pages per second.
Submits files in parallel to saturate workers.
"""

import os
import sys
import time
import asyncio
import aiohttp
import aiofiles
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pypdfium2
import docker
import threading
import click
import psutil

try:
    import pynvml

    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False


class DockerContainerManager:
    """Manages Docker container lifecycle using Docker Python API."""

    def __init__(
        self,
        image_name: str = "datalab-inference-combined",
        container_name: str = "test-inference",
    ):
        self.image_name = image_name
        self.container_name = container_name
        self.client = docker.from_env()
        self.container = None

    def build_image(self) -> bool:
        """Build the Docker image."""
        print(f"Building Docker image: {self.image_name}")
        try:
            self.client.images.build(
                path=".",
                dockerfile="Dockerfile.combined",
                tag=self.image_name,
                rm=True,
            )
            print("✓ Docker image built successfully")
            return True
        except docker.errors.BuildError as e:
            print(f"✗ Failed to build Docker image: {e}")
            return False

    def start_container(self, port: int = 8000) -> bool:
        """Start the Docker container."""
        print(f"Starting container: {self.container_name}")
        try:
            # Stop and remove existing container if it exists
            try:
                existing = self.client.containers.get(self.container_name)
                existing.stop()
                existing.remove()
            except docker.errors.NotFound:
                pass

            # Start new container with GPU access
            self.container = self.client.containers.run(
                self.image_name,
                name=self.container_name,
                ports={"8000/tcp": port},
                device_requests=[
                    docker.types.DeviceRequest(device_ids=["0"], capabilities=[["gpu"]])
                ],
                detach=True,
            )

            print(f"✓ Container started with ID: {self.container.id}")

            # Wait for container to be ready
            return self._wait_for_health_check(port)

        except Exception as e:
            print(f"✗ Failed to start container: {e}")
            return False

    def _wait_for_health_check(self, port: int, timeout: int = 60) -> bool:
        """Wait for the container to be ready by checking health endpoint."""
        print("Waiting for container to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                import requests

                response = requests.get(
                    f"http://localhost:{port}/health_check", timeout=5
                )
                if response.status_code == 200:
                    print("✓ Container is ready")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)

        print("✗ Container failed to become ready within timeout")
        return False

    def stop_container(self):
        """Stop and remove the Docker container."""
        if self.container:
            print(f"Stopping container: {self.container_name}")
            try:
                self.container.stop()
                self.container.remove()
                print("✓ Container stopped and removed")
            except Exception as e:
                print(f"Warning: Error stopping container: {e}")


class ResourceMonitor:
    """Monitors CPU and GPU utilization during testing."""

    def __init__(self, container_name: str = "test-inference"):
        self.container_name = container_name
        self.monitoring = False
        self.cpu_samples = []
        self.gpu_samples = []
        self.monitor_thread = None

        # Initialize NVIDIA ML if available
        self.gpu_available = False
        if NVIDIA_ML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_max_power = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(
                    self.gpu_handle
                )
                self.gpu_available = True
            except Exception as e:
                print(f"Warning: Could not initialize GPU monitoring: {e}")

    def _get_container_cpu_usage(self) -> float:
        """Get CPU usage percentage of system"""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception:
            return 0.0

    def _get_gpu_utilization(self) -> Tuple[float, float, float]:
        """Get GPU utilization percentage, power draw percentage, and memory usage."""
        if not self.gpu_available:
            return 0.0, 0.0, 0.0

        try:
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            gpu_util = utilization.gpu

            # Power draw
            power_draw_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
            power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(
                self.gpu_handle
            )[1]
            power_percent = (
                (power_draw_mw / power_limit_mw) * 100 if power_limit_mw > 0 else 0.0
            )

            # Memory usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            memory_percent = (memory_info.used / memory_info.total) * 100

            return gpu_util, power_percent, memory_percent
        except Exception:
            return 0.0, 0.0, 0.0

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            timestamp = time.time()

            # Get CPU usage
            cpu_usage = self._get_container_cpu_usage()
            self.cpu_samples.append((timestamp, cpu_usage))

            # Get GPU metrics
            gpu_util, power_percent, memory_percent = self._get_gpu_utilization()
            self.gpu_samples.append(
                (timestamp, gpu_util, power_percent, memory_percent)
            )

            time.sleep(1)  # Sample every second

    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.cpu_samples = []
        self.gpu_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✓ Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("✓ Resource monitoring stopped")

    def get_stats(self) -> Dict:
        """Get resource utilization statistics."""
        if not self.cpu_samples and not self.gpu_samples:
            return {
                "cpu_avg": 0.0,
                "cpu_max": 0.0,
                "gpu_util_avg": 0.0,
                "gpu_util_max": 0.0,
                "gpu_power_avg": 0.0,
                "gpu_power_max": 0.0,
                "gpu_memory_avg": 0.0,
                "gpu_memory_max": 0.0,
                "samples_count": 0,
            }

        # CPU stats
        cpu_values = [sample[1] for sample in self.cpu_samples]
        cpu_avg = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
        cpu_max = max(cpu_values) if cpu_values else 0.0

        # GPU stats
        gpu_util_values = [sample[1] for sample in self.gpu_samples]
        gpu_power_values = [sample[2] for sample in self.gpu_samples]
        gpu_memory_values = [sample[3] for sample in self.gpu_samples]

        gpu_util_avg = (
            sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else 0.0
        )
        gpu_util_max = max(gpu_util_values) if gpu_util_values else 0.0

        gpu_power_avg = (
            sum(gpu_power_values) / len(gpu_power_values) if gpu_power_values else 0.0
        )
        gpu_power_max = max(gpu_power_values) if gpu_power_values else 0.0

        gpu_memory_avg = (
            sum(gpu_memory_values) / len(gpu_memory_values)
            if gpu_memory_values
            else 0.0
        )
        gpu_memory_max = max(gpu_memory_values) if gpu_memory_values else 0.0

        return {
            "cpu_avg": cpu_avg,
            "cpu_max": cpu_max,
            "gpu_util_avg": gpu_util_avg,
            "gpu_util_max": gpu_util_max,
            "gpu_power_avg": gpu_power_avg,
            "gpu_power_max": gpu_power_max,
            "gpu_memory_avg": gpu_memory_avg,
            "gpu_memory_max": gpu_memory_max,
            "samples_count": len(self.cpu_samples),
        }


class AsyncPDFProcessor:
    """Handles PDF processing via the API with async support."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def get_pdf_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF file."""
        try:
            doc = pypdfium2.PdfDocument(pdf_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            print(f"Warning: Could not get page count for {pdf_path}: {e}")
            return 0

    async def submit_pdf(self, pdf_path: str, config: Dict = None) -> str:
        """Submit a PDF for processing and return the file_id."""
        if config is None:
            config = {}

        async with aiofiles.open(pdf_path, "rb") as f:
            file_content = await f.read()

            data = aiohttp.FormData()
            data.add_field(
                "file",
                file_content,
                filename=os.path.basename(pdf_path),
                content_type="application/pdf",
            )
            data.add_field("config", json.dumps(config))

            async with self.session.post(
                f"{self.base_url}/marker/inference", data=data
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["file_id"]

    async def check_status(self, file_id: str) -> Tuple[str, str]:
        """Check the status of a processing job. Returns (status, result/error)."""
        async with self.session.get(
            f"{self.base_url}/marker/results", params={"file_id": file_id}
        ) as response:
            response.raise_for_status()
            result = await response.json()
            status = result.get("status")

            if status == "done":
                return "done", "completed"
            elif status == "failed":
                return "failed", result.get("error", "Unknown error")
            elif status == "processing":
                return "processing", ""
            else:
                return "unknown", f"Unknown status: {status}"

    async def wait_for_completion(
        self, file_id: str, timeout: int = 1200
    ) -> Tuple[bool, str]:
        """Wait for PDF processing to complete. Returns (success, result/error)."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                status, result = await self.check_status(file_id)

                if status == "done":
                    return True, result
                elif status == "failed":
                    return False, result
                elif status == "processing":
                    await asyncio.sleep(2)
                    continue
                else:
                    return False, result

            except Exception as e:
                return False, f"Request failed: {e}"

        return False, "Timeout waiting for completion"


class PerformanceTester:
    """Runs performance tests on PDF files with parallel submission."""

    def __init__(
        self,
        pdf_dir: str,
        base_url: str = "http://localhost:8000",
        max_concurrent: int = 10,
        max_files: int | None = None,
        format_lines: bool = False,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.results: List[Dict] = []
        self.results_lock = threading.Lock()
        self.max_files = max_files
        self.config = {
            "format_lines": format_lines,
        }

    def find_pdf_files(self) -> List[Path]:
        """Find all PDF files in the directory."""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        pdf_files.extend(self.pdf_dir.glob("**/*.pdf"))
        if self.max_files is not None:
            pdf_files = pdf_files[: self.max_files]
        return sorted(pdf_files)

    async def process_single_pdf(
        self, processor: AsyncPDFProcessor, pdf_path: Path
    ) -> Dict:
        """Process a single PDF file asynchronously."""
        print(f"\nSubmitting: {pdf_path.name}")

        # Get page count (sync operation)
        page_count = processor.get_pdf_page_count(str(pdf_path))
        print(f"  Pages: {page_count}")

        start_time = time.time()
        try:
            # Submit for processing
            file_id = await processor.submit_pdf(str(pdf_path), self.config)
            submit_time = time.time()
            print(
                f"  Submitted with file_id: {file_id} (submit time: {submit_time - start_time:.2f}s)"
            )

            # Wait for completion
            success, result = await processor.wait_for_completion(file_id)
            end_time = time.time()

            total_time = end_time - start_time
            processing_time = end_time - submit_time
            pages_per_second = (
                page_count / processing_time if processing_time > 0 else 0
            )

            test_result = {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
                "page_count": page_count,
                "submit_time": submit_time - start_time,
                "processing_time": processing_time,
                "total_time": total_time,
                "pages_per_second": pages_per_second,
                "success": success,
                "result": result,
            }

            if success:
                print(
                    f"  ✓ {pdf_path.name} completed in {processing_time:.2f}s ({pages_per_second:.2f} pages/sec)"
                )
            else:
                print(f"  ✗ {pdf_path.name} failed: {result}")

            return test_result

        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            print(f"  ✗ {pdf_path.name} error: {e}")

            return {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
                "page_count": page_count,
                "submit_time": 0,
                "processing_time": total_time,
                "total_time": total_time,
                "pages_per_second": 0,
                "success": False,
                "result": str(e),
            }

    async def run_tests_async(self) -> List[Dict]:
        """Run tests on all PDF files with controlled concurrency."""
        pdf_files = self.find_pdf_files()

        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return []

        print(f"Found {len(pdf_files)} PDF files to test")
        print(f"Using max concurrency: {self.max_concurrent}")

        async with AsyncPDFProcessor(self.base_url) as processor:
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def process_with_semaphore(pdf_file):
                async with semaphore:
                    return await self.process_single_pdf(processor, pdf_file)

            # Submit all files concurrently but with limited concurrency
            tasks = [process_with_semaphore(pdf_file) for pdf_file in pdf_files]
            self.results = await asyncio.gather(*tasks)

        return self.results

    def run_tests(self) -> List[Dict]:
        """Run tests synchronously (wrapper for async version)."""
        return asyncio.run(self.run_tests_async())

    def print_summary(self):
        """Print a summary of test results."""
        if not self.results:
            print("\nNo test results to summarize")
            return

        successful_tests = [r for r in self.results if r["success"]]
        failed_tests = [r for r in self.results if not r["success"]]

        total_pages = sum(r["page_count"] for r in successful_tests)
        total_processing_time = sum(r["processing_time"] for r in successful_tests)
        total_wall_time = max((r["total_time"] for r in self.results), default=0)

        print("\n" + "=" * 70)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 70)
        print(f"Total files tested: {len(self.results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(failed_tests)}")
        print(f"Max concurrent submissions: {self.max_concurrent}")

        if successful_tests:
            print(f"\nTotal pages processed: {total_pages}")
            print(f"Total processing time (sum): {total_processing_time:.2f}s")
            print(f"Total wall clock time: {total_wall_time:.2f}s")
            print(
                f"Throughput (pages/sec): {total_pages / total_wall_time if total_wall_time > 0 else 0:.2f}"
            )
            print(
                f"Processing efficiency: {total_processing_time / total_wall_time if total_wall_time > 0 else 0:.2f}"
            )

            avg_pages_per_second = sum(
                r["pages_per_second"] for r in successful_tests
            ) / len(successful_tests)
            print(f"Average pages per second per file: {avg_pages_per_second:.2f}")

            fastest = max(successful_tests, key=lambda x: x["pages_per_second"])
            slowest = min(successful_tests, key=lambda x: x["pages_per_second"])

            print(
                f"\nFastest: {fastest['filename']} ({fastest['pages_per_second']:.2f} pages/sec)"
            )
            print(
                f"Slowest: {slowest['filename']} ({slowest['pages_per_second']:.2f} pages/sec)"
            )

            # Submission timing analysis
            avg_submit_time = sum(r["submit_time"] for r in successful_tests) / len(
                successful_tests
            )
            print(f"\nAverage submission time: {avg_submit_time:.3f}s")

        if failed_tests:
            print("\nFailed files:")
            for test in failed_tests:
                print(f"  - {test['filename']}: {test['result']}")

    def print_resource_summary(self, resource_stats: Dict):
        """Print resource utilization summary."""
        print("\n" + "=" * 70)
        print("RESOURCE UTILIZATION SUMMARY")
        print("=" * 70)

        if resource_stats["samples_count"] == 0:
            print("No resource monitoring data available")
            return

        print(f"Monitoring samples collected: {resource_stats['samples_count']}")

        # CPU utilization
        print("\nCPU Utilization:")
        print(f"  Average: {resource_stats['cpu_avg']:.1f}%")
        print(f"  Peak: {resource_stats['cpu_max']:.1f}%")

        # GPU utilization
        if resource_stats["gpu_util_avg"] > 0 or resource_stats["gpu_power_avg"] > 0:
            print("\nGPU Utilization:")
            print(f"  Average utilization: {resource_stats['gpu_util_avg']:.1f}%")
            print(f"  Peak utilization: {resource_stats['gpu_util_max']:.1f}%")
            print(
                f"  Average power draw: {resource_stats['gpu_power_avg']:.1f}% of max"
            )
            print(f"  Peak power draw: {resource_stats['gpu_power_max']:.1f}% of max")
            print(f"  Average memory usage: {resource_stats['gpu_memory_avg']:.1f}%")
            print(f"  Peak memory usage: {resource_stats['gpu_memory_max']:.1f}%")
        else:
            print("\nGPU: Not available or not utilized")

        # Efficiency analysis
        if resource_stats["gpu_util_avg"] > 0:
            if resource_stats["gpu_util_avg"] > 80:
                efficiency = "Excellent"
            elif resource_stats["gpu_util_avg"] > 60:
                efficiency = "Good"
            elif resource_stats["gpu_util_avg"] > 40:
                efficiency = "Moderate"
            else:
                efficiency = "Poor"

            print(f"\nGPU Efficiency: {efficiency}")
            print(
                f"(Based on {resource_stats['gpu_util_avg']:.1f}% average utilization)"
            )

        if resource_stats["cpu_avg"] > 0:
            if resource_stats["cpu_avg"] > 80:
                cpu_efficiency = "High"
            elif resource_stats["cpu_avg"] > 50:
                cpu_efficiency = "Moderate"
            else:
                cpu_efficiency = "Low"

            print(f"\nCPU Usage: {cpu_efficiency}")
            print(f"(Based on {resource_stats['cpu_avg']:.1f}% average utilization)")


@click.command()
@click.option(
    "--pdf_dir", type=str, required=True, help="Directory containing PDF files to test"
)
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--build", is_flag=True, help="Build Docker image before testing")
@click.option(
    "--max-concurrent",
    type=int,
    default=10,
    help="Maximum number of concurrent PDF submissions",
)
@click.option(
    "--max_files", type=int, default=None, help="Maximum number of PDF files to process"
)
@click.option(
    "--format_lines", is_flag=True, help="Format lines in the output (default: False)"
)
def main(
    pdf_dir: str,
    port: int,
    build: bool,
    max_concurrent: int,
    max_files: int,
    format_lines: bool,
):
    # Verify PDF directory exists
    if not os.path.isdir(pdf_dir):
        print(f"Error: Directory {pdf_dir} does not exist")
        sys.exit(1)

    container_manager = None

    try:
        container_manager = DockerContainerManager()

        # Build image if requested
        if build:
            if not container_manager.build_image():
                sys.exit(1)

        # Start container
        if not container_manager.start_container(port):
            sys.exit(1)

        # Start resource monitoring
        monitor = ResourceMonitor(container_manager.container_name)
        monitor.start_monitoring()

        # Run performance tests
        tester = PerformanceTester(
            pdf_dir, f"http://localhost:{port}", max_concurrent, max_files, format_lines
        )
        start_time = time.time()
        tester.run_tests()
        total_time = time.time() - start_time

        # Stop monitoring and get resource stats
        monitor.stop_monitoring()
        resource_stats = monitor.get_stats()

        print(f"\nTotal test execution time: {total_time:.2f}s")
        tester.print_summary()
        tester.print_resource_summary(resource_stats)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)
    finally:
        # Clean up container
        if container_manager:
            container_manager.stop_container()


if __name__ == "__main__":
    main()
