# PyTorch on GPU: Simple Test

This repository contains a simple test script to verify that PyTorch is properly configured and has access to your NVIDIA GPU via CUDA.

## Prerequisites

Before running the test, ensure that you have the correct NVIDIA graphics drivers and CUDA Toolkit installed.

1. **Check your NVIDIA setup:**
   Run the following command in your terminal to find your NVIDIA driver and CUDA version:
   ```bash
   nvidia-smi
   ```

2. **Install the matching PyTorch version:**
   Visit the [PyTorch Local Start Guide](https://pytorch.org/get-started/locally/) to select the appropriate installation command for your setup, OS, and CUDA version.

---

## Getting Started

This project uses `uv` for fast package management.

### Installation

If you have `uv` installed, you can run the script directly, and it will handle dependencies automatically. Alternatively, you can install the dependencies into your environment:

```bash
uv sync
```

---

## How to Run the Test

To run the GPU verification script, execute:

```bash
uv run python main.py
```

---

## Code Overview (`main.py`)

The verification script contains two main helper functions:

### 1. `test_gpu()`
Checks if CUDA is available, identifies the current device ID, and queries the GPU model name:
```python
def test_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    device_number = torch.cuda.current_device()
    print(f"Device number: {device_number}")

    device_name = torch.cuda.get_device_name(device_number)
    print(f"Device name: {device_name}")
```

### 2. `some_gpu_operation()`
Creates a random 4x4 matrix tensor on the CPU and transfers it to the GPU:
```python
def some_gpu_operation():
    # create a tensor on the CPU
    T1 = torch.randn(4, 4)
    print(f" CPU tensor: {T1}")

    # move the tensor to the GPU
    T2 = T1.to("cuda")
    print(f" GPU tensor: {T2}")
```

---

## Expected Output

When run successfully on a GPU-enabled environment, you should see output similar to this:

```text
Using device: cuda
Device number: 0
Device name: NVIDIA GeForce RTX 5080 Laptop GPU
 CPU tensor: tensor([[-0.7864,  1.3581, -1.2307, -1.0441],
        [ 0.5284, -1.1062,  1.4906, -0.3464],
        [-0.1283, -0.0799, -0.3552,  0.7625],
        [-1.0369,  0.2755,  0.8702, -1.2210]])
 GPU tensor: tensor([[-0.7864,  1.3581, -1.2307, -1.0441],
        [ 0.5284, -1.1062,  1.4906, -0.3464],
        [-0.1283, -0.0799, -0.3552,  0.7625],
        [-1.0369,  0.2755,  0.8702, -1.2210]], device='cuda:0')
```

Note the `device='cuda:0'` parameter on the GPU tensor, indicating that the tensor has successfully been allocated on the first GPU device.
