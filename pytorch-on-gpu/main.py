import torch


def test_gpu():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    device_number = torch.cuda.current_device()
    print(f"Device number: {device_number}")

    device_name = torch.cuda.get_device_name(device_number)
    print(f"Device name: {device_name}")


def some_gpu_operation():
    # create a tensor on the CPU
    T1 = torch.randn(4, 4)
    print(f" CPU tensor: {T1}")

    # move the tensor to the GPU
    T2 = T1.to("cuda")  # device from the test_gpu function
    print(f" GPU tensor: {T2}")


if __name__ == "__main__":
    test_gpu()
    some_gpu_operation()
