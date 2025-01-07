import torch

def estimate_model_memory(model, input_size):
    try:
        model_params_size = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"Model parameters size: {model_params_size / (1024 ** 2):.2f} MB")
        dummy_input = torch.randn(*input_size)
        input_size_bytes = dummy_input.numel() * dummy_input.element_size()
        print(f"Input size: {input_size_bytes / (1024 ** 2):.2f} MB")
        with torch.no_grad():
            output = model(dummy_input)
        output_size_bytes = output.numel() * output.element_size()
        print(f"Output size: {output_size_bytes / (1024 ** 2):.2f} MB")
        total_memory = model_params_size + input_size_bytes + output_size_bytes
        return total_memory
    except Exception as e:
        print(f"Error in estimating model memory: {e}")
        return float("inf")
def check_cuda_availability(model, input_size,device_id):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    total_memory = torch.cuda.get_device_properties(device_id).total_memory
    reserved_memory = torch.cuda.memory_reserved(device_id)
    allocated_memory = torch.cuda.memory_allocated(device_id)
    free_memory = total_memory - reserved_memory - allocated_memory
    estimated_memory = estimate_model_memory(model, input_size)
    print(f"Estimated memory required: {estimated_memory / (1024 ** 2):.2f} MB")
    print(f"Available memory: {free_memory / (1024 ** 2):.2f} MB")
    if estimated_memory < free_memory:
        return torch.device("cuda")
    else:
        return torch.device("cpu")
