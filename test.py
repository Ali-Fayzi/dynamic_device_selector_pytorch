import torch.nn as nn
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6400, 6400)
        self.fc2 = nn.Linear(6400, 100)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def fill_gpu_memory():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"Total GPU memory: {total_memory / (1024 ** 2):.2f} MB")
        tensor_size = int(0.98 * (total_memory )) 
        print(f"Allocating tensor of size: {tensor_size / (1024 ** 2):.2f} MB")
        large_tensor = torch.empty(int(tensor_size), dtype=torch.uint8, device="cuda")
        print("GPU memory filled successfully.")
    except RuntimeError as e:
        print(f"Failed to allocate memory: {e}")
if __name__ =="__main__":
  fill_gpu_memory() #fill 98% of memory
  # define input size and neural network
  input_size = (6400, 6400)  
  model = MyNeuralNetwork()
  device = check_cuda_availability(model, input_size,device_id=0)
  print(f"Using device: {device}")
  # finally
  device = torch.device(device)
  print(f"Selected Device:{device}")
