# Dynamic Device Selector for PyTorch

This repository provides a robust solution for dynamically selecting the appropriate device (CPU or GPU) based on the memory requirements of a PyTorch model and its input data. The tool ensures optimal utilization of available hardware resources by estimating the memory required for the model and checking GPU capacity before execution.

## 💡 Why Use This Tool?
- Avoid **out-of-memory errors** during model execution.
- Automatically leverage GPU acceleration when resources are available.
- Simplify your codebase by automating device selection.

## How It Works
1. Estimates the memory required for the model (parameters, inputs, and outputs).
2. Checks the available GPU memory.
3. Executes the model on GPU if there is enough free memory; otherwise, falls back to CPU.
