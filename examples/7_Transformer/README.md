# Example 7 - Transformer for Precipitation Prediction

This example demonstrates how to use FTorch to integrate a Transformer model for predicting precipitation based on latitude and longitude. The process includes defining, training, and saving a Transformer model in Python, then using FTorch in Fortran to load the model and run inference.

---

## Description

In this example:
1. A Transformer model is defined and trained in Python using PyTorch.
2. The trained model is converted to a TorchScript file for use in FTorch.
3. FTorch is used to load the TorchScript model and run inference from Fortran.
4. The program outputs precipitation predictions for given latitude and longitude inputs.

---


## Steps

### **1. Define and Train the Transformer in Python**

- Create a Python file (`transformer_precipitation_model.py`) to define and train the Transformer model.
- Train the model with a dataset of latitude and longitude inputs and corresponding precipitation values.
- Save the trained model in `.pth` format.
- Convert the trained model to TorchScript format (`.pt`) for compatibility with FTorch.

---

### **2. Prepare FTorch for Fortran Integration**

- Install FTorch as described in the main FTorch documentation.
- Write a `CMakeLists.txt` file to build the Fortran code. This file should include the FTorch library and link it to the Fortran executable.

---

### **3. Write Fortran Inference Script**

- Create a Fortran script (`transformer_infer_fortran.f90`) that:
  - Loads the TorchScript model using `torch_model_load`.
  - Maps input latitude and longitude data to tensors using `torch_tensor_from_array`.
  - Runs inference using `torch_model_forward`.
  - Extracts and prints the model's predictions.

---

### **4. Run the Workflow**

1. **Install Python Dependencies**
   - From the `7_Transformer` directory, run:
     ```bash
     pip install -r requirements.txt
     ```

2. **Train and Save the Transformer Model**
   - Run the Python script to define, train, and save the model:
     ```bash
     python transformer_precipitation_model.py
     ```

   - This will generate the `transformer_precipitation_model.pt` file in the current directory.

3. **Build the Fortran Code**
   - Create a `build` directory and configure the project:
     ```bash
     mkdir build
     cd build
     cmake .. -DCMAKE_PREFIX_PATH=<path_to_ftorch_installation>
     make
     ```

4. **Run the Fortran Program**
   - After building, run the generated executable:
     ```bash
     ./transformer_infer_fortran
     ```

---

## Key Fortran Code for Data Conversion and Inference

The following code demonstrates how FTorch integrates Fortran arrays with PyTorch tensors:

```fortran
! Map Fortran arrays to PyTorch tensors
call torch_tensor_from_array(in_tensors(1), in_data, in_layout, torch_kCPU)
call torch_tensor_from_array(out_tensors(1), out_data, out_layout, torch_kCPU)

! Infer
call torch_model_forward(model, in_tensors, out_tensors)
```

## Output Example

When running the example, the output will include the input latitude and longitude values along with the predicted precipitation values. For example:

```plaintext
Input Data (Latitude, Longitude):
   0.0       0.0
  45.0      90.0
 -45.0     -90.0
  90.0     180.0
 -90.0    -180.0

Predicted Precipitation:
   0.0054
   0.7845
  -0.2342
   1.1920
  -0.9731
