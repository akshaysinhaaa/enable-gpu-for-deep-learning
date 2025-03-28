# Enable GPU Usage Locally for Deep Learning

This is a step-by-step guide to enable GPU training for deep learning projects in any code editor, whether it's Jupyter Notebook, VS Code, or PyCharm Professional.

> **Note:** This setup was personally tested on my university's lab computer, which has an **NVIDIA RTX A6000 GPU**.

---

## Step 1: Download NVIDIA Video Driver  
- Check your GPU model by running the following command in the terminal:

```sh
nvidia-smi
```
- Copy and paste your GPU name into the search bar on NVIDIA's driver download page.
- Download the latest driver for your GPU:
  - [NVIDIA GPU Driver Download](https://www.nvidia.com/Download/index.aspx)

---

## Step 2: Install Visual Studio C++  
- Download and install **Visual Studio Community Edition**.
- Open Visual Studio and update if prompted, then close it.
- [Download Visual Studio](https://visualstudio.microsoft.com/vs/community/)

---

## Step 3: Install Anaconda Navigator  
- Install everything using the default settings.
- [Download Anaconda](https://www.anaconda.com/download/success)

---

## Step 4: Install CUDA Toolkit  
- Run the following command to check your CUDA version:

```sh
nvidia-smi
```
- Visit [PyTorch Local Installation Guide](https://pytorch.org/get-started/locally/) to find the correct CUDA version.
- Download the **CUDA Toolkit** for your version:
  - Windows -> Architecture: `x86_64`
  - Version: `11` (For windows 11)
  - Installer Type: `.exe (local)`
  - [Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

### **Verify CUDA Installation:**
1. Go to `C:\Program Files`
2. Look for **NVIDIA GPU Computing Toolkit** folder
3. Inside, check for a **CUDA** folder
4. Open it and ensure it contains a folder named `12.x` (or your installed version)

---

## Step 5: Install cuDNN  
- Download the latest **cuDNN** version.
- Choose **Local Windows Installer (ZIP)**.
- [Download cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)

---

## Step 6: Copy & Paste cuDNN Files  
### **Steps:**
1. Open `File Explorer` and split the screen into two sections.
2. Navigate to:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\12.x
   ```
3. Open the **cuDNN archive** in another window.
4. Copy and paste the contents of the following folders:
   - all contents of **bin** from cuDNN archive  → Paste into `CUDA\12.x\bin`
   - all contents of **include** from cuDNN archive  → Paste into `CUDA\12.x\include`
   - all contents of **lib** from cuDNN archive  → Paste into `CUDA\12.x\lib`

### **Verify Environment Variables:**
1. Open **Edit the System Environment Variables** via Windows Search.
2. Check if **CUDA_PATH** and **CUDA_PATH_V12_x** exist under system variables.
3. If they match your CUDA installation path, you're good to go!

---

## Step 7: Install PyTorch  
- Go to [PyTorch Official Site](https://pytorch.org/get-started/locally/)
- Select the **stable** version.
- Choose the correct **CUDA** version.
- Copy the given pip command and paste it into your VS Code terminal:

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12
```

### **Using Git Bash in VS Code:**
1. Open **VS Code Terminal**
2. Change the terminal to **Git Bash**
3. Paste and execute the copied command

---

## Step 8: Verify GPU Installation  
Run the following script to check if your GPU is correctly set up:

```python
import torch

print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
```

### **Expected Output:**
```
Number of GPUs: 1
GPU Name: NVIDIA RTX A6000
Using device: cuda
```

---

✅ **Now, you are ready to train deep learning models using GPU!** 🚀