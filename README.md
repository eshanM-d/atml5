# Lab 5: WGAN CIFAR-10 Explorer

This project implements a **Wasserstein Generative Adversarial Network (WGAN)** trained on the CIFAR-10 dataset. It features a modular architecture including a training pipeline, a Flask REST API for inference, and an interactive Streamlit frontend for exploring the latent space.

---

## 1. Project Overview
The system is divided into three main components:
* **The Model (`wgan_cifar10.py`)**: Implements a Generator and a Critic using PyTorch. It utilizes Wasserstein loss with weight clipping for improved training stability compared to standard GANs.
* **The API (`flask_api.py`)**: A Flask server that loads the trained checkpoints and provides endpoints for generating images and performing latent space interpolations.
* **The Frontend (`wgan_frontend.py`)**: A Streamlit web application that allows users to interact with the model without writing code.



---

## 2. Technical Specifications
| Feature | Details |
| :--- | :--- |
| **Dataset** | CIFAR-10 ($32 \times 32$ images) |
| **Latent Dimension** | 100 |
| **Optimization** | RMSprop with Weight Clipping ($0.01$) |
| **Backend** | Flask with CORS support |
| **Frontend** | Streamlit |

---

## 3. Setup and Installation

### Step 1: Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install torch torchvision flask flask-cors streamlit pandas pillow
```

### Step 2: Training the Model
Run the training script to generate the necessary model checkpoints and metrics:
```bash
python wgan_cifar10.py --epochs 100 --batch-size 64 --gpu
```
* [cite_start]**Output**: Checkpoints are saved in `./checkpoints/` and training metrics in `training_metrics.json`[cite: 1].

### Step 3: Launch the API
Start the Flask server to handle inference requests:
```bash
python flask_api.py
```
* The API will listen on `http://localhost:5000`.

### Step 4: Run the UI
Open a new terminal and launch the Streamlit dashboard:
```bash
streamlit run wgan_frontend.py
```

---

## 4. Usage Features
* **Image Generation**: Generate a batch of synthetic CIFAR-10 images using random or specific seeds.
* **Latent Interpolation**: Visualize the smooth transition between two different points in the latent space (morphing).
* **Metrics Dashboard**: Track the Wasserstein distance and loss curves to monitor training health.
* **Sample Gallery**: Browse and load sample grids generated during the training process.

---

## 5. File Manifest
* `wgan_cifar10.py`: Model definitions, training loop, and weight initialization.
* `flask_api.py`: REST API endpoints and base64 image encoding logic.
* `wgan_frontend.py`: Streamlit layout, API request handling, and visualization.
* [cite_start]`.gitignore`: Prevents large datasets, caches, and checkpoints from being tracked by Git[cite: 1].
