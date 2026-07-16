# Traffic Sign Classification using a Custom CNN in PyTorch

Classifying German traffic signs from images using a custom Convolutional Neural Network (CNN) built with PyTorch.

**Live API:** [Link](https://traffic-sign-api-gvhe.onrender.com/docs)

---

## Overview

This project performs 43-class traffic sign image classification using a custom Convolutional Neural Network (CNN).

The model is built entirely from scratch in PyTorch and trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The project also includes an inference pipeline, FastAPI REST API, and Docker containerization for deployment.

---

## Model Architecture

- **Model:** Custom Convolutional Neural Network (CNN)
- **Framework:** PyTorch
- **Convolution Layers:**
  - Conv2D (3 → 32)
  - ReLU
  - MaxPool2D
  - Conv2D (32 → 64)
  - ReLU
  - MaxPool2D
- **Classifier:**
  - Flatten
  - Linear (2304 → 128)
  - ReLU
  - Linear (128 → 43)
- **Input Resolution:** 32 × 32 RGB images
- **Image Preprocessing:**
  - Resize (32 × 32)
  - ToTensor()

---

## Training Configuration

- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Epochs:** 10
- **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)
- **Checkpointing:** Trained model saved as `traffic_sign_cnn.pth`

---

## Performance

**Test Accuracy:** **93.53%**

The model achieved 93.53% accuracy on the GTSRB test set after training for 10 epochs, demonstrating strong performance using a compact CNN architecture built from scratch.

---

## Deployment

The project includes:

- FastAPI inference API
- Docker support for reproducible deployment
- Image upload endpoint
- Confidence score prediction
- Saved model checkpoint loading

---

## Dataset

This project uses the German Traffic Sign Recognition Benchmark (GTSRB), containing over 50,000 images across 43 traffic sign classes.

The dataset is not included in this repository and should be placed in the `data/` directory before retraining the model.

The pretrained checkpoint (`models/traffic_sign_cnn.pth`) is included, so the API can be used immediately without retraining.

---

## Running the Project

### Clone the repository

```bash
git clone https://github.com/vanshikacy/traffic-sign-classification
cd traffic-sign-classification
```

### Build the Docker image

```bash
docker build -t traffic-sign-api .
```

### Run the container

```bash
docker run -p 8000:8000 traffic-sign-api
```

### Access the API

Swagger UI:

```
http://localhost:8000/docs
```

Interactive API documentation:

```
http://localhost:8000/redoc
```

---

## API Endpoint

### Predict Traffic Sign

**POST**

```
/predict
```

Upload an image using multipart/form-data.

**Response**

```json
{
  "prediction": {
    "class_id": 16,
    "class_name": "Vehicles over 3.5 tons prohibited",
    "confidence": 0.9987
  }
}
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- FastAPI
- Docker
- Pillow

The model is served through a FastAPI REST API and can be deployed locally using Docker.