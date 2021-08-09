# Bayesian Gabor Network with Uncertainty Estimation for Pedestrian Lane Segmentation in Assistive Navigation
## Descriptions
This project proposes a new light-weight Bayesian Gabor Network for camera-based detection of pedestrian lanes in unstructured scenes. The proposed method is fast, compact, and
suitable for real-time operations on edge computers.

![alt_text](/output/image/000029.jpg) ![alt_text](/output/image/000059.jpg) ![alt_text](/output/image/000219.jpg) ![alt_text](/output/image/000371.jpg)  ![alt_text](/output/image/000981.jpg) ![alt_text](/output/image/008639.jpg) 
![alt_text](/output/groundtruth/000029.png) ![alt_text](/output/groundtruth/000059.png) ![alt_text](/output/groundtruth/000219.png) ![alt_text](/output/groundtruth/000371.png)  ![alt_text](/output/groundtruth/000981.png) ![alt_text](/output/groundtruth/008639.png) 
![alt_text](/output/segmentation/000029.png) ![alt_text](/output/segmentation/000059.png) ![alt_text](/output/segmentation/000219.png) ![alt_text](/output/segmentation/000371.png)  ![alt_text](/output/segmentation/000981.png) ![alt_text](/output/segmentation/008639.jpg) 
![alt_text](/output/aleatoric/000029.png) ![alt_text](/output/aleatoric/000059.png) ![alt_text](/output/aleatoric/000219.png) ![alt_text](/output/aleatoric/000371.png)  ![alt_text](/output/aleatoric/000981.png) ![alt_text](/output/aleatoric/008639.jpg)
![alt_text](/output/epistemic/000029.png) ![alt_text](/output/epistemic/000059.png) ![alt_text](/output/epistemic/000219.png) ![alt_text](/output/epistemic/000371.png)  ![alt_text](/output/epistemic/000981.png) ![alt_text](/output/epistemic/008639.jpg) 
**Figure 1.** Examples from PLVP3 dataset. Row 1: Input colour images. Row 2: ground-truth. Row 3: Output segmentation maps. Row 4: Output aleatoric uncertainty maps. Row 5: Output epistemic uncertainty maps.
## Quick start
### Installation
1. Install PyTorch=1.2.0 following [the official instructions](https://pytorch.org/)

2. git clone https://github.com/hthanhle/Bayesian-Gabor-Networks

3. Install dependencies: `pip install -r requirements.txt`

### Data preparation

The lane dataset PLVP3 is available at: http://documents.uow.edu.au/~phung/plvp3.html

The data should be under: `./lane_dataset/PLVP3/`

### Train and test

Please run the following commands: `python train.py` and `python test.py`

