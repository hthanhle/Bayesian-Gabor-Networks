# Bayesian Gabor Network with Uncertainty Estimation for Pedestrian Lane Segmentation in Assistive Navigation
## Descriptions
We propose a new light-weight Bayesian Gabor Network for camera-based detection of pedestrian lanes in unstructured scenes. The proposed method is fast, compact, and
suitable for real-time operations on edge computers.

![alt_text](/output/image/000029.jpg) ![alt_text](/output/image/000059.jpg) ![alt_text](/output/image/000219.jpg) ![alt_text](/output/image/000371.jpg)  ![alt_text](/output/image/000981.jpg) ![alt_text](/output/image/008639.jpg) 
![alt_text](/output/groundtruth/000029.png) ![alt_text](/output/groundtruth/000059.png) ![alt_text](/output/groundtruth/000219.png) ![alt_text](/output/groundtruth/000371.png)  ![alt_text](/output/groundtruth/000981.png) ![alt_text](/output/groundtruth/008639.png) 
![alt_text](/output/segmentation/000029.png) ![alt_text](/output/segmentation/000059.png) ![alt_text](/output/segmentation/000219.png) ![alt_text](/output/segmentation/000371.png)  ![alt_text](/output/segmentation/000981.png) ![alt_text](/output/segmentation/008639.jpg) 
![alt_text](/output/aleatoric/000029.png) ![alt_text](/output/aleatoric/000059.png) ![alt_text](/output/aleatoric/000219.png) ![alt_text](/output/aleatoric/000371.png)  ![alt_text](/output/aleatoric/000981.png) ![alt_text](/output/aleatoric/008639.jpg)
![alt_text](/output/epistemic/000029.png) ![alt_text](/output/epistemic/000059.png) ![alt_text](/output/epistemic/000219.png) ![alt_text](/output/epistemic/000371.png)  ![alt_text](/output/epistemic/000981.png) ![alt_text](/output/epistemic/008639.jpg) 
**Figure 1.** Examples of pedestrian lane-detection results produced by BGN. A brighter intensity in the uncertainty maps indicates a higher uncertainty level. Row 1: Input images. Row 2: Ground-truth. Row 3: Output segmentation maps. Row 4: Output aleatoric uncertain maps. Row 5: Output epistemic uncertain maps.
## Quick start
### Installation
1. Install PyTorch=1.2.0 following [the official instructions](https://pytorch.org/)

2. git clone https://github.com/hthanhle/Bayesian-Gabor-Networks

3. Install dependencies: `pip install -r requirements.txt`

### Data preparation

The lane dataset PLVP3 is available at: http://documents.uow.edu.au/~phung/plvp3.html

The data should be under: `./lane_dataset/PLVP3/`

### Train and test

1. To train a Bayesian Gabor Network, run the following command: `python train.py` 
2. To test the model, run the following command: `python test.py`

## Citation
If you find this work or code is helpful for your research, please cite:
```
@ARTICLE{9684439,
  author={Le, Hoang Thanh and Phung, Son Lam and Bouzerdoum, Abdesselam},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Bayesian Gabor Network with Uncertainty Estimation for Pedestrian Lane Detection in Assistive Navigation}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3144184}}
  ```
## Reference
[1] H. T. Le, S. L. Phung and A. Bouzerdoum, "Bayesian Gabor Network with Uncertainty Estimation for Pedestrian Lane Detection in Assistive Navigation," IEEE Transactions on Circuits and Systems for Video Technology, 2022, doi: 10.1109/TCSVT.2022.3144184.
