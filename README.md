# skeleton_fusion
The skeleton fusion tool reads two frames from two cameras and gets two skeletons from the OpenPose [1], and the Graduate Non-Convexity [2] optimization algorithm is implemented to obtain a single robust 3D skeleton.

## Requirements
Cuda (tested 10.2, 11.8) <br />
*cudnn  (tested 7.6.5, 8x) <br />
OpenPose [1] (1.7.0)<br />
Apriltag [3] <br />

## References
[1] OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose)  <br />
[2] Graduate Non-Convexity from the work: Robust Event Detection based on Spatio-Temporal Latent Action Unit using Skeletal Information (https://ieeexplore.ieee.org/abstract/document/9636553) <br />
[3] Apriltag (https://april.eecs.umich.edu/software/apriltag) <br />
