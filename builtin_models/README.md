# builtin_models

Run `make install-deps` in the root `onnxruntime` (`..`) to get the dependencies needed for generating bindata.
Run `make generate` in the root `onnxruntime` (`..`) after updating model descriptions.

# model test
### Image Classification
Run `go test -run ImageClassification`

| Name                        | Image                               | Label                  | Score       |
|:---------------------------:|:-----------------------------------:|:----------------------:|:-----------:|
| TorchVision_AlexNet         | ../predictor/_fixtures/platypus.jpg | n01873310 platypus ... | 15.774      |

### Image Object Detection
Run `go test -run ObjectDetection`

Note: Only recording first five detection, excluding background.
| Name                        | Image                                   | Label | Xmin  | Xmax  | Ymin  | Ymax  | Probability |
|:---------------------------:|:---------------------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|
| OnnxVision_SSD              | ../predictor/_fixtures/lane_control.jpg | car   | 0.502 | 0.603 | 0.289 | 0.432 | 0.972       |
|                             |                                         | car   | 0.567 | 0.801 | 0.111 | 0.293 | 0.962       |
|                             |                                         | car   | 0.571 | 0.810 | 0.017 | 0.300 | 0.955       |
|                             |                                         | car   | 0.538 | 0.596 | 0.495 | 0.560 | 0.874       |
|                             |                                         | car   | 0.600 | 0.966 | 0.725 | 0.999 | 0.792       |

### Image Instance Segmentation
Run `go test -run InstanceSegmentation`

| Name                          | Image                                   | Label | Probability |
|:-----------------------------:|:---------------------------------------:|:-----:|:-----------:|
| OnnxVision_Mask_RCNN_R_50_FPN | ../predictor/_fixtures/lane_control.jpg | car   | 0.972       |
