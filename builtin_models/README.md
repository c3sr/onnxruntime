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
| MobileNet_SSD_v1.0          | ../predictor/_fixtures/lane_control.jpg | car   | 0.574 | 0.990 | 0.634 | 1.004 | 0.998       |
|                             |                                         | car   | 0.567 | 0.801 | 0.001 | 0.344 | 0.998       |
|                             |                                         | car   | 0.575 | 0.990 | 0.629 | 0.998 | 0.997       |
|                             |                                         | car   | 0.584 | 0.801 | 0.006 | 0.342 | 0.996       |
|                             |                                         | car   | 0.569 | 0.806 | 0.013 | 0.339 | 0.994       |
| MobileNet_SSD_Lite_v2.0     | ../predictor/_fixtures/lane_control.jpg | car   | 0.581 | 0.992 | 0.612 | 0.992 | 0.999       |
|                             |                                         | car   | 0.585 | 0.800 | 0.020 | 0.331 | 0.999       |
|                             |                                         | car   | 0.584 | 0.806 | 0.011 | 0.337 | 0.998       |
|                             |                                         | car   | 0.578 | 0.992 | 0.608 | 0.996 | 0.993       |
|                             |                                         | car   | 0.583 | 0.807 | 0.007 | 0.335 | 0.987       |

### Image Instance Segmentation
Run `go test -run InstanceSegmentation`

| Name                          | Image                                   | Label | Probability |
|:-----------------------------:|:---------------------------------------:|:-----:|:-----------:|
| OnnxVision_Mask_RCNN_R_50_FPN | ../predictor/_fixtures/lane_control.jpg | car   | 0.993       |

### Image Semantic Segmentation
Run `go test -run SemanticSegmentation`

| Name                            | Image                                   | label at bottom-right corner |
|:-------------------------------:|:---------------------------------------:|:----------------------------:|
| TorchVision_DeepLabv3_Resnet101 | ../predictor/_fixtures/lane_control.jpg | 7 (car)                      |
| TorchVision_FCN_Resnet101       | ../predictor/_fixtures/lane_control.jpg | 7 (car)                      |
