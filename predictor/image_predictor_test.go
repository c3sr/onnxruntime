package predictor

import (
	"context"
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/k0kubun/pp"
	"github.com/rai-project/dlframework/framework/options"
	raiimage "github.com/rai-project/image"
	"github.com/rai-project/image/types"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/onnxruntime"
	"github.com/stretchr/testify/assert"
	gotensor "gorgonia.org/tensor"
)

func normalizeImageHWC(in0 image.Image, mean []float32, scale []float32) ([]float32, error) {
	height := in0.Bounds().Dy()
	width := in0.Bounds().Dx()
	out := make([]float32, 3*height*width)
	switch in := in0.(type) {
	case *types.RGBImage:
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*in.Stride + x*3
				rgb := in.Pix[offset : offset+3]
				r, g, b := rgb[0], rgb[1], rgb[2]
				out[offset+0] = (float32(r) - mean[0]) / scale[0]
				out[offset+1] = (float32(g) - mean[1]) / scale[1]
				out[offset+2] = (float32(b) - mean[2]) / scale[2]
			}
		}
	case *types.BGRImage:
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*in.Stride + x*3
				bgr := in.Pix[offset : offset+3]
				b, g, r := bgr[0], bgr[1], bgr[2]
				out[offset+0] = (float32(b) - mean[0]) / scale[0]
				out[offset+1] = (float32(g) - mean[1]) / scale[1]
				out[offset+2] = (float32(r) - mean[2]) / scale[2]
			}
		}
	default:
		panic("unreachable")
	}

	return out, nil
}

func normalizeImageCHW(in0 image.Image, mean []float32, scale []float32) ([]float32, error) {
	height := in0.Bounds().Dy()
	width := in0.Bounds().Dx()
	out := make([]float32, 3*height*width)
	switch in := in0.(type) {
	case *types.RGBImage:
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*in.Stride + x*3
				rgb := in.Pix[offset : offset+3]
				r, g, b := rgb[0], rgb[1], rgb[2]
				out[0*width*height+y*width+x] = (float32(r) - mean[0]) / scale[0]
				out[1*width*height+y*width+x] = (float32(g) - mean[1]) / scale[1]
				out[2*width*height+y*width+x] = (float32(b) - mean[2]) / scale[2]
			}
		}
	case *types.BGRImage:
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*in.Stride + x*3
				rgb := in.Pix[offset : offset+3]
				r, g, b := rgb[0], rgb[1], rgb[2]
				out[0*width*height+y*width+x] = (float32(b) - mean[0]) / scale[0]
				out[1*width*height+y*width+x] = (float32(g) - mean[1]) / scale[1]
				out[2*width*height+y*width+x] = (float32(r) - mean[2]) / scale[2]
			}
		}
	default:
		panic("unreachable")
	}
	return out, nil
}

func TestPredictorNew(t *testing.T) {
	onnxruntime.Register()
	model, err := onnxruntime.FrameworkManifest.FindModel("TorchVision_Alexnet:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	predictor, err := NewImageClassificationPredictor(*model)
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)

	defer predictor.Close()

	_, ok := predictor.(*ImageClassificationPredictor)
	assert.True(t, ok)
}

func TestImageClassification(t *testing.T) {
	onnxruntime.Register()
	model, err := onnxruntime.FrameworkManifest.FindModel("TorchVision_VGG_19_BN:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	batchSize := 1
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := NewImageClassificationPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "platypus.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	preprocessOpts, err := predictor.GetPreprocessOptions()
	assert.NoError(t, err)
	channels := preprocessOpts.Dims[0]
	height := preprocessOpts.Dims[1]
	width := preprocessOpts.Dims[2]
	mode := preprocessOpts.ColorMode

	var imgOpts []raiimage.Option
	if mode == types.RGBMode {
		imgOpts = append(imgOpts, raiimage.Mode(types.RGBMode))
	} else {
		imgOpts = append(imgOpts, raiimage.Mode(types.BGRMode))
	}

	img, err := raiimage.Read(r, imgOpts...)
	if err != nil {
		panic(err)
	}

	imgOpts = append(imgOpts, raiimage.Resized(height, width))
	imgOpts = append(imgOpts, raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear))
	resized, err := raiimage.Resize(img, imgOpts...)

	input := make([]*gotensor.Dense, batchSize)
	imgFloats, err := normalizeImageCHW(resized, preprocessOpts.MeanImage, preprocessOpts.Scale)
	if err != nil {
		panic(err)
	}

	for i := 0; i < batchSize; i++ {
		input[i] = gotensor.New(
			gotensor.WithShape(channels, height, width),
			gotensor.WithBacking(imgFloats),
		)
	}

	err = predictor.Predict(ctx, input)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pp.Println("Index: ", pred[0][0].GetClassification().GetIndex())
	pp.Println("Label: ", pred[0][0].GetClassification().GetLabel())
	pp.Println("Probability: ", pred[0][0].GetProbability())

	// The score not applied softmax for torchvision alexnet
	// assert.InDelta(t, float32(15.774), pred[0][0].GetProbability(), 0.001)
	// assert.Equal(t, int32(103), pred[0][0].GetClassification().GetIndex())
}

func TestObjectDetection(t *testing.T) {
	onnxruntime.Register()
	model, err := onnxruntime.FrameworkManifest.FindModel("MobileNet_SSD_Lite_v2.0:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	batchSize := 1
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := NewObjectDetectionPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "lane_control.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	preprocessOpts, err := predictor.GetPreprocessOptions()
	assert.NoError(t, err)
	channels := preprocessOpts.Dims[0]
	height := preprocessOpts.Dims[1]
	width := preprocessOpts.Dims[2]
	mode := preprocessOpts.ColorMode

	var imgOpts []raiimage.Option
	if mode == types.RGBMode {
		imgOpts = append(imgOpts, raiimage.Mode(types.RGBMode))
	} else {
		imgOpts = append(imgOpts, raiimage.Mode(types.BGRMode))
	}

	img, err := raiimage.Read(r, imgOpts...)
	if err != nil {
		panic(err)
	}

	imgOpts = append(imgOpts, raiimage.Resized(height, width))
	imgOpts = append(imgOpts, raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear))
	resized, err := raiimage.Resize(img, imgOpts...)

	input := make([]*gotensor.Dense, batchSize)
	imgFloats, err := normalizeImageCHW(resized, preprocessOpts.MeanImage, preprocessOpts.Scale)
	if err != nil {
		panic(err)
	}

	for i := 0; i < batchSize; i++ {
		input[i] = gotensor.New(
			gotensor.WithShape(channels, height, width),
			gotensor.WithBacking(imgFloats),
		)
	}

	err = predictor.Predict(ctx, input)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	for i , cnt := 0, 0; i < len(pred[0]) && cnt < 5; i++ {
		// skip background
		if pred[0][i].GetBoundingBox().GetIndex() != 0 {
			cnt++
			pp.Println("Label: ", pred[0][i].GetBoundingBox().GetLabel())
			pp.Println("Probability: ", pred[0][i].GetProbability())
			pp.Println("Xmax: ", pred[0][i].GetBoundingBox().GetXmax())
			pp.Println("Xmin: ", pred[0][i].GetBoundingBox().GetXmin())
			pp.Println("Ymax: ", pred[0][i].GetBoundingBox().GetYmax())
			pp.Println("Ymin: ", pred[0][i].GetBoundingBox().GetYmin())
		}
	}
}

func TestInstanceSegmentation(t *testing.T) {
	onnxruntime.Register()
	model, err := onnxruntime.FrameworkManifest.FindModel("OnnxVision_Mask_RCNN_R_50_FPN:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	batchSize := 1
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := NewInstanceSegmentationPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "lane_control.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	preprocessOpts, err := predictor.GetPreprocessOptions()
	assert.NoError(t, err)
	channels := preprocessOpts.Dims[0]
	height := preprocessOpts.Dims[1]
	width := preprocessOpts.Dims[2]
	mode := preprocessOpts.ColorMode

	var imgOpts []raiimage.Option
	if mode == types.RGBMode {
		imgOpts = append(imgOpts, raiimage.Mode(types.RGBMode))
	} else {
		imgOpts = append(imgOpts, raiimage.Mode(types.BGRMode))
	}

	img, err := raiimage.Read(r, imgOpts...)
	if err != nil {
		panic(err)
	}

	imgOpts = append(imgOpts, raiimage.Resized(height, width))
	imgOpts = append(imgOpts, raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear))
	resized, err := raiimage.Resize(img, imgOpts...)

	input := make([]*gotensor.Dense, batchSize)
	imgFloats, err := normalizeImageCHW(resized, preprocessOpts.MeanImage, preprocessOpts.Scale)
	if err != nil {
		panic(err)
	}

	for i := 0; i < batchSize; i++ {
		input[i] = gotensor.New(
			gotensor.WithShape(channels, height, width),
			gotensor.WithBacking(imgFloats),
		)
	}

	err = predictor.Predict(ctx, input)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pp.Println("Index: ", pred[0][0].GetInstanceSegment().GetIndex())
	pp.Println("Label: ", pred[0][0].GetInstanceSegment().GetLabel())
	pp.Println("Probability: ", pred[0][0].GetProbability())
}

func TestSemanticSegmentation(t *testing.T) {
	onnxruntime.Register()
	model, err := onnxruntime.FrameworkManifest.FindModel("TorchVision_FCN_Resnet101:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	batchSize := 1
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := NewSemanticSegmentationPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "lane_control.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	preprocessOpts, err := predictor.GetPreprocessOptions()
	assert.NoError(t, err)
	mode := preprocessOpts.ColorMode

	var imgOpts []raiimage.Option
	if mode == types.RGBMode {
		imgOpts = append(imgOpts, raiimage.Mode(types.RGBMode))
	} else {
		imgOpts = append(imgOpts, raiimage.Mode(types.BGRMode))
	}

	img, err := raiimage.Read(r, imgOpts...)
	if err != nil {
		panic(err)
	}

	height := img.Bounds().Dy()
	width := img.Bounds().Dx()
	channels := 3

	input := make([]*gotensor.Dense, batchSize)
	imgFloats, err := normalizeImageCHW(img, preprocessOpts.MeanImage, preprocessOpts.Scale)
	if err != nil {
		panic(err)
	}

	for i := 0; i < batchSize; i++ {
		input[i] = gotensor.New(
			gotensor.WithShape(channels, height, width),
			gotensor.WithBacking(imgFloats),
		)
	}

	err = predictor.Predict(ctx, input)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	sseg := pred[0][0].GetSemanticSegment()
	intMask := sseg.GetIntMask()

	assert.Equal(t, int32(7), intMask[247039])
}