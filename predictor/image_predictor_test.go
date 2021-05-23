package predictor

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path/filepath"
	"testing"

	"github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework/options"
	raiimage "github.com/c3sr/image"
	"github.com/c3sr/image/types"
	nvidiasmi "github.com/c3sr/nvidia-smi"
	"github.com/c3sr/onnxruntime"
	"github.com/k0kubun/pp/v3"
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
	model, err := onnxruntime.FrameworkManifest.FindModel("MLPerf_Mobilenet_v1:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	batchSize := 10
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

	input := make([]gotensor.Tensor, batchSize)
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

	joined, err := gotensor.Concat(0, input[0], input[1:]...)
	if err != nil {
		return
	}
	joined.Reshape(append([]int{len(input)}, input[0].Shape()...)...)

	err = predictor.Predict(ctx, []gotensor.Tensor{joined})

	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	for i := 0; i < batchSize; i++ {
		pp.Println("Index: ", pred[i][0].GetClassification().GetIndex())
		pp.Println("Label: ", pred[i][0].GetClassification().GetLabel())
		pp.Println("Probability: ", pred[i][0].GetProbability())
	}

	// The score not applied softmax for torchvision alexnet
	// assert.InDelta(t, float32(0.702554), pred[0][0].GetProbability(), 0.001)
	// assert.Equal(t, int32(103), pred[0][0].GetClassification().GetIndex())
}

func TestObjectDetection(t *testing.T) {
	onnxruntime.Register()
	model, err := onnxruntime.FrameworkManifest.FindModel("MLPerf_SSD_ResNet34_1200x1200:1.0")
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

	input := make([]gotensor.Tensor, batchSize)
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

	joined, err := gotensor.Concat(0, input[0], input[1:]...)
	if err != nil {
		return
	}
	joined.Reshape(append([]int{len(input)}, input[0].Shape()...)...)

	err = predictor.Predict(ctx, []gotensor.Tensor{joined})

	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	for ii, cnt := 0, 0; ii < len(pred[0]) && cnt < 3; ii++ {
		//	if pred[0][ii].GetProbability() >= 0.5 {
		cnt++
		fmt.Printf("|                             | ./_fixtures/lane_control.jpg           | %s   | %.3f | %.3f | %.3f | %.3f | %.3f       |\n",
			pred[0][ii].GetBoundingBox().GetLabel(),
			pred[0][ii].GetBoundingBox().GetXmin(),
			pred[0][ii].GetBoundingBox().GetXmax(),
			pred[0][ii].GetBoundingBox().GetYmin(),
			pred[0][ii].GetBoundingBox().GetYmax(),
			pred[0][ii].GetProbability())
		//	}
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

	input := make([]gotensor.Tensor, batchSize)
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

	joined, err := gotensor.Concat(0, input[0], input[1:]...)
	if err != nil {
		return
	}
	joined.Reshape(append([]int{len(input)}, input[0].Shape()...)...)

	err = predictor.Predict(ctx, []gotensor.Tensor{joined})

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

	input := make([]gotensor.Tensor, batchSize)
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

	joined, err := gotensor.Concat(0, input[0], input[1:]...)
	if err != nil {
		return
	}
	joined.Reshape(append([]int{len(input)}, input[0].Shape()...)...)

	err = predictor.Predict(ctx, []gotensor.Tensor{joined})

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

func TestImageEnhancement(t *testing.T) {
	onnxruntime.Register()
	model, err := onnxruntime.FrameworkManifest.FindModel("SRGAN:1.0")
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

	predictor, err := NewImageEnhancementPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "penguin.png")
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

	channels := 3
	height := img.Bounds().Dy()
	width := img.Bounds().Dx()

	input := make([]gotensor.Tensor, batchSize)
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

	joined, err := gotensor.Concat(0, input[0], input[1:]...)
	if err != nil {
		return
	}
	joined.Reshape(append([]int{len(input)}, input[0].Shape()...)...)

	err = predictor.Predict(ctx, []gotensor.Tensor{joined})

	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		panic(err)
	}

	f, ok := pred[0][0].Feature.(*dlframework.Feature_RawImage)
	if !ok {
		panic("expecting an image feature")
	}

	fl := f.RawImage.GetFloatList()
	outWidth := f.RawImage.GetWidth()
	outHeight := f.RawImage.GetHeight()
	offset := 0
	outImg := types.NewRGBImage(image.Rect(0, 0, int(outWidth), int(outHeight)))
	for h := 0; h < int(outHeight); h++ {
		for w := 0; w < int(outWidth); w++ {
			R := uint8(fl[offset+0])
			G := uint8(fl[offset+1])
			B := uint8(fl[offset+2])
			outImg.Set(w, h, color.RGBA{R, G, B, 255})
			offset += 3
		}
	}

	if true {
		output, err := os.Create("output.jpg")
		if err != nil {
			panic(err)
		}
		defer output.Close()
		err = jpeg.Encode(output, outImg, nil)
		if err != nil {
			panic(err)
		}
	}

	assert.Equal(t, int32(1356), outHeight)
	assert.Equal(t, int32(2040), outWidth)
	assert.Equal(t, types.RGB{
		R: 0xc2,
		G: 0xc2,
		B: 0xc6,
	}, outImg.At(0, 0))
}
