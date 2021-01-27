package predictor

import (
	"bufio"
	"context"
	"os"
	"strings"

	"github.com/k0kubun/pp"
	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/c3sr/config"
	"github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework/agent"
	"github.com/c3sr/dlframework/framework/options"
	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/downloadmanager"
	goonnxruntime "github.com/c3sr/go-onnxruntime"
	"github.com/c3sr/onnxruntime"
	"github.com/c3sr/tracer"
	gotensor "gorgonia.org/tensor"
)

// InstanceSegmentationPredictor ...
type InstanceSegmentationPredictor struct {
	common.ImagePredictor
	predictor *goonnxruntime.Predictor
	labels    []string
}

// New ...
func NewInstanceSegmentationPredictor(model dlframework.ModelManifest, os ...options.Option) (common.Predictor, error) {
	opts := options.New(os...)
	ctx := opts.Context()

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()

	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}
	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(InstanceSegmentationPredictor)

	return predictor.Load(ctx, model, os...)
}

// Download ...
func (p *InstanceSegmentationPredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	ip := &InstanceSegmentationPredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return err
	}

	return nil
}

// Load ...
func (p *InstanceSegmentationPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &InstanceSegmentationPredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return nil, err
	}

	if err = ip.loadPredictor(ctx); err != nil {
		return nil, err
	}

	return ip, nil
}

func (p *InstanceSegmentationPredictor) download(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(
		ctx,
		tracer.APPLICATION_TRACE,
		"download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"feature_url":         p.GetFeaturesUrl(),
			"target_feature_file": p.GetFeaturesPath(),
		},
	)
	defer span.Finish()

	model := p.Model
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		span.LogFields(
			olog.String("event", "download model archive"),
		)
		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx))
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
	} else {
		span.LogFields(
			olog.String("event", "download graph"),
		)
		checksum := p.GetGraphChecksum()
		if checksum != "" {
			if _, _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
				return err
			}
		} else {
			if _, _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath()); err != nil {
				return err
			}
		}
	}

	span.LogFields(
		olog.String("event", "download features"),
	)
	checksum := p.GetFeaturesChecksum()
	if checksum != "" {
		if _, _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath(), downloadmanager.MD5Sum(checksum)); err != nil {
			return err
		}
	} else {
		if _, _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath()); err != nil {
			return err
		}
	}

	return nil
}

func (p *InstanceSegmentationPredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
	defer span.Finish()

	span.LogFields(
		olog.String("event", "read features"),
	)

	var labels []string
	f, err := os.Open(p.GetFeaturesPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetFeaturesPath())
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}
	p.labels = labels

	span.LogFields(
		olog.String("event", "creating predictor"),
	)

	opts, err := p.GetPredictionOptions()
	if err != nil {
		return err
	}

	pred, err := goonnxruntime.New(
		ctx,
		options.WithOptions(opts),
		options.Graph([]byte(p.GetGraphPath())),
	)
	if err != nil {
		return err
	}

	p.predictor = pred

	return nil
}

// Predict ...
func (p *InstanceSegmentationPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
	if data == nil {
		return errors.New("input data nil")
	}

	gotensors, ok := data.([]*gotensor.Dense)
	if !ok {
		return errors.New("input data is not slice of dense tensors")
	}

	fst := gotensors[0]
	// TODO: right now the only model for instance segmentation accepts CHW without batch
	dims := fst.Shape()
	// debug
	pp.Println(dims)
	// TODO support data types other than float32
	var input []float32
	for _, t := range gotensors {
		input = append(input, t.Float32s()...)
	}

	err := p.predictor.Predict(ctx, []gotensor.Tensor{
		gotensor.New(
			gotensor.Of(gotensor.Float32),
			gotensor.WithBacking(input),
			gotensor.WithShape(dims...),
		),
	})
	if err != nil {
		return err
	}

	return nil
}

// ReadPredictedFeatures ...
func (p *InstanceSegmentationPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	// TODO: the code is for OnnxVision_Mask_RCNN_R_50_FPN right now for testing

	outputs, err := p.predictor.ReadPredictionOutput(ctx)
	if err != nil {
		return nil, err
	}

	boxes_layer_index, err := p.GetOutputLayerIndex("boxes_layer")
	if err != nil {
		return nil, err
	}
	raw_boxes := outputs[boxes_layer_index].Data().([]float32)

	nbox := outputs[boxes_layer_index].Shape()[0]

	boxes := make([][]float32, nbox)
	for nb := 0; nb < nbox; nb++ {
		boxes[nb] = raw_boxes[nb*4 : nb*4+4]
	}

	probabilities_layer_index, err := p.GetOutputLayerIndex("probabilities_layer")
	if err != nil {
		return nil, err
	}
	probabilities := outputs[probabilities_layer_index].Data().([]float32)

	masks_layer_index, err := p.GetOutputLayerIndex("masks_layer")
	if err != nil {
		return nil, err
	}
	raw_masks := outputs[masks_layer_index].Data().([]float32)
	output_height := outputs[masks_layer_index].Shape()[2]
	output_width := outputs[masks_layer_index].Shape()[3]

	masks := make([][][]float32, nbox)
	for nb := 0; nb < nbox; nb++ {
		masks[nb] = make([][]float32, output_height)
		for h := 0; h < output_height; h++ {
			masks[nb][h] = make([]float32, output_width)
		}
	}
	for nb := 0; nb < nbox; nb++ {
		for h := 0; h < output_height; h++ {
			for w := 0; w < output_width; w++ {
				masks[nb][h][w] = raw_masks[nb*output_height*output_width+h*output_width+w]
			}
		}
	}

	classes_layer_index, err := p.GetOutputLayerIndex("classes_layer")
	if err != nil {
		return nil, err
	}
	input_classes := outputs[classes_layer_index]

	// convert int64 to float32 if necessary
	var classes []float32

	if input_classes.Dtype() != gotensor.Float32 {
		raw_classes := input_classes.Data().([]int64)
		for i := 0; i < len(raw_classes); i++ {
			classes = append(classes, float32(raw_classes[i]))
		}
	} else {
		classes = input_classes.Data().([]float32)
	}

	return p.CreateInstanceSegmentFeatures(ctx, [][]float32{probabilities}, [][]float32{classes}, [][][]float32{boxes}, [][][][]float32{masks}, p.labels)
}

// Reset ...
func (p *InstanceSegmentationPredictor) Reset(ctx context.Context) error {
	return nil
}

// Close ...
func (p *InstanceSegmentationPredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Close()
	}
	return nil
}

func (p *InstanceSegmentationPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageInstanceSegmentationModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := onnxruntime.FrameworkManifest
		agent.AddPredictor(framework, &InstanceSegmentationPredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
