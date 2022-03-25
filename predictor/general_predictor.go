package predictor

import (
	"context"

	"github.com/c3sr/config"
	"github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework/agent"
	"github.com/c3sr/dlframework/framework/options"
	common "github.com/c3sr/dlframework/framework/predictor"
	goonnxruntime "github.com/c3sr/go-onnxruntime"
	"github.com/c3sr/onnxruntime"
	"github.com/c3sr/tracer"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	gotensor "gorgonia.org/tensor"
)

// GeneralPredictor ...
type GeneralPredictor struct {
	common.Base
	predictor *goonnxruntime.Predictor
	desiredModality *dlframework.Modality
}

// NewGeneralPredictor ...
func NewGeneralPredictor(model dlframework.ModelManifest, os ...options.Option) (common.Predictor, error) {
	opts := options.New(os...)
	ctx := opts.Context()

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()

	predictor := new(GeneralPredictor)

	return predictor.Load(ctx, model, os...)
}

// Load ...
func (p *GeneralPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	gp := &GeneralPredictor{
		Base: common.Base{
			Framework: *model.Framework,
			Model:     model,
			WorkDir:   workDir,
			Options:   options.New(opts...),
		},
	}

	if err = gp.loadPredictor(ctx); err != nil {
		return nil, err
	}

	return gp, nil
}

func (p *GeneralPredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
	defer span.Finish()

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
func (p *GeneralPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
	if data == nil {
		return errors.New("input data nil")
	}

	gotensors, ok := data.([]gotensor.Tensor)
	if !ok {
		return errors.New("input data is not slice of dense tensors")
	}

	return p.predictor.Predict(ctx, gotensors)
}

// ReadPredictedFeaturesAsMap ...
func (p *GeneralPredictor) ReadPredictedFeaturesAsMap(ctx context.Context) (map[string]interface{}, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features_as_map")
	defer span.Finish()

	outputs, err := p.predictor.ReadPredictionOutput(ctx)
	if err != nil {
		return nil, err
	}

	res := make(map[string]interface{})
	res["outputs"] = outputs

	if labels, err := p.GetLabels(); err == nil {
		res["labels"] = labels
	}

	return res, nil
}

// Reset ...
func (p *GeneralPredictor) Reset(ctx context.Context) error {
	return nil
}

// Close ...
func (p *GeneralPredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Close()
	}

	return nil
}

// Modality ...
func (p *GeneralPredictor) Modality() (dlframework.Modality, error) {
  if p.desiredModality != nil {
    return *p.desiredModality, nil
  }
	return dlframework.GeneralModality, nil
}

// This allows postprocess to use different output formats, however, the model has to output in
// a format that the desired modality postprocess can handle
func (p *GeneralPredictor) SetDesiredOutput(modality dlframework.Modality) {
  p.desiredModality = &modality
}

func init() {
	config.AfterInit(func() {
		framework := onnxruntime.FrameworkManifest
		agent.AddPredictor(framework, &GeneralPredictor{
			Base: common.Base{
				Framework: framework,
			},
		})
	})
}
