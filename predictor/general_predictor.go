package predictor

import (
	"context"

	"github.com/c3sr/config"
	"github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework/agent"
	"github.com/c3sr/dlframework/framework/options"
	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/downloadmanager"
	goonnxruntime "github.com/c3sr/go-onnxruntime"
	"github.com/c3sr/onnxruntime"
	"github.com/c3sr/tracer"
	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	gotensor "gorgonia.org/tensor"
)

// GeneralPredictor ...
type GeneralPredictor struct {
	common.Base
	predictor *goonnxruntime.Predictor
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

// Download ...
func (p *GeneralPredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	gp := &GeneralPredictor{
		Base: common.Base{
			Framework: framework,
			Model:     model,
			WorkDir:   workDir,
			Options:   options.New(opts...),
		},
	}

	if err = gp.download(ctx); err != nil {
		return err
	}

	return nil
}

// Load ...
func (p *GeneralPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	gp := &GeneralPredictor{
		Base: common.Base{
			Framework: framework,
			Model:     model,
			WorkDir:   workDir,
			Options:   options.New(opts...),
		},
	}

	if err = gp.download(ctx); err != nil {
		return nil, err
	}

	if err = gp.loadPredictor(ctx); err != nil {
		return nil, err
	}

	return gp, nil
}

func (p *GeneralPredictor) download(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(
		ctx,
		tracer.APPLICATION_TRACE,
		"download",
		opentracing.Tags{
			"graph_url":         p.GetGraphUrl(),
			"target_graph_file": p.GetGraphPath(),
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

	return nil
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

// ReadPredictedFeatures ...
func (p *GeneralPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	return nil, errors.New("Not Implemented.")
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
	return dlframework.GeneralModality, nil
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
