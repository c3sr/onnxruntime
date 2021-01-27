package predictor

import (
	"os"
	"testing"

	"github.com/c3sr/config"
	_ "github.com/c3sr/tracer/jaeger"
)

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("carml"),
		config.DebugMode(true),
		config.VerboseMode(true),
	)
	os.Exit(m.Run())
}
