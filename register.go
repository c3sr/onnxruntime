package onnxruntime

import (
	"os"

	assetfs "github.com/elazarl/go-bindata-assetfs"
	"github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework"
)

var FrameworkManifest = dlframework.FrameworkManifest{
	Name:    "Onnxruntime",
	Version: "1.3.1",
	Container: map[string]*dlframework.ContainerHardware{
		"amd64": {
			Cpu: "raiproject/carml-onnxruntime:amd64-cpu",
			Gpu: "raiproject/carml-onnxruntime:amd64-gpu",
		},
		"ppc64le": {
			Cpu: "raiproject/carml-onnxruntime:ppc64le-gpu",
			Gpu: "raiproject/carml-onnxruntime:ppc64le-gpu",
		},
	},
}

func assetFS() *assetfs.AssetFS {
	assetInfo := func(path string) (os.FileInfo, error) {
		return os.Stat(path)
	}
	for k := range _bintree.Children {
		return &assetfs.AssetFS{Asset: Asset, AssetDir: AssetDir, AssetInfo: assetInfo, Prefix: k}
	}
	panic("unreachable")
}

func Register() {
	err := framework.Register(FrameworkManifest, assetFS())
	if err != nil {
		log.WithError(err).Error("Failed to register server")
	}
}
