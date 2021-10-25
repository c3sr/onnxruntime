package onnxruntime

import (
	"github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework"
)

var FrameworkManifest = dlframework.FrameworkManifest{
	Name:    "Onnxruntime",
	Version: "1.7.1",
}

func Register() {
	err := framework.Register(FrameworkManifest)
	if err != nil {
		log.WithError(err).Error("Failed to register server")
	}
}
