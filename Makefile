all: generate

install-deps:
	go get github.com/jteeuwen/go-bindata/...

generate: clean generate-models

generate-models:
	go-bindata -nomemcopy -prefix builtin_models/ -pkg onnxruntime -o builtin_models_static.go -ignore=.DS_Store  -ignore=README.md builtin_models/...

clean-models:
	rm -fr builtin_models_static.go

clean: clean-models