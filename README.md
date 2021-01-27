# MLModelScope Onnxruntime Agent

[![Build Status](https://dev.azure.com/yhchang/c3sr/_apis/build/status/c3sr.onnxruntime?branchName=master)](https://dev.azure.com/yhchang/c3sr/_build/latest?definitionId=8&branchName=master)

This is the Onnxruntime agent for [MLModelScope](mlmodelscope.org), an open-source framework and hardware agnostic, extensible and customizable platform for evaluating and profiling ML models across datasets / frameworks / systems, and within AI application pipelines.

Check out [MLModelScope](mlmodelscope.org) and welcome to contribute.

# Bare Minimum Installation

## Prerequsite System Library Installation
We first discuss a bare minimum onnxruntime-agent installation without the tracing and profiling capabilities. To make this work, you will need to have the following system libraries preinstalled in your system.

- The CUDA library (required)
- The CUPTI library (required)
- The Onnxruntime C++ library (required)
- The libjpeg-turbo library (optional, but preferred)

### The CUDA Library

Please refer to Nvidia CUDA library installation on this. Find the localation of your local CUDA installation, which is typically at `/usr/local/cuda/`, and setup the path to the `libcublas.so` library. Place the following in either your `~/.bashrc` or `~/.zshrc` file:

```
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

### The CUPTI Library

Please refer to Nvidia CUPTI library installation on this. Find the localation of your local CUPTI installation, which is typically at `/usr/local/cuda/extras/CUPTI`, and setup the path to the `libcupti.so` library. Place the following in either your `~/.bashrc` or `~/.zshrc` file:

```
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```

### The Onnxruntime C++ library
Refer to [github.com/c3sr/go-onnxruntime](https://github.com/c3sr/go-onnxruntime#onnxruntime-c-library)

### Use libjpeg-turbo for Image Preprocessing

[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) is a JPEG image codec that uses SIMD instructions (MMX, SSE2, AVX2, NEON, AltiVec) to accelerate baseline JPEG compression and decompression. It outperforms libjpeg by a significant amount.

You need libjpeg installed.
```
sudo apt-get install libjpeg-dev
```
The default is to use libjpeg-turbo, to opt-out, use build tag `nolibjpeg`.

To install libjpeg-turbo, refer to [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases).

Linux

```
  export TURBO_VER=2.0.2
  cd /tmp
  wget https://cfhcable.dl.sourceforge.net/project/libjpeg-turbo/${TURBO_VER}/libjpeg-turbo-official_${TURBO_VER}_amd64.deb
  sudo dpkg -i libjpeg-turbo-official_${TURBO_VER}_amd64.deb
```

macOS

```
brew install jpeg-turbo
```


## Installation of GO for Compilation

Since we use `go` for MLModelScope development, it's required to have `go` installed in your system before proceeding.

Please follow [Installing Go Compiler](https://github.com/c3sr/rai/blob/master/docs/developer_guide.md) to have `go` installed.

## Bare Minimum Onnxruntime-agent Installation

Download and install the MLModelScope Onnxruntime Agent by running the following command in any location, assuming you have installed `go` following the above instruction.

```
go get -v github.com/c3sr/onnxruntime
```

You can then install the dependency packages through `go get` [Dep](https://github.com/golang/dep).

```
dep ensure -v
```

This installs the dependency in `vendor/`.

The CGO interface passes go pointers to the C API. There is an error in the CGO runtime. We can disable the error by placing

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`


Build the Onnxruntime agent with GPU enabled
```
cd $GOPATH/src/github.com/c3sr/onnxruntime/onnxruntime-agent
go build
```

Build the Onnxruntime agent without GPU or libjpeg-turbo
```
cd $GOPATH/src/github.com/c3sr/onnxruntime/onnxruntime-agent
go build -tags="nogpu nolibjpeg"
```

If everything is successful, you should have an executable `onnxruntime-agent` binary in the current directory.

### Configuration Setup

To run the agent, you need to setup the correct configuration file for the agent. Some of the information may not make perfect sense for all testing scenarios, but they are required and will be needed for later stage testing. Some of the port numbers as specified below can be changed depending on your later setup for those service.

So let's just set them up as is, and worry about the detailed configuration parameter values later.

You must have a `carml` config file called `.carml_config.yml` under your home directory. An example config file `carml_config.yml.example` is in [github.com/c3sr/MLModelScope](https://github.com/c3sr/MLModelScope) . You can move it to `~/.carml_config.yml`.

The following configuration file can be placed in `$HOME/.carml_config.yml` or can be specified via the `--config="path"` option.

```yaml
app:
  name: carml
  debug: true
  verbose: true
  tempdir: ~/data/carml
registry:
  provider: consul
  endpoints:
    - localhost:8500
  timeout: 20s
  serializer: jsonpb
database:
  provider: mongodb
  endpoints:
    - localhost
tracer:
  enabled: true
  provider: jaeger
  endpoints:
    - localhost:9411
  level: FULL_TRACE
logger:
  hooks:
    - syslog
```

## Test Installation

With the configuration and the above bare minimumn installation, you should be ready to test the installation and see how things works.

Here are a few examples. First, make sure we are in the right location
```
cd $GOPATH/src/github.com/c3sr/onnxruntime/onnxruntime-agent
```

To see a list of help
```
./onnxruntime-agent -h
```

To see a list of models that we can run with this agent
```
./onnxruntime-agent info models
```

To run an inference using the default DNN model `alexnet` with a default input image.

```
./onnxruntime-agent predict urls --model_name TorchVision_Alexnet --profile=false --publish=false
```

The above `--profile=false --publish=false` command parameters tell the agent that we do not want to use profiling capability and publish the results, as we haven't installed the MongoDB database to store profiling data and the tracer service to accept tracing information.

# External Service Installation to Enable Tracing and Profiling

We now discuss how to install a few external services that make the agent fully useful in terms of collecting tracing and profiling data.

## External Srvices

MLModelScope relies on a few external services. These services provide tracing, registry, and database servers.

These services can be installed and enabled in different ways. We discuss how we use `docker` below to show how this can be done. You can also not use `docker` but install those services from either binaries or source codes directly.

### Installing Docker

Refer to [Install Docker](https://docs.docker.com/install/).

On Ubuntu, an easy way is using

```
curl -fsSL get.docker.com -o get-docker.sh | sudo sh
sudo usermod -aG docker $USER
```

On macOS, [intsall Docker Destop](https://docs.docker.com/docker-for-mac/install/)


### Starting Trace Server

This service is required.

- On x86 (e.g. intel) machines, start [jaeger](http://jaeger.readthedocs.io/en/latest/getting_started/) by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:latest
```

- On ppc64le (e.g. minsky) machines, start [jaeger](http://jaeger.readthedocs.io/en/latest/getting_started/) machine by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 carml/jaeger:ppc64le-latest
```

The trace server runs on http://localhost:16686

### Starting Registry Server

This service is not required if using onnxruntime-agent for local evaluation.

- On x86 (e.g. intel) machines, start [consul](https://hub.docker.com/_/consul/) by

```
docker run -p 8500:8500 -p 8600:8600 -d consul
```

- On ppc64le (e.g. minsky) machines, start [consul](https://hub.docker.com/_/consul/) by

```
docker run -p 8500:8500 -p 8600:8600 -d carml/consul:ppc64le-latest
```

The registry server runs on http://localhost:8500

### Starting Database Server

This service is not required if not using database to publish evaluation results.

- On x86 (e.g. intel) machines, start [mongodb](https://hub.docker.com/_/mongo/) by

```
docker run -p 27017:27017 --restart always -d mongo:3.0
```

You can also mount the database volume to a local directory using

```
docker run -p 27017:27017 --restart always -d  -v $HOME/data/carml/mongo:/data/db mongo:3.0
```

### Configuration

You must have a `carml` config file called `.carml_config.yml` under your home directory. An example config file `~/.carml_config.yml` is already discussed above. Please update the port numbers for the above external services accordingly if you decide to choose a different ports above.


# Use the Agent with the [MLModelScope Web UI](https://github.com/c3sr/mlmodelscope)

```
./onnxruntime-agent serve -l -d -v
```

Refer to [here](https://docs.mlmodelscope.org/installation/webserver/) to run the web UI to interact with the agent.

# Use the Agent through Command Line

Run ```./onnxruntime-agent -h``` to list the available commands.

Run ```./onnxruntime-agent info models``` to list the available models.

Run ```./onnxruntime-agent predict``` to evaluate a model. This runs the default evaluation.

Run ```./onnxruntime-agent predict -h``` shows the available flags you can set.

An example run is

```
./onnxruntime-agent predict urls --model_name TorchVision_Alexnet --profile=false --publish=false
```

# Use the Agent through Pre-built Docker Images

We have [pre-built docker images](https://hub.docker.com/r/c3sr/onnxruntime-agent/tags) on Dockerhub. The images are `c3sr/onnxruntime-agent:amd64-cpu-latest` and `c3sr/onnxruntime-agent:amd64-gpu-latest`. The entrypoint is set as `onnxruntime-agent` thus these images act similar as the command line above.

An example run is

```
docker run --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --privileged=true \
    --network host \
    -v ~/.carml_config.yml:/root/.carml_config.yml \
    -v ~/results:/go/src/github.com/c3sr/onnxruntime/results \
    c3sr/onnxruntime-agent:amd64-gpu-latest predict urls --model_name TorchVision_Alexnet --profile=false --publish=false
```
NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be insufficient for Onnxruntime.  NVIDIA recommends the use of the following flags:
   ```--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 ...```

NOTE: To run with GPU, you need to meet following requirements:

- Docker >= 19.03 with nvidia-container-toolkit (otherwise need to use nvidia-docker)
- CUDA >= 10.1
- NVIDIA Driver >= 418.39

