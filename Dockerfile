FROM c3sr/go-onnxruntime:amd64-cpu-onnxruntime1.7.1-latest



RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && python3 get-pip.py && pip3 install opencv-python==4.2.0.32 && pip3 install numpy && pip3 install torchvision && pip3 install scipy


# Make our go directory structure
RUN  mkdir -p /go/src/github.com/c3sr/onnxruntime
WORKDIR /go/src/github.com/c3sr/onnxruntime

# Handle Go Depedencies and cache it as a layer
COPY go.* ./

# Enable these lines if you want to work with an inline version of dlframework or dldataset and update the root go.mod
#COPY dlframework/go.mod dlframework/
#COPY dldataset/go.mod dldataset/
RUN go mod download

# Get the rest of the project in
COPY . .

RUN cd onnxruntime-agent && \
    go install -a -tags=nogpu -installsuffix cgo

CMD ["/go/bin/onnxruntime-agent", "worker"]
