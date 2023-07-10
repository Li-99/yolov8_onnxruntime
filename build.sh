rm -r build
rm -r Imgoutput
mkdir build
cd build
cmake .. -DONNXRUNTIME_DIR=path/to/onnxruntime -DCMAKE_BUILD_TYPE=Debug
make
cd ..
sh run.sh