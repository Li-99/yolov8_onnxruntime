./build/yolov8_ort -m ./models/yolov8m.onnx -i ./Imginput -o ./Imgoutput -c ./models/coco.names -x m --gpu
./build/yolov8_ort -m ./models/yolov8m-seg.onnx -i ./Imginput -o ./Imgoutput -c ./models/coco.names -x ms --gpu

