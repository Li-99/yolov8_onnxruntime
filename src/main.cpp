#include <regex>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <ctime>
#include "cmdline.h"
#include "utils.h"
#include "yolov8Predictor.h"

int main(int argc, char *argv[])
{
    float confThreshold = 0.4f;
    float iouThreshold = 0.4f;

    float maskThreshold = 0.5f;

    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to onnx model.", false, "yolov8m.onnx");
    cmd.add<std::string>("image_path", 'i', "Image source to be predicted.", false, "./Imginput");
    cmd.add<std::string>("out_path", 'o', "Path to save result.", false, "./Imgoutput");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", false, "coco.names");

    cmd.add<std::string>("suffix_name", 'x', "Suffix names.", false, "yolov8m");

    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string imagePath = cmd.get<std::string>("image_path");
    const std::string savePath = cmd.get<std::string>("out_path");
    const std::string suffixName = cmd.get<std::string>("suffix_name");
    const std::string modelPath = cmd.get<std::string>("model_path");

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }
    if (!std::filesystem::exists(modelPath))
    {
        std::cerr << "Error: There is no model." << std::endl;
        return -1;
    }
    if (!std::filesystem::is_directory(imagePath))
    {
        std::cerr << "Error: There is no model." << std::endl;
        return -1;
    }
    if (!std::filesystem::is_directory(savePath))
    {
        std::filesystem::create_directory(savePath);
    }
    std::cout << "Model from :::" << modelPath << std::endl;
    std::cout << "Images from :::" << imagePath << std::endl;
    std::cout << "Resluts will be saved :::" << savePath << std::endl;

    YOLOPredictor predictor{nullptr};
    try
    {
        predictor = YOLOPredictor(modelPath, isGPU,
                                  confThreshold,
                                  iouThreshold,
                                  maskThreshold);
        std::cout << "Model was initialized." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    assert(classNames.size() == predictor.classNums);
    std::regex pattern(".+\\.(jpg|jpeg|png|gif)$");
    std::cout << "Start predicting..." << std::endl;

    clock_t startTime, endTime;
    startTime = clock();

    int picNums = 0;

    for (const auto &entry : std::filesystem::directory_iterator(imagePath))
    {
        if (std::filesystem::is_regular_file(entry.path()) && std::regex_match(entry.path().filename().string(), pattern))
        {
            picNums += 1;
            std::string Filename = entry.path().string();
            std::string baseName = std::filesystem::path(Filename).filename().string();
            std::cout << Filename << " predicting..." << std::endl;

            cv::Mat image = cv::imread(Filename);
            std::vector<Yolov8Result> result = predictor.predict(image);
            utils::visualizeDetection(image, result, classNames);

            std::string newFilename = baseName.substr(0, baseName.find_last_of('.')) + "_" + suffixName + baseName.substr(baseName.find_last_of('.'));
            std::string outputFilename = savePath + "/" + newFilename;
            cv::imwrite(outputFilename, image);
            std::cout << outputFilename << " Saved !!!" << std::endl;
        }
    }
    endTime = clock();
    std::cout << "The total run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "seconds" << std::endl;
    std::cout << "The average run time is: " << (double)(endTime - startTime) / picNums / CLOCKS_PER_SEC << "seconds" << std::endl;

    std::cout << "##########DONE################" << std::endl;

    return 0;
}
