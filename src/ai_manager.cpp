#include "include/ai_manager.hpp"


AIManager::AIManager()
: a_env(ORT_LOGGING_LEVEL_WARNING, "AIvoice"),
  a_session(nullptr)
{}

AIManager::~AIManager(){

}

void AIManager::load_model(const std::string & model_path){
    if(a_model_loaded){
        std::cout << "model already loaded." << std::endl;
        return;
    }

    //create session options, and you can set the thread number;
    Ort::SessionOptions session_options;

    //create session
    a_session = Ort::Session(a_env, model_path.c_str(), session_options);
    a_model_loaded = true;
    std::cout << "Model loaded successfully from " << model_path << std::endl;
}


std::string AIManager::run_inference(const std::string & image_file_path){
    if(!a_model_loaded){
        return "Error: Model not loaded.\n";
    }
    //1. read the image
    //2. change the image to onnx's tensor
    //3. run the a_session
    //4. parse the output tensor
    //5. change the tensor to string and return
    return "Inference result: placeholder";
}