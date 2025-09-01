#pragma once

#include <string>
#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>


class AIManager{
    public:
        AIManager();
        ~AIManager();

        void load_model(const std::string & modelpath);

        std::string run_inference(const std::string & image_file_path);


    private:
        Ort::Env a_env;
        Ort::Session a_session;
        bool a_model_loaded;
};