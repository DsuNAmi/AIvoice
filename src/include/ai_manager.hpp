#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <onnxruntime/onnxruntime_cxx_api.h>


class AIManager{
    public:
        AIManager();
        ~AIManager();

        void load_image_model(const std::string & modelpath);

        std::string run_image_inference(const std::string & image_file_path);

        void load_audio_model(const std::string & encoder_path, const std::string & decoder_path);
        std::string transcribe_audio(const std::string & audio_file_path);


    private:

        void load_labels(const std::string & label_path);

        Ort::Env a_env;
        Ort::Session a_session;
        Ort::Session a_encoder_session;
        Ort::Session a_decoder_session;

        bool a_image_model_loaded;
        bool a_audio_model_loaded;


        const int a_sample_rate;
        const int a_channles;

        std::vector<std::string> a_labels;
};