#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "include/stb_image_resize.h"


extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
}

#include "include/ai_manager.hpp"

AIManager::AIManager()
: a_env(ORT_LOGGING_LEVEL_WARNING, "AIvoice"),
  a_session(nullptr), a_encoder_session(nullptr), a_decoder_session(nullptr),
  a_image_model_loaded(false), a_audio_model_loaded(false),
  a_sample_rate(16000), a_channles(1)
{
    load_labels("../labels/imagenet_classes.txt");
    av_log_set_level(AV_LOG_QUIET);
}

AIManager::~AIManager(){

}


void AIManager::load_audio_model(const std::string & encoder_path, const std::string & decoder_path){
    if(a_audio_model_loaded){
        std::cout << "Audio models already loaded." << std::endl;
        return;
    }
    try
    {
        Ort::SessionOptions seesion_options;
        seesion_options.SetIntraOpNumThreads(1);

        a_encoder_session = Ort::Session(a_env, encoder_path.c_str(), seesion_options);
        a_decoder_session = Ort::Session(a_env, decoder_path.c_str(), seesion_options);


        a_audio_model_loaded = true;
        std::cout << "Audio models loaded successfully." << std::endl;
    }
    catch(const Ort::Exception & e)
    {
        std::cerr << "ONNX Runtime Error loading audio models: " << e.what() << std::endl;
        a_audio_model_loaded = false;
    }
    
}

std::string AIManager::transcribe_audio(const std::string & audio_file_path){
    if(!a_audio_model_loaded){
        return "Error: Audio models not loaded.\n";
    }

    std::cout << "Starting audio transcription on: " << audio_file_path << std::endl;

    //1.use FFmpeg decode the audio to PCM floats
    AVFormatContext * fmt_ctx = nullptr;
    AVCodecContext * codec_ctx = nullptr;
    AVPacket * pkt = av_packet_alloc();
    AVFrame * frame = av_frame_alloc();
    const AVCodec * codec = nullptr;

    //open the audio stream
    if(avformat_open_input(&fmt_ctx, audio_file_path.c_str(), nullptr, nullptr) < 0){
        return "Error: Could not open audio file.\n";
    }
    //find the audio stream
    if(avformat_find_stream_info(fmt_ctx,nullptr) < 0){
        return "Error: Could not find stream information.\n";
    }

    int audio_stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1,&codec_ctx,0);
    if(audio_stream_idx < 0){
        return "Error: Could not find audio stream.\n";
    }

    //open the decoder
    

    //2.convert PCM to Mel Spectrogram
    //3.convert Mel to tensor
    //4.get the encoder's output
    //5.cal
    //6.decode round by round, and get a token
    //7.change the token to str


    return "This is a placeholder for the transcribed text.\n";
}

void AIManager::load_labels(const std::string & label_path){
    std::ifstream outfile(label_path);
    if(!outfile.is_open()){
        std::cerr << "Error: Could not open labels file: " << label_path << std::endl;
        return;
    }
    std::string label;
    while(std::getline(outfile, label)){
        auto class_name_it = label.find_first_of(',');
        label = label.substr(class_name_it + 2);
        a_labels.push_back(label);
    }
    std::cout << "Loaded " << a_labels.size() << " labels." << std::endl;
}

void AIManager::load_image_model(const std::string & model_path){
    if(a_image_model_loaded){
        std::cout << "model already loaded." << std::endl;
        return;
    }

    //create session options, and you can set the thread number;
    Ort::SessionOptions session_options;

    //create session
    a_session = Ort::Session(a_env, model_path.c_str(), session_options);
    
    

    try {
        a_session = Ort::Session(a_env, model_path.c_str(), session_options);
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // 获取并打印输入层名称
        size_t num_inputs = a_session.GetInputCount();
        std::cout << "Number of model inputs: " << num_inputs << std::endl;

        Ort::Allocator allocator(a_session, memory_info);
        for (size_t i = 0; i < num_inputs; ++i) {
            Ort::AllocatedStringPtr input_name = a_session.GetInputNameAllocated(i, allocator);
            std::cout << "Input Name " << i << ": " << input_name << std::endl;
        }

        // 获取并打印输出层名称
        size_t num_outputs = a_session.GetOutputCount();
        std::cout << "Number of model outputs: " << num_outputs << std::endl;

        for (size_t i = 0; i < num_outputs; ++i) {
            Ort::AllocatedStringPtr output_name = a_session.GetOutputNameAllocated(i, allocator);
            std::cout << "Output Name " << i << ": " << output_name.get() << std::endl;
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        a_image_model_loaded = false;
        return;
    }
    
    a_image_model_loaded = true;
    std::cout << "Model loaded successfully from " << model_path << std::endl;
}


std::string AIManager::run_image_inference(const std::string & image_file_path){
    if(!a_image_model_loaded){
        return "Error: Model not loaded.\n";
    }

    std::cout << "Running inference on " << image_file_path << std::endl;
    //1. read the image
    int width, height, channels;
    unsigned char * image_data = stbi_load(image_file_path.c_str(), &width, &height, &channels, 3);
    if(!image_data){
        return "Error: Could not load image.\n";
    }

    //resize to 224 224
    const int target_width = 224;
    const int target_height = 224;
    std::vector<unsigned char> resize_image(target_width * target_width * channels);
    stbir_resize_uint8(image_data, width, height, 0, resize_image.data(), target_width, target_height, 0, channels);
    stbi_image_free(image_data);



    //2. change the image to onnx's tensor
    // mobilenetv2 -> [1, 3, H, W]
    std::vector<float> input_tensor_values(target_width * target_height * 3);
    for(int c = 0; c < 3; ++c){
        for(int h = 0; h < target_height; ++h){
            for(int w = 0; w < target_width; ++w){
                int hwc_index = (h * target_width + w) * 3 + c;
                int cwh_index = c * (target_height * target_width) + (h * target_width + w);

                float pixel_value = static_cast<float>(resize_image[hwc_index]);

                input_tensor_values[cwh_index] = (pixel_value / 255.0f - 0.5f) + 2.0f;
            }
        }
    }


    //free
    std::array<int64_t, 4> input_shape {1,3,target_height,target_width};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    //3. run the a_session
    std::vector<const char*> input_names {"data"};
    std::vector<const char*> output_names {"mobilenetv20_output_flatten0_reshape0"};

    auto output_tensors = a_session.Run(
        Ort::RunOptions(nullptr),
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1
    );


    //4. parse the output tensor
    float * output_data = output_tensors.front().GetTensorMutableData<float>();
    //5. change the tensor to string and return
    std::vector<float> output_values(output_data, output_data + 1000);
    auto max_it = std::max_element(output_values.begin(), output_values.end());
    int max_index = std::distance(output_values.begin(), max_it);


    //convert to labels
    std::string predicted_label = "Unknown";
    if(max_index >= 0 && max_index < a_labels.size()){
        predicted_label = a_labels[max_index];
    }

    // return "Inference result: placeholder";
    return "Predicted class index : " + predicted_label + ", socre" + std::to_string(*max_it); 
}