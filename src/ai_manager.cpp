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

void AIManager::get_input_output_name(const Ort::Session & session){
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    size_t num_inputs = session.GetInputCount();
    std::cout << "Number of model inputs: " << num_inputs << std::endl;

    Ort::Allocator allocator(session, memory_info);
    for(size_t i = 0; i < num_inputs; ++i){
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "Input Name " << i << ": " << input_name << std::endl;
    }

    size_t num_outputs = session.GetOutputCount();
    std::cout << "Number of model outputs: " << num_outputs << std::endl;

    for(size_t i = 0; i < num_outputs; ++i){
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        std::cout << "Output Name " << i << ": " << output_name << std::endl;
    }

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
        
        get_input_output_name(a_encoder_session);
        get_input_output_name(a_decoder_session);

        a_audio_model_loaded = true;
        std::cout << "Audio models loaded successfully." << std::endl;
    }
    catch(const Ort::Exception & e)
    {
        std::cerr << "ONNX Runtime Error loading audio models: " << e.what() << std::endl;
        a_audio_model_loaded = false;
    }
    
}

std::string AIManager::decode_and_transcribe(const Ort::Value & encoder_output){
    Ort::SessionOptions session_options;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<const char*> decoder_input_names = {"input_ids","encoder_hidden_states"};
    std::vector<const char*> decoder_output_names = {"logits"};

    //the token is general, so what's the main token?
    std::vector<int64_t> input_ids {50257};

    std::vector<int64_t> output_tokens;
    const int64_t EOT_TOKEN = 50256;
    const int MAX_LENGTH = 200;

    for(int i = 0; i < MAX_LENGTH; ++i){
        //create input tensor
        std::array<int64_t, 2> input_shape = {1, static_cast<long long>(input_ids.size())};
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            input_ids.data(),
            input_ids.size(),
            input_shape.data(),
            input_shape.size()
        );

        std::vector<Ort::Value> decoder_inputs;
        decoder_inputs.push_back(std::move(input_ids_tensor));
        decoder_inputs.push_back(std::move(const_cast<Ort::Value&>(encoder_output)));

        auto decoder_outputs = a_decoder_session.Run(
            Ort::RunOptions{nullptr},
            decoder_input_names.data(),
            decoder_inputs.data(),
            1,
            decoder_output_names.data(),
            1
        );

        //get the highest token
        float * logits = decoder_outputs[0].GetTensorMutableData<float>();

        auto logits_shape = decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocab_size = logits_shape[2];
        float * last_logits = logits + (logits_shape[1] - 1) * vocab_size;

        int64_t next_token_index = std::distance(last_logits, std::max_element(last_logits, last_logits + vocab_size));

        if(next_token_index == EOT_TOKEN){
            break;
        }

        input_ids.push_back(next_token_index);

    }

    return "Final transcribed text : Placeholder";
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

    int audio_stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if(audio_stream_idx < 0){
        return "Error: Could not find audio stream.\n";
    }

    //open the decoder
    codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[audio_stream_idx]->codecpar);
    if(avcodec_open2(codec_ctx, codec, nullptr) < 0){
        return "Error: Could not open codec.\n";
    }

    std::vector<float> pcm_data;
    while(av_read_frame(fmt_ctx, pkt) >= 0){
        if(pkt->stream_index == audio_stream_idx){
            if(avcodec_send_packet(codec_ctx, pkt) >= 0){
                while(avcodec_receive_frame(codec_ctx, frame) >= 0){
                    //simple deal
                    for(int i = 0; i < frame->nb_samples; ++i){
                        pcm_data.push_back(static_cast<float>(reinterpret_cast<int16_t*>(frame->data[0])[i]) / 32768.0f);
                    }
                }
            }
        }
        av_packet_unref(pkt);
    }
    //2.convert PCM to Mel Spectrogram
    const int n_fft = 400; //windwos size
    const int hop_length = 160; //frame offset
    const int n_mels = 80; //mel channels
    const int sample_rate = 16000;
    //3.convert Mel to tensor
    //4.get the encoder's output
    //5.cal

    //6.decode round by round, and get a token
    // std::vector<float> mel_spectrogram_data;
    std::vector<std::vector<float>> mel_spectrogram_data;

    if(pcm_data.size() < n_fft){
        return "Error: PCM data is too short for Mel Spectrogram.\n";
    }

    int num_frame = (pcm_data.size() - n_fft) / hop_length + 1;
    mel_spectrogram_data.resize(num_frame, std::vector<float>(n_mels, 0.0f));

    for(int i = 0; i < num_frame; ++i){
        int start_pos = i * hop_length;
        std::vector<float> frame_data(pcm_data.begin() + start_pos, pcm_data.begin() + start_pos + n_fft);

        //simple deal

        for(int j = 0; j < n_mels; ++j){
            mel_spectrogram_data[i][j] = std::log(std::abs(frame_data[j] * 10.0f) + 1e-6);
        }
    }

    std::vector<float> encoder_input_values;
    encoder_input_values.reserve(num_frame * n_mels);
    for(const auto & row : mel_spectrogram_data){
        encoder_input_values.insert(encoder_input_values.end(), row.begin(), row.end());
    }



    std::array<int64_t, 3> encoder_input_shape {1, n_mels, static_cast<long long>(num_frame)};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,OrtMemType::OrtMemTypeDefault);


    Ort::Value encoder_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        encoder_input_values.data(),
        encoder_input_values.size(),
        encoder_input_shape.data(),
        encoder_input_shape.size()
    );

    std::vector<const char*> encoder_input_names = {"input_features"};
    std::vector<const char*> encoder_output_names = {"last_hidden_state"};

    auto encoder_outputs = a_encoder_session.Run(
        Ort::RunOptions{nullptr},
        encoder_input_names.data(),
        &encoder_input_tensor,
        1,
        encoder_input_names.data(),
        1 
    );
    

    //7.change the token to str
    std::string transcription = decode_and_transcribe(encoder_outputs[0]);


    //free sources
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);


    return transcription;
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
            std::cout << "Output Name " << i << ": " << output_name << std::endl;
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