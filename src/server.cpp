//file:: server.cpp
#include "include/server.hpp"



namespace AIvoice{

    Server::Server(std::string ipaddr, unsigned int port, int threads)
    : s_ipaddr(std::move(ipaddr)), s_port(port),
    s_ioc(threads), s_acceptor(s_ioc, s_make_endpoint()),
    s_ai_manager()
    {
    }

    boost::asio::ip::tcp::endpoint Server::s_make_endpoint(){
        if(s_ipaddr == "localhost"){
            return boost::asio::ip::tcp::endpoint(
                boost::asio::ip::tcp::v4(), s_port
            );
        }else{
            return boost::asio::ip::tcp::endpoint(
                    boost::asio::ip::make_address(s_ipaddr),s_port
                );
        }
    }


    void Server::run(){
        //load the ai model
        //wait a min, reload when somebody refresh the website every time?
        s_ai_manager.load_model("../models/");
        do_accept();
        std::cout << "Server listening on " << s_ipaddr << ":" << s_port << std::endl;
        s_ioc.run();
    }

    void Server::do_accept(){
        s_acceptor.async_accept(
            boost::asio::make_strand(s_ioc),
            [this](boost::beast::error_code ec, boost::asio::ip::tcp::socket socket){
                if(!ec){
                    boost::beast::tcp_stream stream(std::move(socket));

                    boost::beast::flat_buffer buffer;
                    boost::beast::http::request<boost::beast::http::string_body> req;
                    boost::beast::http::read(stream, buffer, req);

                    handle_request(std::move(req), stream);
                }
                do_accept();
            }
        );
    }

    void Server::handle_request(
        boost::beast::http::request<boost::beast::http::string_body> && req,
        boost::beast::tcp_stream & stream
    ){
        std::string response_body;
        boost::beast::http::status response_status;


        if(req.method() == boost::beast::http::verb::get && req.target() == "/"){
            response_status = boost::beast::http::status::ok;
            nlohmann::json res_json;
            res_json["status"] = "ok";
            res_json["message"] = "Welcome to AIvoice";
            response_body = res_json.dump();
        }else if(req.method() == boost::beast::http::verb::post && req.target() == "/upload"){
            try{
                // the whole body just contains <upload_files>
                const std::string & body_content = req.body();

                auto now = std::chrono::system_clock::now();
                auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
                std::string filename = "file_" + std::to_string(now_ms.count()) + ".bin";
                std::string filepath = "../uploaded_files/" + filename;


                std::cout << filepath << std::endl;

                std::ofstream outfile(filepath, std::ios::binary);
                if(!outfile.is_open()){
                    throw std::runtime_error("Could not open file for writing.\n");
                }
                outfile.write(body_content.c_str(), body_content.length());
                outfile.close();


                //use the ai
                std::string inference_result = s_ai_manager.run_inference(filepath);

                response_status = boost::beast::http::status::ok;
                nlohmann::json res_json;
                res_json["status"] = "ok";
                res_json["message"] = "File uploaded successfully.";
                res_json["filename"] = filename;
                res_json["inference_result"] = inference_result;
                response_body = res_json.dump();
                

            }catch(const std::exception & e){
                std::cerr << "Error during file upload: " << e.what() << std::endl;
                response_status = boost::beast::http::status::internal_server_error;
                nlohmann::json res_json;
                res_json["status"] = "error";
                res_json["message"] = "Server error during upload.";
                response_body = res_json.dump();
            }
        }else{
            response_status = boost::beast::http::status::not_found;
            nlohmann::json res_json;
            res_json["status"] = "error";
            res_json["message"] = "Resource not found";
            response_body = res_json.dump();
        }

        boost::beast::http::response<boost::beast::http::string_body> res{response_status, req.version()};
        res.set(boost::beast::http::field::server, "AIvoice-Server");
        res.set(boost::beast::http::field::content_type, "application/json");
        res.content_length(response_body.size());
        res.body() = response_body;
        res.prepare_payload();

        boost::beast::http::write(stream, res);
    }
}