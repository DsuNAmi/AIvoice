//file:: server.hpp
#pragma once

#include <string>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <numeric>
#include <vector>
#include <algorithm>

#include "ai_manager.hpp"


namespace AIvoice{
    class Server{
        public:
            Server(std::string ipaddr, unsigned int port, int threads);


            void run();


        private:

            void do_accept();

            void handle_request(
                boost::beast::http::request<boost::beast::http::string_body> && req,
                boost::beast::tcp_stream & stream
            );

            boost::asio::ip::tcp::endpoint s_make_endpoint();

            std::string s_ipaddr;
            unsigned int s_port;


            boost::asio::io_context s_ioc;
            boost::asio::ip::tcp::acceptor s_acceptor;

            AIManager s_ai_manager;
    };
}