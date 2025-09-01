//file:: main.cpp
#include "include/server.h"

#include <stdexcept>

int main(){
    try{
        auto sp_server = std::make_shared<AIvoice::Server>("localhost",8080,1);
        sp_server->run();
    }catch(const std::exception & e){
        std::cerr << "[[Error]]:: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}