 #include <opencv2/core/core.hpp>
 #include <opencv2/highgui.hpp>
 #include <opencv2/imgcodecs.hpp>
 #include <opencv2/imgproc.hpp>

 #include <unistd.h>
 #include <stdio.h>
 
 #include <errno.h>
 #include <fcntl.h>
 #include <stdlib.h>
 #include <string.h>
 #include <iostream>
 #include <termios.h>
 #include <thread>
 #include <sstream>
 #include <time.h>
 #include <sys/stat.h>
 #include <sys/types.h>
 #include <chrono>
 #include <sys/socket.h>
 #include <netinet/in.h>
 #include <arpa/inet.h>
 #include <cstdlib>
 #include <cstring>
 #include "sys/sysinfo.h"

using namespace std;

int main(int argc, char *argv[]) {
 
    char buffer[1024] = {0}; 
    int sockfd, portno;
    socklen_t clilen;
  
    portno = 2345;
    sockfd = socket(AF_INET, SOCK_STREAM, 0); 
    struct sockaddr_in serv_addr;
 
    if (sockfd < 0)  
        perror("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    // localhost
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    serv_addr.sin_port = htons(portno);
    cout << serv_addr.sin_addr.s_addr << endl;
    cout << serv_addr.sin_family << endl;
 
    int rval = connect(sockfd,(struct sockaddr*)&serv_addr, sizeof(serv_addr));
    cout << "val" << rval << endl;
    char* test = "test";
    send(sockfd , test, strlen(test), 0 ); 
    int valread;
    valread = read(sockfd, buffer, 1024); 
    cout << buffer << endl;

    //int h=1080, w=1080;
    int h=100, w=50;
    cv::Mat img = cv::Mat::zeros(h, w, CV_8UC1);
    img.colRange(10, 20).setTo(cv::Scalar(100));
    cv::imwrite("send.jpg", img);
    int imageSize = img.total()*img.elemSize();
    char* cmd = "img";
    send(sockfd , cmd, strlen(cmd), 0 ); 
    valread = read(sockfd, buffer, 1024); 
    cout << buffer << endl;

    send(sockfd , img.data, imageSize, 0 ); 
    valread = read(sockfd, buffer, 1024); 
    cout << buffer << endl;

    return 0;
}

