/*
 * Copyright (c) 2012 Aldebaran Robotics. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the COPYING file.
 */
#include <iostream>
#include <cstdlib>
#include <unistd.h>

#include <boost/shared_ptr.hpp>

#include <alcommon/almodule.h>
#include <alcommon/albroker.h>
#include <alcommon/albrokermanager.h>

#include "potentialNavigationNAO.h"
#include <qi/os.hpp>

using namespace std;

int main(int argc, char* argv[])
{
    int pport = 9559;
    //int camera_flag = -1;
    string pip = "127.0.0.1";
    string video_path = "/home/marcubuntu/Documenti/Software/qi_work_tree/potentialNavigationNAO/NAOvideo1.avi";
    bool online = false;

    if(argc != 3 && argc != 5){
        cout << "Wrong number of arguments!" << endl;
        cout << "Usage: \n\t[ONLINE]: potentialNavigationNAO [--pip NAO_ip_number] [--pport NAO_port]"// [--camera camera_id]"
                       //"\n\t\t(camera_id = 0 -> top camera "
                       //"\n\t\t camera_id = 1 -> bottom camera)"
                       "\n\t[OFFLINE]: potentialNavigationNAO [--video input_video_path]" << endl;
        exit(1);
    }

    if(argc == 3){
        if(string(argv[1])=="--video"){
            video_path = string(argv[2]);
        }
        else if(string(argv[1])=="--pip"){
            pip = string(argv[2]);
            online = true;
        }
        else if(string(argv[1])=="--pport"){
            pport = atoi(argv[2]);
            online = true;
        }
    }

    if(argc == 5){
        online = true;
        if(string(argv[1]) == "--pport" && string(argv[3]) == "--pip"){
            pport = atoi(argv[2]);
            pip = argv[4];
        }
        else if(string(argv[3]) == "--pport" && string(argv[1]) == "--pip"){
            pport = atoi(argv[4]);
            pip = argv[2];
        }
        else{
            cout << "Wrong number of arguments!" << endl;
            cout << "Usage: \n\t[ONLINE]: potentialNavigationNAO [--pip NAO_ip_number] [--pport NAO_port]"
                           "\n\t[OFFLINE]: potentialNavigationNAO [--video input_video_path]" << endl;
            exit(1);
        }

    }

    if(online){ //create a broker and initialize the structure with NAO ip and port
        // Need this to for SOAP serialization of floats to work
        setlocale(LC_NUMERIC, "C");

        // A broker needs a name, an IP and a port:
        const std::string brokerName = "mybroker";
        // FIXME: would be a good idea to look for a free port first
        int brokerPort = 54000;
        // listen port of the broker (here an anything)
        const std::string brokerIp = "0.0.0.0";


        // Create your own broker
            boost::shared_ptr<AL::ALBroker> broker;
        try
        {
          broker = AL::ALBroker::createBroker(
              brokerName,
              brokerIp,
              brokerPort,
              pip,
              pport,
              0    // you can pass various options for the broker creation,
                   // but default is fine
            );
        }
        catch(...)
        {
          std::cerr << "Fail to connect broker to: "
                    << pip
                    << ":"
                    << pport
                    << std::endl;

          AL::ALBrokerManager::getInstance()->killAllBroker();
          AL::ALBrokerManager::kill();

          return 1;
        }

        // Deal with ALBrokerManager singleton (add your borker into NAOqi)
        AL::ALBrokerManager::setInstance(broker->fBrokerManager.lock());
        AL::ALBrokerManager::getInstance()->addBroker(broker);//*/

        // Now it's time to load your module with
        // AL::ALModule::createModule<your_module>(<broker_create>, <your_module>);
        AL::ALModule::createModule<potentialNavigationNAO>(broker, "potentialNavigationNAO");

        while (true){
            qi::os::sleep(1);

        }
        //potentialNavigationNAO(string(pip),pport);
    }
    else{ //initialize the structure with the video path of the recorded video
        //potentialNavigationNAO(string(video_path));
    }

    return 0;
}
