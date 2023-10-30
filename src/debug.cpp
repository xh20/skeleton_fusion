//
// Created by geriatronics on 20.10.23.
//
#include <TagDetection.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <librealsense2/rs.hpp>
#include <experimental/filesystem>
#include <fstream>

using namespace std;
using namespace cv;
using namespace rs2;
namespace fs = std::experimental::filesystem;

void isCameraReady(pipeline *pipe, bool* isReady, double* fx, double*fy, double*cx, double*cy){
    *isReady = false;
    *fx=0.0, *fy=0.0, *cx=0.0, *cy=0.0;
    rs2::frameset frames;
    for(int i = 0; i < 30; i++)
    {
        //Wait for all configured streams to produce a frame
        frames = pipe->wait_for_frames();
    }
    rs2::video_frame colorFrame = frames.get_color_frame();
    if(colorFrame&& frames.get_depth_frame()){
        rs2_intrinsics inrist = video_stream_profile(colorFrame.get_profile()).get_intrinsics();
        *fx = double(inrist.fx);
        *fy = double(inrist.fy);
        *cx = double(inrist.ppx);
        *cy = double(inrist.ppy);
        cout<<"Camera is warm up!"<<endl;
        *isReady = true;
    }
}


int main(int argc, char** argv){
    context ctx;
    int WIDTH = 848, HEIGHT = 480;
    fs::path saveDir = "/home/geriatronics/hao/skeleton_fusion/results/";
    string cam1SaveName = "color1_param.txt";
    string cam2SaveName = "color2_param.txt";
    ofstream predictionTxt;
    predictionTxt.open(fs::path(saveDir)/"cam_parameters.txt");

    vector<string> serials;
    vector<pipeline> pipelines;
    vector<pipeline_profile> profiles;
    vector<float> depthScales;
    for (auto&& dev : ctx.query_devices()) {
        serials.emplace_back(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    }
    if ((serials).size() != 2){
        cout<< "doesn't detect two cameras!"<<endl;
        return 0;
    }

    for (auto&& serial : serials)
    {
        pipeline pipe(ctx);
        config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, RS2_FORMAT_Z16, 30);
        cfg.enable_device(serial);
        pipeline_profile profile = pipe.start(cfg);
        pipelines.emplace_back(pipe);
//        depthScales.emplace_back(getDepthScale(profile.get_device()));
//            cout<<"serial: "<<serial<<endl;
//            cout<<"profile.get_device() "<<profile.get_device()<<endl;
    }
    // drop first N frames
    auto pipe1 = pipelines[0];
    auto pipe2 = pipelines[1];
    bool isCamReady1=false, isCamReady2=false;
    double fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2;
    isCameraReady(&pipe1, &isCamReady1, &fx1, &fy1, &cx1, &cy1);
    isCameraReady(&pipe2, &isCamReady2, &fx2, &fy2, &cx2, &cy2);
    predictionTxt << serials[0] << " "<< fx1 << " " << fy1 << " " << cx1 << " " << cy1 <<'\n'
            << serials[1] << " "<< fx2 << " " << fy2 << " " << cx2 << " " << cy2 <<'\n';
    predictionTxt.close();
}