//
// Created by geriatronics on 20.10.23.
//
#include <TagDetection.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    string imagePath = "/home/geriatronics/hao/skeleton_fusion/results/color1/color_36.png";
    Mat img = imread(imagePath);
    imshow("test", img);
    TagDetector TD;
    TD.setTagCode("36h11");
    int tagID = 89;
    bool isDetected = TD.isDetected(tagID, &img);

    cout<<isDetected<<endl;
}