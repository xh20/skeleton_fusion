//
// Created by hao on 11.08.21.
//

#ifndef SKELETON_FUSION_TAGDETECTION_HPP
#define SKELETON_FUSION_TAGDETECTION_HPP

#endif //SKELETON_FUSION_TAGDETECTION_HPP

#include "opencv2/opencv.hpp"
#include <iostream>
#include <Eigen/Dense>
//extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include <apriltag/tag16h5.h>
#include <apriltag/tag25h9.h>
#include <apriltag/tagStandard41h12.h>
#include <apriltag/tagStandard52h13.h>
#include <apriltag/apriltag_pose.h>
//}

using namespace Eigen;

class TagDetector{

private:
    string tagCode;
    zarray_t *detections{};
    apriltag_detector_t* tagDetector{};
    apriltag_family_t* tagFamily{};
    apriltag_detection_t* det{};
    apriltag_detection_info_t camInfo{};
    apriltag_pose_t pose{};
    Matrix3d R;
    Vector3d T;
    Mat frameD;
    int ID;
    static tuple<Matrix3d, Vector3d> pose2Matrix(const apriltag_pose_t *pose);
    void poseEstimation(Mat& depthImg);

public:
    TagDetector();
    ~TagDetector();
    void setCamInfo(const double& tagsize, const double& fx, const double& fy,
            const double& cx, const double& cy);
    void setTagCode(const string& s);
    bool isDetected(const int& detectionID, const Mat& img);
    tuple<Matrix3d, Vector3d> getTagPose();
    void cleanUp() const;
    Vector3d getTagToCam(Mat& depthImg);
    Vector3d getTagCenter();
    tuple<MatrixXd, VectorXd> getRelativeTransform();
};

inline TagDetector::TagDetector() {
    tagDetector = apriltag_detector_create();
}

inline TagDetector::~TagDetector() {
    cleanUp();
}

inline void TagDetector::setCamInfo(const double& tagSize, const double& fx, const double& fy, const double& cx,
        const double& cy) {
    camInfo.det = det;
    camInfo.tagsize = tagSize;  // In meters.
    camInfo.fx = fx;    // In pixels.
    camInfo.fy = fy;    // In pixels.
    camInfo.cx = cx;    // In pixels.
    camInfo.cy = cy;    // In pixels.
}

inline void TagDetector::setTagCode(const string& s) {
    tagCode = s;
    if (s=="16h5") {
        tagFamily = tag16h5_create();
    } else if (s=="25h9") {
        tagFamily = tag25h9_create();
    } else if (s=="36h11") {
        tagFamily = tag36h11_create();
    } else if (s=="41h12") {
        tagFamily = tagStandard41h12_create();
    } else if (s=="52h13") {
        tagFamily = tagStandard52h13_create();
    }
    else {
        cout << "Invalid tag family specified" << endl;
        exit(1);
    }
}


inline tuple<Matrix3d, Vector3d> TagDetector::pose2Matrix(const apriltag_pose_t* pose){
    double* RData = pose->R->data;

    Map<Matrix3d> R(pose->R->data,3,3); //,3,3
    Map<Vector3d> T(pose->t->data);
    return make_tuple(R,T);
}

inline bool TagDetector::isDetected(const int& detectionID, const Mat& matImg) {
    bool isTagdetected = false;
    ID = detectionID;
    apriltag_detector_add_family(tagDetector, tagFamily);

    // convert image into grayscale, filter depth image
    Mat imgGray;
    cvtColor(matImg, imgGray, CV_RGB2GRAY);
    image_u8_t img = { .width = imgGray.cols,
            .height = imgGray.rows,
            .stride = imgGray.cols,
            .buf = imgGray.data
    };

    detections = apriltag_detector_detect(tagDetector, &img);

    if (zarray_size(detections)==0){
        return false;
    }
    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* detTemp;
        zarray_get(detections, i, &detTemp);
        if(detTemp->id!=ID)
        {
            continue;
        }
        isTagdetected = true;
        det = detTemp;
        line(matImg, cv::Point(det->p[0][0], det->p[0][1]),
             cv::Point(det->p[1][0], det->p[1][1]),
             Scalar(0, 0xff, 0), 2);
        line(matImg, cv::Point(det->p[0][0], det->p[0][1]),
             cv::Point(det->p[3][0], det->p[3][1]),
             Scalar(0, 0, 0xff), 2);
        line(matImg, cv::Point(det->p[1][0], det->p[1][1]),
             cv::Point(det->p[2][0], det->p[2][1]),
             Scalar(0xff, 0, 0), 2);
        line(matImg, cv::Point(det->p[2][0], det->p[2][1]),
             cv::Point(det->p[3][0], det->p[3][1]),
             Scalar(0xff, 0xff, 0), 2);

        int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontscale = 1.0;
        int baseline;
        stringstream ss;
        ss << det->id;
        string text = ss.str();
        Size textsize = getTextSize(text, fontface, fontscale, 2,
                                    &baseline);
        putText(matImg, text, cv::Point(det->c[0]-double(textsize.width/2.0),
                                   det->c[1]+double(textsize.height/2.0)),
                fontface, fontscale, Scalar(0xff, 0x99, 0), 2);
    }
    cout <<"Is tag "<<ID<<" detected? "<<isTagdetected << endl;
    return isTagdetected;
}

inline Vector3d TagDetector::getTagToCam(Mat& depthImg){
    medianBlur(depthImg, depthImg, 5);
    double px = det->c[0];
    double py = det->c[1];
    double depth = depthImg.at<ushort>(cvRound(py),cvRound(px))/1000.0;
    if(px > 0.0 && py>0.0 && depth<=0.0){
        depth = 3.500;
    }
    double X = ((px - camInfo.cx)/camInfo.fx)*depth;
    double Y = ((py - camInfo.cy)/camInfo.fy)*depth;
    Vector3d pose2Cam;
    pose2Cam << X, Y, depth;
//    cout<<"cam parameters: "<<camInfo.fx<<" "<<camInfo.fy<<" "<<camInfo.cx<<" "<<camInfo.cy<<" "<<endl;
//    cout<<"pixel info of P: "<< px <<" "<< py << " " << depth<<endl;
    return pose2Cam;
}

inline Vector3d TagDetector::getTagCenter(){
    Vector3d tagCenter;
    tagCenter << det->c[0], det->c[1], 1;
    return tagCenter;
}

inline tuple<Matrix3d, Vector3d> TagDetector::getTagPose() {
    double err = estimate_tag_pose(&camInfo, &pose);
//    estimate_pose_for_tag_homography(&camInfo, &pose);
    return pose2Matrix(&pose);

}

inline void TagDetector::cleanUp() const{
    if (tagCode=="16h5") {
        tag16h5_destroy(tagFamily);
    } else if (tagCode=="25h9") {
        tag25h9_destroy(tagFamily);
    } else if (tagCode=="36h11") {
        tag36h11_destroy(tagFamily);
    } else if (tagCode=="41h12") {
        tagStandard41h12_destroy(tagFamily);
    } else if (tagCode=="52h13") {
        tagStandard52h13_destroy(tagFamily);
    } else {
        cout << "Invalid tag family specified" << endl;
        exit(1);
    }
    apriltag_detector_destroy(tagDetector);
}

tuple<MatrixXd, VectorXd> TagDetector::getRelativeTransform() {
    std::vector<cv::Point3f> objPts;
    std::vector<cv::Point2f> imgPts;
    double s = camInfo.tagsize/2.;
    objPts.push_back(cv::Point3f(s,-s, 0));
    objPts.push_back(cv::Point3f(-s,-s, 0));
    objPts.push_back(cv::Point3f(-s, s, 0));
    objPts.push_back(cv::Point3f(s, s, 0));

    imgPts.push_back(cv::Point2f(det->p[0][0], det->p[0][1]));
    imgPts.push_back(cv::Point2f(det->p[1][0], det->p[1][1]));
    imgPts.push_back(cv::Point2f(det->p[2][0], det->p[2][1]));
    imgPts.push_back(cv::Point2f(det->p[3][0], det->p[3][1]));

    cv::Mat rvec, tvec;
    cv::Matx33f cameraMatrix(
            camInfo.fx, 0, camInfo.cx,
            0, camInfo.fy, camInfo.cy,
            0,  0,  1);
    cv::Vec4f distParam(0,0,0,0); // all 0?
    cv::solvePnP(objPts, imgPts, cameraMatrix, distParam, rvec, tvec);
    cv::Mat r(3,3,CV_32FC1);
    cv::Rodrigues(rvec, r);
    Matrix3d wRo, Ro;
    Vector3d Tr;
    Ro << r.at<float>(0,0), r.at<float>(0,1), r.at<float>(0,2), r.at<float>(1,0), r.at<float>(1,1), r.at<float>(1,2), r.at<float>(2,0), r.at<float>(2,1), r.at<float>(2,2);
    Tr << tvec.at<float>(0), tvec.at<float>(1), tvec.at<float>(2);

    return make_tuple(Ro, Tr);
}