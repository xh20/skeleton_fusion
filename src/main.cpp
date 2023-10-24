// https://github.com/Cryoris/matplotlib-cpp.git//
// Created by hao on 09.08.21.
//

#include <SkeletonFusion.hpp>
#include <TagDetection.hpp>
#include <GraduateNonConvexity.hpp>
#include <string>
#include <experimental/filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <matplotlibcpp.h>
#include <experimental/filesystem>
#include <map>
#include <eigen3/Eigen/Dense>
#include <librealsense2/rs.hpp>

using namespace std;
using namespace rs2;
namespace fs = std::experimental::filesystem;

DEFINE_string(video1, "/media/dataset/translation/S00V00A16O00T1/colorframes/out.mp4",
              "Give the first video file path");
DEFINE_string(video2, "/media/dataset/translation/S00V01A16O00T1/colorframes/out.mp4",
              "Give the second video file path");
DEFINE_bool(image, false, "is input as image");
DEFINE_bool(demo, true, "is input as camera");
DEFINE_bool(save_images, true, "save all images");


// Tag size
//DEFINE_string(colorDir1, "/media/dataset/translation/results_medical2/001622070717/color/",  // results_dailytask results_medical
//              "Give the first video file path");
//DEFINE_string(colorDir2, "/media/dataset/translation/results_medical2/845112070795/color/", // results_test results_daily1
//              "Give the second video file path");
//DEFINE_string(depthDir1, "/media/dataset/translation/results_medical2/001622070717/depth/",
//              "Give the first video file path");
//DEFINE_string(depthDir2, "/media/dataset/translation/results_medical2/845112070795/depth/",
//              "Give the second video file path");
DEFINE_string(colorDir1, "/media/dataset/HDTR_bagfile/results/cam1/color/",  // results_dailytask results_medical
              "Give the first video file path");
DEFINE_string(colorDir2, "/media/dataset/HDTR_bagfile/results/cam2/color/", // results_test results_daily1
              "Give the second video file path");
DEFINE_string(depthDir1, "/media/dataset/HDTR_bagfile/results/cam1/depth/",
              "Give the first video file path");
DEFINE_string(depthDir2, "/media/dataset/HDTR_bagfile/results/cam2/depth/",
              "Give the second video file path");


DEFINE_string(tagCode, "36h11", "Give the tag code to choose tag family");
DEFINE_int32(TagID, 89, "Give the Tag ID to calculate the extrinsic parameters, hand marker: 89, table left: 54, table right: 25");
DEFINE_string(saveDir, "/home/geriatronics/hao/skeleton_fusion/results",
              "Give the save path");

double standardRad(double t) {
    if (t >= 0.) {
        t = fmod(t+CV_PI, 2.0*CV_PI) - CV_PI;
    } else {
        t = fmod(t-CV_PI, -2.0*CV_PI) + CV_PI;
    }
    return t;
}

void wRo_to_euler(const Eigen::Matrix3d& wRo, double& yaw, double& pitch, double& roll) {
    yaw = standardRad(atan2(wRo(1,0), wRo(0,0)));
    double c = cos(yaw);
    double s = sin(yaw);
    pitch = standardRad(atan2(-wRo(2,0), wRo(0,0)*c + wRo(1,0)*s));
    roll  = standardRad(atan2(wRo(0,2)*s - wRo(1,2)*c, -wRo(0,1)*s + wRo(1,1)*c));
}

void MyFilledCircle( Mat img, const cv::Point& center, const Scalar& color)
{
    circle( img,
            center,
            10,
            color,
            FILLED,
            LINE_8 );
}

void MyTextOnImage(Mat img, const string& text){
    cv::putText(img, //target image
                text, //tex
                cv::Point(10, 50), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(118, 185, 0), //font color
                2);
}

class SkeletonMerger
{
private:
    double cx1, cy1, fx1, fy1, cx2, cy2, fx2, fy2, tagSize, L1, L2, Lambda1, Lambda2, stepSize;
    Mat frame1, frame2, frameD1, frameD2;
    VectorXd Acc1, Acc2, AccTarget;
    MatrixXd Sk1, Sk2, Sk1Cam2, Sk2Cam1, PTarget, PCam1, PCam2;
    Matrix3d RSum, R12, K1, K2, R21, RHumanCam;
    Vector3d TSum, T12, T21, P1, P2, P3, THumanCam, PNeck, PHip, PShoulder;
    vector<double> L12, L23;
    bool gotTrans;
    // tag detector
    TagDetector* TD;
    // openpose wrapper
    OpWrapper* OW;
    // Graduate Non-Convexity
    GraduateNonConvexity GNC;
    int index, *target, targetCount;
    bool transformed = false;
    int countTF;
    void setFrames(const Mat& f1, const Mat& f2, const Mat& fD1, const Mat& fD2);
    void updateRT();
public:
    SkeletonMerger();
    ~SkeletonMerger();
    void setCamParameters(const double&, const double&,const double& , const double&, const double&, const double&,
                          const double&, const double&);
    void setTarget(int t[], const double& length, const double& lambda, const double& delta, const double& size);
    void setThreeTargets(int *t, const double& l1, const double& l2, const double& lambda1,
                        const double& lambda2, const double& delta, const double& size);
    void processing(Mat* f1, Mat* f2, const Mat* fD1, const Mat* fD2, bool*);
    tuple<Vector3d, Vector3d> getEstimation();
    tuple<Vector3d, Vector3d, Vector3d> getThreeEstimation();
    bool isTransformed() const;
    tuple<Matrix3d, Vector3d> getHumanCamTF();
};

#include <map>

// "001622070717", {912.036, 910.78, 635.183, 366.158}
// "845112070795", {915.888, 915.828, 631.489, 372.897}
// "cam_1", {919.0277709960938, 0.0, 647.3279418945312, 0.0, 917.8308715820312, 368.21026611328125, 0.0, 0.0, 1.0}
// "cam_2", {921.8261108398438, 0.0, 641.4801635742188, 0.0, 921.8929443359375, 375.79901123046875, 0.0, 0.0, 1.0}



inline SkeletonMerger::SkeletonMerger():cx1(647.3279),cy1(368.2103),fx1(919.0278),fy1(917.8309),
cx2(641.4802),cy2(375.7990),fx2(921.8261),fy2(921.8929), gotTrans(false),
tagSize(0.1660),index(1), countTF(0){
    OW = new OpWrapper;
    TD = new TagDetector;
    RSum.setZero();
    TSum.setZero();
    R12.setZero();
    T12.setZero();
    R21.setZero();
    T21.setZero();

    TD->setTagCode(FLAGS_tagCode);
    K1 << fx1, 0.0, cx1, 0.0, fy1, cy1, 0.0, 0.0, 1.0;
    K2 << fx2, 0.0, cx2, 0.0, fy2, cy2, 0.0, 0.0, 1.0;
}

inline SkeletonMerger::~SkeletonMerger() = default;

inline bool SkeletonMerger::isTransformed() const{
    return transformed;
}

inline tuple<Matrix3d, Vector3d> SkeletonMerger::getHumanCamTF() {
    return make_tuple(RHumanCam, THumanCam);
}

inline void SkeletonMerger::setTarget(int *t, const double& length, const double& lambda1, const double& delta, const double& size) {
    target = t;
    tagSize = size;
    targetCount = target[0];
    // targets are 3: right elbow; 4: right wrist
    PCam1.setZero(3, targetCount);
    PCam2.setZero(3, targetCount);
    PTarget.setZero(3,2*targetCount);
    AccTarget.setZero(2*targetCount);
    L1 = length;
    Lambda1 = lambda1;
    stepSize = delta;
}

inline void SkeletonMerger::setThreeTargets(int *t, const double& l1, const double& l2, const double& lambda1,
                                           const double& lambda2, const double& delta, const double& size) {
    target = t;
    tagSize = size;
    targetCount = target[0];
    cout<<"target count: "<<targetCount<<endl;
    // targets are 2: right shoulder; 3: right elbow; 4: right wrist
    PCam1.setZero(3, targetCount);
    PCam2.setZero(3, targetCount);
    PTarget.setZero(3,2*targetCount);
    AccTarget.setZero(2*targetCount);
    L1 = l1;
    L2 = l2;
    Lambda1 = lambda1;
    Lambda2 = lambda2;
    stepSize = delta;
}

inline void SkeletonMerger::setFrames(const Mat &f1, const Mat &f2, const Mat &fD1, const Mat &fD2) {
    frame1 = f1;
    frame2 = f2;
    frameD1 = fD1;
    frameD2 = fD2;

}

inline void SkeletonMerger::setCamParameters(const double& ffx1, const double& ffy1, const double& fcx1,
                                             const double& fcy1, const double& ffx2,const double& ffy2,
                                             const double& fcx2, const double& fcy2) {
    fx1 = ffx1;
    fy1 = ffy1;
    cx1 = fcx1;
    cy1 = fcy1;
    fx2 = ffx2;
    fy2 = ffy2;
    cx2 = fcx2;
    cy2 = fcy2;
    K1 << fx1, 0.0, cx1, 0.0, fy1, cy1, 0.0, 0.0, 1.0;
    K2 << fx2, 0.0, cx2, 0.0, fy2, cy2, 0.0, 0.0, 1.0;
}

inline void SkeletonMerger::updateRT(){
    Matrix3d R1, R2, R1PnP, R2PnP; Vector3d T1, T2, c1, c2, C1, C2, T1PnP, T2PnP;

    Matrix3d F;
    Matrix3d fixR;
    double yaw, pitch, roll;
    Matrix3d m1, m2, m12;

    F << 1, 0, 0, 0, -1, 0, 0, 0, 1;
    // cam1 coordiante
    bool detected1, detected2;
    TD->isDetected(&FLAGS_TagID, &frame1, &detected1);
    if(detected1){
        c1.setZero();
        TD->setCamInfo(tagSize, fx1, fy1, cx1, cy1);
        // Rotation and translation from original function
        tie(R1, T1) = TD->getTagPose();
//        tie(R1, T1) = TD->getRelativeTransform();
        // translation using depth image
        C1 = TD->getTagToCam(frameD1);
        // center pixel position
        c1 = TD->getTagCenter();
        // 3D point of c1 on cam1
        P1 = T1(2)*K1.inverse()*c1;
        c1 = K1*(C1/C1(2));
//        cout<<"K1\n"<<K1<<endl;
        MyFilledCircle(frame1, cv::Point(c1(0),c1(1)),Scalar(0,255,0 ));

    }
    // cam2 coordiante
    TD->isDetected(&FLAGS_TagID, &frame2, &detected2);
    if(detected2){
        c2.setZero();
        TD->setCamInfo(tagSize, fx2, fy2, cx2, cy2);
        // Rotation and translation from original function
        tie(R2, T2) = TD->getTagPose();
//        tie(R2, T2) = TD->getRelativeTransform();
//        cout<<"tag pose: \n"<< R2 <<"\n"<< T2<<endl;
//        cout<<"Pose using PnP: \n"<<R2PnP<<"\n"<<T2PnP<<endl;
        // translation using depth image
        C2 = TD->getTagToCam(frameD2);
        // center pixel position
        c2 = TD->getTagCenter();
        // 3D point of c2 on cam2
        P2 = T2(2)*K2.inverse()*c2;
        c2 = K2*(C2/C2(2));
//        cout<<"K2\n"<<K2<<endl;
        MyFilledCircle(frame2, cv::Point(c2(0),c2(1)),Scalar(0,0,255)); //red
    }
    if(detected1 && detected2) {
//        R12 = m2*m1.transpose(); // cam1 -> cam2
//        T12 = T2-m2*m1.transpose()*T1;
//        R12 = R2PnP.transpose()*R1PnP;
//        T12 = T1PnP - R1PnP.transpose()*R2PnP*T2PnP;
        R12 = R2.transpose() * R1;
//        T12 = T1 - R1.transpose() * R2 * T2;
        T12 = C1 - R1.transpose() * R2 * C2;
//        R21 = R1PnP.transpose()*R2PnP;
//        T21 = T2PnP - R2PnP.transpose()*R1PnP*T1PnP;

//        R21 = R1.transpose()*R2;
//        T21 = C2 - R2.transpose()*R1*C1;
//        R21 = R1.transpose()*R2;
//        T21 = T2 - R2.transpose()*R1*T1;
        RSum = RSum + R12;
        TSum = TSum + T12;
//        R12 = RSum/index;
//        T12 = TSum/index;
        index++;
        gotTrans = true;
    }
//        Vector3d P1Cam2 = R21.transpose()*T1 + T21;
//    Vector3d P1Cam2 = R12*(T1 - T12);
    Vector3d P1Cam2 = R12*(C1 - T12);
//    cout<<"difference of P1 at cam2: "<<(P1Cam2 - T2).norm()<<endl;
    auto tag1OnCam2 = K2*(P1Cam2/P1Cam2(2));
    // wrong re projection
    MyFilledCircle(frame2, cv::Point(tag1OnCam2(0),tag1OnCam2(1)),Scalar( 0, 255, 255 ));
//    Vector3d P2Cam1 = R12.transpose()*T2 + T12;
    Vector3d P2Cam1 = R12.transpose()*C2 + T12;
//        Vector3d P2Cam1 = R21*(T2 - T21);
//    cout<<"difference of P2 at cam1: "<<(P2Cam1 - T1).norm()<<endl;
    auto tag2OnCam1 = K1*(P2Cam1/P2Cam1(2));
    MyFilledCircle(frame1, cv::Point(tag2OnCam1(0),tag2OnCam1(1)),Scalar( 0, 255, 255 ));
}


inline void SkeletonMerger::processing(Mat *f1, Mat *f2, const Mat *fD1, const Mat *fD2, bool* skDetected) {
    *skDetected = false;
    setFrames(*f1, *f2, *fD1, *fD2);
//    Matrix3d PEstimated, PETranslated, PixelEstimated;
    Vector3d PEstimated, PETranslated, PixelEstimated;
    if(!gotTrans){
        updateRT();
    }
//    updateRT();

    // Skeleton prediction
    // from cam1
    bool noDisplay = true;
    thread t5(&OpWrapper::run, OW, &frame1, &frameD1, &fx1, &fy1, &cx1, &cy1, &noDisplay, &Sk1, &Acc1);
    thread t6(&OpWrapper::run, OW, &frame2, &frameD2, &fx2, &fy2, &cx2, &cy2, &noDisplay, &Sk2, &Acc2);
    t5.join();
    t6.join();

//    tie(Sk1, Acc1) = OW->getSkeleton();
//    tie(Sk2, Acc2) = OW->getSkeleton();
    if (Sk1.cols()==0 || Sk2.cols()==0){
        return;
    }
    *skDetected = true;
    auto PNeckTest = Sk2.col(1);
    auto PHipTest = Sk2.col(8);
    auto PShoulderTest = Sk2.col(2);

    auto ANeck = Acc2(1);
    auto AHip = Acc2(8);
    auto AShoulder = Acc2(2);
    // initialize TF
    if (ANeck>0.5 && AHip>0.5 && AShoulder>0.5 && countTF<5){
        PNeck = PNeckTest;
        PHip = PHipTest;
        PShoulder = PShoulderTest;
        Vector3d XHumanCam = PShoulder - PNeck;
        XHumanCam = XHumanCam/XHumanCam.norm();
        Vector3d ZHumanCam = PNeck - PHip;
        ZHumanCam = ZHumanCam/ZHumanCam.norm();
        Vector3d YHumanCam = ZHumanCam.cross(XHumanCam);
        cout<<"YHuman "<<YHumanCam.norm()<<endl;
        Vector3d XHuman(1,0,0);
        Vector3d YHuman(0,1,0);
        Vector3d ZHuman(0,0,1);
        MatrixXd B = XHuman*(XHumanCam.transpose()) + ZHuman*(ZHumanCam.transpose()) + YHuman*(YHumanCam.transpose()) ;
        JacobiSVD<MatrixXd> svd(B, ComputeThinU | ComputeThinV);
        Matrix3d M;
        const auto& U = svd.matrixU();
        const auto& V = svd.matrixV();
        M.diagonal()<< 1, 1, U.determinant()*V.determinant();
        RHumanCam = U*M*V.transpose();
        THumanCam = PNeck;
        countTF ++;
        cout<<"Rotation: \n"<<RHumanCam<<endl;
//        cout<<"Neck: \n"<<PNeck<<endl;
//        cout<<"Hip: \n"<<PHip<<endl;
//        cout<<"Shoulder: \n"<<PShoulder<<endl;
        cout<<"Translation: \n"<<THumanCam<<endl;

    }
    else if(countTF >= 5){
        transformed = true;
    }

    // translate sk1 from cam1 to cam2

    Sk1Cam2 = R12*(Sk1.colwise() - T12);
//    Sk1Cam2 = RHumanCam*(Sk1Cam2.colwise() - THumanCam);
//    Sk2 = RHumanCam*(Sk2.colwise()-THumanCam);
    for (int i=0; targetCount > i; i++) {
        // PTarget: P11, P21, P12, P22
        PTarget.middleCols(2*i,2) << Sk1Cam2.col(target[i+1]), Sk2.col(target[i+1]);
        AccTarget.middleRows(2*i,2) << Acc1(target[i+1]), Acc2(target[i+1]);
        PCam1.col(i) << Sk1.col(target[i+1]);
        PCam2.col(i) << Sk2.col(target[i+1]);
        Vector3d Ptest = Sk1Cam2.col(target[i+1]);
        Vector3d pTestCam2 = K2*(Ptest/Ptest(2));
        MyFilledCircle(frame2, cv::Point(pTestCam2(0),pTestCam2(1)),Scalar( 0, 255, 0 ));
        Vector3d Ptest2 = Sk1.col(target[i+1]);
        Vector3d pTest2Cam1 = K1*(Ptest2/Ptest2(2));
        MyFilledCircle(frame1, cv::Point(pTest2Cam1(0),pTest2Cam1(1)),Scalar( 0, 255, 0 ));

    }
//    cout<<"Accuracy: \n"<<AccTarget<<endl;
    cout<<"target joints: \n"<<PTarget<<endl;
    cout<<"accuracy: \n"<<AccTarget<<endl;
    L12.push_back((PTarget.col(3) - PTarget.col(1)).norm());
    L23.push_back((PTarget.col(5) - PTarget.col(3)).norm());
    cout<<"average measured length of elbow to wirst: "<<accumulate(L12.begin(),L12.end(),0.0)/L12.size()<<endl;
    cout<<"average measured length of shoulder to elbow: "<<accumulate(L23.begin(),L23.end(),0.0)/L23.size()<<endl;
    // GNC-MC
    // length: contrain; lambda: loose the contrain; stepSize: convergence
    GNC.setupGNC(PTarget, AccTarget, L1, Lambda1, stepSize, 3);
    GNC.run(true, index);
//        PEstimated = GNC.getEstimation();
        // P1, P2, P3 Vector3d
//    tie(P1, P2) = GNC.getTwoPointsEstimation();
    tie(P1, P2, P3) = GNC.getThreePointsEstimation();
//    cout<<"target point: \n"<<PTarget<<endl;
    cout<<"Estimated point 1: \n"<<P1<<endl;
    cout<<"Estimated point 2: \n"<<P2<<endl;
    cout<<"Estimated point 3: \n"<<P3<<endl;
    // Visulization
    // P1 on cam 1
    PETranslated = (R12.transpose()*P1) + T12;
    PixelEstimated = K1*(PETranslated/PETranslated(2));
    MyFilledCircle(frame1, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));
    // P1 on cam 2
    PixelEstimated = K2*(P1/P1(2));
    MyFilledCircle(frame2, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));

    // P2 on cam 1
    PETranslated = (R12.transpose()*P2) + T12;
    PixelEstimated = K1*(PETranslated/PETranslated(2));
    MyFilledCircle(frame1, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));
    // P2 on cam 2
    PixelEstimated = K2*(P2/P2(2));
    MyFilledCircle(frame2, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));

    // P3 on cam 1
    PETranslated = (R12.transpose()*P3) + T12;
    PixelEstimated = K1*(PETranslated/PETranslated(2));
    MyFilledCircle(frame1, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));
    // P3 on cam 2
    PixelEstimated = K2*(P3/P3(2));
    MyFilledCircle(frame2, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));

    // P21 on cam1
    auto PTest21 = PTarget.col(1);
    PETranslated = (R12.transpose()*PTest21) + T12;
    PixelEstimated = K1*(PETranslated/PETranslated(2));
//    MyFilledCircle(frame1, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar(255,255,0));
    // P22 on cam1
    auto PTest22 = PTarget.col(3);
    PETranslated = (R12.transpose()*PTest22) + T12;
    PixelEstimated = K1*(PETranslated/PETranslated(2));
//    MyFilledCircle(frame1, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar(255,255,0));

    // P11 on cam2
    auto PTest11 = PTarget.col(0);
//    auto PETranslated21 = R21.transpose()*PTest11 + T21;
    auto PETranslated21 = R12*(PTest11 - T12);
    auto PixelEstimated21 = K2*(PETranslated21/PETranslated21(2));
//    MyFilledCircle(frame2, cv::Point(PixelEstimated21(0),PixelEstimated21(1)),Scalar(255,0,0));
    // P12 on cam2
    auto PTest12 = PTarget.col(2);
//    PETranslated = R21.transpose()*PTest12 + T21;
    PETranslated = R12*(PTest12 - T12);
    PixelEstimated = K2*(PETranslated/PETranslated(2));
//    MyFilledCircle(frame2, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar(255,0,0));

    *f1 = frame1;
    *f2 = frame2;
}


inline tuple<Vector3d, Vector3d> SkeletonMerger::getEstimation() {
    return make_tuple(P1, P2);
}

inline tuple<Vector3d, Vector3d, Vector3d> SkeletonMerger::getThreeEstimation() {
    return make_tuple(P1, P2, P3);
}


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

rs2_stream findStreamToAlign(const vector<stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream alignTo = RS2_STREAM_ANY;
    bool isDepthStreamFound = false;
    bool isColorStreamFound = false;

    for (const stream_profile& sp : streams)
    {
        rs2_stream profileStream = sp.stream_type();
        if (profileStream != RS2_STREAM_DEPTH)
        {
            if (!isColorStreamFound) //Prefer color
                alignTo = profileStream;

            if (profileStream == RS2_STREAM_COLOR)
            {
                isColorStreamFound = true;
            }
        }
        else
        {
            isDepthStreamFound = true;
        }
    }

    if (!isDepthStreamFound)
        throw std::runtime_error("No Depth stream available");

    if (alignTo == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return alignTo;
}


float getDepthScale(const rs2::device& dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = rs2::depth_sensor(sensor))
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}


void getFrames(pipeline *pipe, const int *WIDTH, const int *HEIGHT, const float *depthScale,
               Mat *colorMat, Mat *depthMat){
    frameset frameset = pipe->wait_for_frames();
//    frameset *frameset;
//    pipe.poll_for_frames(frameset);
    //Align depth2color
    rs2_stream alignTo = findStreamToAlign(pipe->get_active_profile().get_streams());
    rs2::align aligned(alignTo);
    //Get processed aligned frame
    auto processed = aligned.process(frameset);

    // Trying to get both color and aligned depth frames
    rs2::video_frame colorFrame = processed.first(alignTo);
    rs2::depth_frame alignedDepthFrame = processed.get_depth_frame();
    rs2::depth_frame depthOriginal = frameset.get_depth_frame();
    if(!colorFrame || !alignedDepthFrame){
        cout<<"ERROR!"<<endl;
        exit(0);
    }
    Mat colorCVMat(Size(*WIDTH, *HEIGHT), CV_8UC3, (void*)colorFrame.get_data(), Mat::AUTO_STEP);
    Mat depthCVMat(Size(*WIDTH, *HEIGHT),CV_16UC1, (void*)alignedDepthFrame.get_data(), Mat::AUTO_STEP);
    Mat depthMatOriginal(Size(*WIDTH, *HEIGHT),CV_16UC1, (void*)depthOriginal.get_data(), Mat::AUTO_STEP);

    //For visualizing depth images
    rs2::colorizer c; // Helper to colorize depth images
    auto dpethColorizedFrame = c.colorize(alignedDepthFrame);
    Mat depthColorizedCVMat(Size(*WIDTH, *HEIGHT),CV_8UC3,
                            (void*)dpethColorizedFrame.get_data(), Mat::AUTO_STEP);

    *colorMat = colorCVMat.clone();
    *depthMat = depthCVMat.clone();
}

void saveImages(const string &saveDir, const string &saveName, std::shared_ptr<std::vector<std::shared_ptr<cv::Mat>>> &images_shared_ptr) {
    if (!images_shared_ptr->empty()){
        fs::path dispSaveDir = saveDir;
        if(!fs::exists(dispSaveDir)){
            fs::create_directories(dispSaveDir);
        }
        int numFrames = int(images_shared_ptr->size());
        for (int i=0; i<numFrames; i++)
        {
            std::shared_ptr<cv::Mat> cvMat = (*images_shared_ptr)[i];
            string imgName = dispSaveDir / saveName;
            imgName = imgName.append(format("_%d.png", i));
            cout<<"save name: "<< imgName << std::endl;
            imwrite(imgName, cvMat->clone());
        }
    }
}

int main(int argc, char** argv){
    cout<<"program starts\n";
    ofstream predictionTxt;
    int frameNumber = 0;

    if (!fs::is_directory(FLAGS_saveDir) || !fs::exists(FLAGS_saveDir))
    {
        fs::create_directory(FLAGS_saveDir);
    }

    predictionTxt.open(FLAGS_saveDir+"prediction.txt");
    // Skeleton merger
    SkeletonMerger SM;
    TagDetector TD;
    // target: size, joints
//    int target[] = {3,7,6,5};
    int target[] = {3,4,3,2};
    double tagSize = 0.166; // 0.215
    double constrainLength1 = 0.23, constrainLength2 = 0.27, lambda1 = 1000.0, lambda2 = 1000.0, stepSize = 0.01;

//    SM.setTarget(target, constrainLength1, lambda1, stepSize, tagSize);
    SM.setThreeTargets(target, constrainLength1, constrainLength2, lambda1, lambda2, stepSize, tagSize);
    int index = 1;
    vector<double> length1, length2;
    Mat frame1, frame2, frameD1, frameD2, result;
    Matrix3d RHumanCam;
    Vector3d THumanCam;

    if (!FLAGS_image and !FLAGS_demo){
        cout<<"video1: "<<FLAGS_video1<<endl;
        cout<<"video2: "<<FLAGS_video2<<endl;
        VideoCapture cap1(FLAGS_video1);
        VideoCapture cap2(FLAGS_video2);

        if (!cap1.isOpened() || !cap2.isOpened()){
            cout << "Error opening video file" << endl;
            return -1;
        }
        while(cap1.isOpened() && cap2.isOpened()){
            cap1.read(frame1);
            cap2.read(frame2);
            if (frame2.empty() || frame1.empty())
                break;

            string frameName = format("frame%d.png", index-1);
            string depthDir1 = FLAGS_depthDir1 + frameName;
            string depthDir2 = FLAGS_depthDir2 + frameName;
            frameD1 = imread(depthDir1, -1);
            frameD2 = imread(depthDir2, -1);
            cout<<"frame: "<<index<<endl;
            bool skDetected = false;
            SM.processing(&frame1, &frame2, &frameD1, &frameD2, &skDetected);
            if(!skDetected)
                continue;
            Vector3d P1, P2;
            tie(P1, P2) = SM.getEstimation();
            double l = (P1-P2).norm();
            length1.push_back(l);

            MyTextOnImage(frame1,"Camera 1: red (approximation)");
            MyTextOnImage(frame2,"Camera 2");
            hconcat(frame1, frame2, result);
            string saveDir = FLAGS_saveDir + format("output%d.png",index);
//            imwrite(saveDir, result);
            imshow("Left: Camera", result);
            index ++;
            int key = waitKey(1); // key is an integer here
            if (key == 27)
                break;
        }
    }
    else if(FLAGS_demo){
        context ctx;
        int WIDTH = 848, HEIGHT = 480;
        fs::path saveDir = "/home/geriatronics/hao/skeleton_fusion/results/";
        string colorSaveName = "color";
        string depthSaveName = "depth";
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
            depthScales.emplace_back(getDepthScale(profile.get_device()));
//            cout<<"serial: "<<serial<<endl;
//            cout<<"profile.get_device() "<<profile.get_device()<<endl;
        }
        // drop first N frames
        auto pipe1 = pipelines[0];
        auto pipe2 = pipelines[1];
        bool isCamReady1=false, isCamReady2=false;
        double fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2;
        thread t1(isCameraReady, &pipe1, &isCamReady1, &fx1, &fy1, &cx1, &cy1);
        thread t2(isCameraReady, &pipe2, &isCamReady2, &fx2, &fy2, &cx2, &cy2);
        t1.join();
        t2.join();
        SM.setCamParameters(fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2);

        if(!isCamReady1 || !isCamReady2){
             cout<<"at least one camera is not ready to receive frame!"<<endl;
             return 0;
        }
        std::shared_ptr<std::vector<std::shared_ptr<cv::Mat>>> colorSaveImages_ptr1 = std::make_shared<std::vector<std::shared_ptr<Mat>>>();
        std::shared_ptr<std::vector<std::shared_ptr<cv::Mat>>> depthSaveImages_ptr1 = std::make_shared<std::vector<std::shared_ptr<Mat>>>();
        std::shared_ptr<std::vector<std::shared_ptr<cv::Mat>>> colorSaveImages_ptr2 = std::make_shared<std::vector<std::shared_ptr<Mat>>>();
        std::shared_ptr<std::vector<std::shared_ptr<cv::Mat>>> depthSaveImages_ptr2 = std::make_shared<std::vector<std::shared_ptr<Mat>>>();
        std::shared_ptr<std::vector<std::shared_ptr<cv::Mat>>> colorSaveOriginal_ptr1 = std::make_shared<std::vector<std::shared_ptr<Mat>>>();
        std::shared_ptr<std::vector<std::shared_ptr<cv::Mat>>> colorSaveOriginal_ptr2 = std::make_shared<std::vector<std::shared_ptr<Mat>>>();


        while(isCamReady1 && isCamReady2){
            Mat colorMat1, colorMat2, depthMat1, depthMat2;

            thread t3(getFrames, &pipe1, &WIDTH, &HEIGHT, &depthScales[0], &colorMat1, &depthMat1);
            thread t4(getFrames, &pipe2, &WIDTH, &HEIGHT, &depthScales[0], &colorMat2, &depthMat2);
            t3.join();
            t4.join();
            Mat colorOriginal1 = colorMat1.clone();
            Mat colorOriginal2 = colorMat2.clone();
//            getFrames(&pipe1, &WIDTH, &HEIGHT, &depthScales[0], &colorMat1, &depthMat1);
//            cout<<"size2: "<<depthMat2.rows<< " "<<depthMat2.cols<<endl;
//            cout<<"size1: "<<depthMat1.rows<< " "<<depthMat1.cols<<endl;
//            imshow("depth1: ", depthMat1);
//            imshow("depth2: ", depthMat2);


//            Mat depthresult;
//            hconcat(depthMat1, depthMat2, depthresult);
//            imshow("Two Depth Cameras", depthresult);
//            Mat colorResult;
//            hconcat(colorMat1, colorMat2, colorResult);
//            imshow("Two Color Cameras", colorResult);

            bool skDetected = false;
            SM.processing(&colorMat1, &colorMat2, &depthMat1, &depthMat2, &skDetected);
//            if(!skDetected)
//                continue;
            Mat colorResult;
            MyTextOnImage(colorMat1,"Camera 1: red (approximation)");
            MyTextOnImage(colorMat2,"Camera 2");
            hconcat(colorMat1, colorMat2, colorResult);
            imshow("Two Color Cameras", colorResult);

            Vector3d P1, P2, P3;
//            tie(P1, P2) = SM.getEstimation();
            tie(P1, P2, P3) = SM.getThreeEstimation();
            bool isTransformed = SM.isTransformed();
            double l1 = (P1-P2).norm();
            double l2 = (P2-P3).norm();
            cout<<"predicted length1 : "<<l1<<" length2: "<< l2 <<endl;
            length1.push_back(l1);

//            predictionTxt << P3(0) << " " << P3(1) << " " << P3(2)
//                          << " " << P2(0) << " " <<P2(1) << " " <<P2(2)
//                          << " " << P1(0) << " " <<P1(1) << " " <<P1(2)<<'\n';
            if(isTransformed){

                tie(RHumanCam, THumanCam) = SM.getHumanCamTF();
                P3 = RHumanCam*(P3 - THumanCam);
                P2 = RHumanCam*(P2 - THumanCam);
                P1 = RHumanCam*(P1 - THumanCam);

                cout <<" "<< P3(0) << " " << P3(1) << " " << P3(2)
                             << " " << P2(0) << " " <<P2(1) << " " <<P2(2)
                             << " " << P1(0) << " " <<P1(1) << " " <<P1(2)<<'\n';
            }

            if(FLAGS_save_images){
                std::shared_ptr<Mat> cSaveImagePtr1 = std::make_shared<Mat>(colorMat1.clone());
                std::shared_ptr<Mat> dSaveImagePtr1 = std::make_shared<Mat>(depthMat1.clone());
                std::shared_ptr<Mat> cSaveImagePtr2 = std::make_shared<Mat>(colorMat2.clone());
                std::shared_ptr<Mat> dSaveImagePtr2 = std::make_shared<Mat>(depthMat2.clone());
                std::shared_ptr<Mat> cSaveOriginalPtr1 = std::make_shared<Mat>(colorOriginal1.clone());
                std::shared_ptr<Mat> cSaveOriginalPtr2 = std::make_shared<Mat>(colorOriginal2.clone());

                colorSaveImages_ptr1->push_back(cSaveImagePtr1);
                depthSaveImages_ptr1->push_back(dSaveImagePtr1);
                colorSaveImages_ptr2->push_back(cSaveImagePtr2);
                depthSaveImages_ptr2->push_back(dSaveImagePtr2);
                colorSaveOriginal_ptr1->push_back(cSaveOriginalPtr1);
                colorSaveOriginal_ptr2->push_back(cSaveOriginalPtr2);
            }
            int key = waitKey(1); // key is an integer here
            if (key == 27)
                break;
//            else if(key == 32)

        }
        if(FLAGS_save_images){
            string colorSaveDir1 = saveDir / "color1";
            string depthSaveDir1 = saveDir / "depth1";
            string colorSaveDir2 = saveDir / "color2";
            string depthSaveDir2 = saveDir / "depth2";
            string colorOriginalDir1 = saveDir / "original1";
            string colorOriginalDir2 = saveDir / "original2";
            saveImages(colorSaveDir1, colorSaveName, colorSaveImages_ptr1);
            saveImages(depthSaveDir1, depthSaveName, depthSaveImages_ptr1);
            saveImages(colorSaveDir2, colorSaveName, colorSaveImages_ptr2);
            saveImages(depthSaveDir2, depthSaveName, depthSaveImages_ptr2);
            saveImages(colorOriginalDir1, colorSaveName, colorSaveOriginal_ptr1);
            saveImages(colorOriginalDir2, colorSaveName, colorSaveOriginal_ptr2);
        }
    }
    else{
        std::vector<fs::path> filenames;
        for (const auto& entry : fs::directory_iterator{FLAGS_colorDir1}) {
            filenames.push_back(entry.path().filename());
            frameNumber ++;
        }
        cout<<"frameNumber: "<<frameNumber<<endl;
        predictionTxt << frameNumber<<" " << target[0]*3 + 1 <<'\n';
        std::sort(filenames.begin(), filenames.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.string() < rhs.string();
                  });
        for (const auto& file : filenames) {
            string img = file.string();
            string stamp = file.string().substr(4,7);
            cout<<"stamp: "<< stamp <<endl;
            string imgD = img;
            imgD.insert(3,"D");
            cout<<"img: "<<img<<endl;
            cout<<"imgD: "<<imgD<<endl;
            string imgDir1 = FLAGS_colorDir1 + img;
            string imgDir2 = FLAGS_colorDir2 + img;
            string imgDepthDir1 = FLAGS_depthDir1 + imgD;
            string imgDepthDir2 = FLAGS_depthDir2 + imgD;

            frame1 = imread(imgDir1, -1);
            frame2 = imread(imgDir2, -1);
            frameD1 = imread(imgDepthDir1, -1);
            frameD2 = imread(imgDepthDir2, -1);

            if (frame2.empty() || frame1.empty() || frameD2.empty() || frameD1.empty())
                break;
            bool skDetected = false;
            SM.processing(&frame1, &frame2, &frameD1, &frameD2, &skDetected);
            if(!skDetected)
                continue;
            Vector3d P1, P2, P3;
//            tie(P1, P2) = SM.getEstimation();
            tie(P1, P2, P3) = SM.getThreeEstimation();
            bool isTransformed = SM.isTransformed();
            double l1 = (P1-P2).norm();
            double l2 = (P2-P3).norm();
            cout<<"predicted length1 : "<<l1<<" length2: "<< l2 <<endl;
            length1.push_back(l1);

//            predictionTxt << P3(0) << " " << P3(1) << " " << P3(2)
//                          << " " << P2(0) << " " <<P2(1) << " " <<P2(2)
//                          << " " << P1(0) << " " <<P1(1) << " " <<P1(2)<<'\n';
            if(isTransformed){

                tie(RHumanCam, THumanCam) = SM.getHumanCamTF();
                P3 = RHumanCam*(P3 - THumanCam);
                P2 = RHumanCam*(P2 - THumanCam);
                P1 = RHumanCam*(P1 - THumanCam);

                predictionTxt<< stamp <<" "<< P3(0) << " " << P3(1) << " " << P3(2)
                          << " " << P2(0) << " " <<P2(1) << " " <<P2(2)
                          << " " << P1(0) << " " <<P1(1) << " " <<P1(2)<<'\n';
//                predictionTxt << stamp << " " <<-P3(0) << " " << -P3(2) << " " << - P3(1)
//                              << " " << -P2(0) << " " << -P2(2) << " " << - P2(1)
//                              << " " << -P1(0)<< " " << -P1(2) << " " << -P1(1) <<'\n';
            }



            MyTextOnImage(frame1,"Camera 1: red (approximation)");
            MyTextOnImage(frame2,"Camera 2");
            hconcat(frame1, frame2, result);
            string saveDir = FLAGS_saveDir + "output_" + stamp + ".png";
            cout<<"savedir: "<<saveDir<<endl;
            imwrite(saveDir, result);
            imshow("Left and right: Camera", result);
            index ++;
            int key = waitKey(1); // key is an integer here
            if (key == 27)
                break;
        }
    }
    predictionTxt.close();
    if(!FLAGS_demo){
        vector<double> constrain1 (frameNumber, constrainLength1);
        plt::named_hist("Constrain length L_ew", constrain1);
        plt::named_hist("Estimated length", length1);
        plt::legend();
        string imgDir = format("/media/dataset/translation/skeleton_fusion/results/cost/Length_over_time.png");
        plt::save(imgDir);
        plt::show();
        plt::close();
    }
    return 0;
}