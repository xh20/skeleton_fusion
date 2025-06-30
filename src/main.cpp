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
//#include <experimental/filesystem>
#include <map>
#include <eigen3/Eigen/Dense>
// #include <OsqpEigen/OsqpEigen.h>
#include <librealsense2/rs.hpp>

using namespace std;
using namespace rs2;
namespace fs = std::experimental::filesystem;

DEFINE_string(video1, "/media/dataset/translation/S00V00A16O00T1/colorframes/out.mp4",
              "Give the first video file path");
DEFINE_string(video2, "/media/dataset/translation/S00V01A16O00T1/colorframes/out.mp4",
              "Give the second video file path");
DEFINE_bool(image, true, "is input as image");
DEFINE_bool(demo, false, "is input as camera");
DEFINE_bool(save_images, false, "save all images");


// Tag size
//DEFINE_string(colorDir1, "/media/dataset/translation/results_medical2/001622070717/color/",  // results_dailytask results_medical
//              "Give the first video file path");
//DEFINE_string(colorDir2, "/media/dataset/translation/results_medical2/845112070795/color/", // results_test results_daily1
//              "Give the second video file path");
//DEFINE_string(depthDir1, "/media/dataset/translation/results_medical2/001622070717/depth/",
//              "Give the first video file path");
//DEFINE_string(depthDir2, "/media/dataset/translation/results_medical2/845112070795/depth/",
//              "Give the second video file path");
DEFINE_string(colorDir1, "/mnt/data_base/skeleton_fusion/records/push_side/push_side_2/142122070979/original/",  // results_dailytask results_medical
              "Give the first video file path");
DEFINE_string(colorDir2, "/mnt/data_base/skeleton_fusion/records/push_side/push_side_2/845112070795/original/", // results_test results_daily1
              "Give the second video file path");
DEFINE_string(depthDir1, "/mnt/data_base/skeleton_fusion/records/push_side/push_side_2/142122070979/depth/",
              "Give the first video file path");
DEFINE_string(depthDir2, "/mnt/data_base/skeleton_fusion/records/push_side/push_side_2/845112070795/depth2/",
              "Give the second video file path");

DEFINE_string(cameraParameters, "/mnt/data_base/skeleton_fusion/records/push_side/push_side_2/cam_parameters.txt",
              "Give the camera parameters text file path");
DEFINE_string(tagCode, "36h11", "Give the tag code to choose tag family");
DEFINE_int32(TagID, 29, "Give the Tag ID to calculate the extrinsic parameters, hand marker: 89, table left: 54, table right: 25");
DEFINE_string(saveDir, "/mnt/data_base/skeleton_fusion/records/push_side/push_side_2/results/",
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
    double cx1, cy1, fx1, fy1, cx2, cy2, fx2, fy2, tagSize, L1, L2, Lambda1, Lambda2, stepSize, upperArmLength, lowerArmLength;
    Mat frame1, frame2, frameD1, frameD2;
    VectorXd Acc1, Acc2, AccTarget;
    MatrixXd Sk1, Sk2, Sk1Cam2, Sk2Cam1, PTarget, PCam1, PCam2;
    Matrix3d RSum, R12, K1, K2, R21, RHumanCam;
    Vector3d TSum, T12, T21, P1, P2, P3, THumanCam, PNeck, PHip, YShoulderOld, XShoulderF0, YShoulderF0, ZShoulderF0;
    Vector3d XCam, YCam, ZCam;
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
    void processing(Mat* f1, Mat* f2, const Mat* fD1, const Mat* fD2, bool*, const bool*);
    tuple<Vector3d, Vector3d> getEstimation();
    tuple<Vector3d, Vector3d, Vector3d> getThreeEstimation();
    [[nodiscard]] bool isTransformed() const;
    tuple<Matrix3d, Vector3d> getHumanCamTF();
    double getAngleOfTwoVectors(const Vector3d&, const Vector3d&);

    Vector3d angles_f0f1, angles_f0f1_old, ZBase, YBase, XBase, angles_ShoulderBase, PShoulder, angles_SB_old;
    Matrix3d RotationBaseShoulder, RShoulderF0F1, RShoulderF0Cam;
    double angleArm, angleArmTorso, angleArmX, angleArmY, angleShoulderTorso, angleShoulderX, angleZ, thetaX, thetaY, thetaZ, thetaZ_;
};

#include <map>

// "001622070717", {912.036, 910.78, 635.183, 366.158}
// "845112070795", {915.888, 915.828, 631.489, 372.897}
// "cam_1", {919.0277709960938, 0.0, 647.3279418945312, 0.0, 917.8308715820312, 368.21026611328125, 0.0, 0.0, 1.0}
// "cam_2", {921.8261108398438, 0.0, 641.4801635742188, 0.0, 921.8929443359375, 375.79901123046875, 0.0, 0.0, 1.0}
// 142122070979 605.05 605.233 425.866 236.747
// 845112070795 610.592 610.552 418.326 248.598


inline SkeletonMerger::SkeletonMerger():cx1(425.866),cy1(236.747),fx1(605.05),fy1(605.233),
cx2(418.326),cy2(248.598),fx2(610.592),fy2(610.552), gotTrans(false),
tagSize(0.1660),index(1), countTF(0), upperArmLength(0.0), lowerArmLength(0.0),
angleArm(0.0), angleArmTorso(0.0), angleArmX(0.0), angleArmY(0.0){
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
//    return make_tuple(RHumanCam, THumanCam);
    return make_tuple(RHumanCam.transpose(), THumanCam);
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

// return rotation and translation between cameras
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
        MyFilledCircle(frame1, cv::Point(int(c1(0)),int(c1(1))),Scalar(0,255,0 ));

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
        MyFilledCircle(frame2, cv::Point(int(c2(0)),int(c2(1))),Scalar(0,0,255)); //red
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
    MyFilledCircle(frame2, cv::Point(int(tag1OnCam2(0)),int(tag1OnCam2(1))),Scalar( 0, 255, 255 ));
//    Vector3d P2Cam1 = R12.transpose()*T2 + T12;
    Vector3d P2Cam1 = R12.transpose()*C2 + T12;
//        Vector3d P2Cam1 = R21*(T2 - T21);
//    cout<<"difference of P2 at cam1: "<<(P2Cam1 - T1).norm()<<endl;
    auto tag2OnCam1 = K1*(P2Cam1/P2Cam1(2));
    MyFilledCircle(frame1, cv::Point(int(tag2OnCam1(0)),int(tag2OnCam1(1))),Scalar( 0, 255, 255 ));
}

inline double SkeletonMerger::getAngleOfTwoVectors(const Vector3d& v1, const Vector3d& v2){
    auto dot = v1.dot(v2);
    auto det = v1.norm()*v2.norm();
    auto angle = acos(dot/det)*180/M_PI;
//    auto angle = atan2(det, dot)*180/M_PI;
    return angle;
}

inline void SkeletonMerger::processing(Mat *f1, Mat *f2, const Mat *fD1, const Mat *fD2, bool* skDetected,
                                       const bool* isDisplay) {
    *skDetected = false;
    setFrames(*f1, *f2, *fD1, *fD2);

//    Matrix3d PEstimated, PETranslated, PixelEstimated;
    Vector3d PEstimated, PETranslated, PixelEstimated;
    if (!gotTrans) {
        updateRT();
    }
//    updateRT();

    // 3D Skeleton prediction (Sk1, Sk2)
    bool noDisplay = true;
    OW->run(f1, fD1, &fx1, &fy1, &cx1, &cy1, &noDisplay, &Sk1, &Acc1);
    OW->run(f2, fD2, &fx2, &fy2, &cx2, &cy2, &noDisplay, &Sk2, &Acc2);
//    cout<<"K1\n"<<K1<<endl;
//    cout<<"K2\n"<<K2<<endl;

//    thread t5(&OpWrapper::run, OW, f1, fD1, &fx1, &fy1, &cx1, &cy1, &noDisplay, &Sk1, &Acc1);
//    thread t6(&OpWrapper::run, OW, f2, fD2, &fx2, &fy2, &cx2, &cy2, &noDisplay, &Sk2, &Acc2);
//    t5.join();
//    t6.join();

    if (Sk1.cols() == 0 || Sk2.cols() == 0) {
        return;
    }
    *skDetected = true;

    auto ANeck = Acc2(1);
    auto AHip = Acc2(8);
    auto AShoulder = Acc2(2);
    // initialize TF and limbs length Base Frame
    if (countTF < 5) {
        PNeck = Sk2.col(1);
        PHip = Sk2.col(8);
        PShoulder = Sk2.col(2);
        YShoulderOld = PNeck - PShoulder;
        // Base frame
        ZBase = PNeck - PHip;
        ZBase = ZBase / ZBase.norm();
        YBase = PNeck - PShoulder;
        YBase = YBase / YBase.norm();
        XBase = YBase.cross(ZBase);
        YBase = ZBase.cross(XBase);
        YBase = YBase / YBase.norm();

        XCam = Eigen::Vector3d::UnitX();
        YCam = Eigen::Vector3d::UnitY();
        ZCam = Eigen::Vector3d::UnitZ();

        /*** For SVD: C = A*B_T, the rotation matrix is map B to A (A = R*B)
         ***/
//        Vector3d XTest(0,0,-1);
//        Vector3d YTest(1, 0, 0);
//        Vector3d ZTest(0,-1,0);
//////        MatrixXd BTest = XCam*(XTest.transpose()) + ZCam*(ZTest.transpose()) + YCam*(YTest.transpose()) ;
//        MatrixXd BTest = XTest*(XCam.transpose()) + ZTest*(ZCam.transpose()) + YTest*(YCam.transpose()) ;
//        JacobiSVD<MatrixXd> svdTest(BTest, ComputeFullU | ComputeFullV);
//        const auto& UTest = svdTest.matrixU();
//        const auto& VTest = svdTest.matrixV();
//        Matrix3d RTest = UTest*VTest.transpose();
//        // index: 0, 1, 2 -> rotation order: x,y,z
//        // index: 2, 1, 0 -> rotation order: z,y,x
//        Vector3d anglesTest = 180/M_PI*RTest.eulerAngles(0, 1, 2);
//        Vector3d xTest = RTest*XCam;
//        Vector3d yTest = RTest*YCam;
//        Vector3d zTest = RTest*ZCam;
//        cout<<"rotated xTest: "<<xTest<<endl;
//        cout<<"original XTest: "<<XTest<<endl;
//        cout<<"rotated yTest: "<<yTest<<endl;
//        cout<<"original YTest: "<<YTest<<endl;
//        cout<<"rotated zTest: "<<zTest<<endl;
//        cout<<"original ZTest: "<<ZTest<<endl;
//        Vector3d xNew = RTest.transpose()*XCam;
//        Vector3d yNew = RTest.transpose()*YCam;
//        Vector3d zNew = RTest.transpose()*ZCam;
//        cout<<"new xNew: "<<xNew<<endl;
//        cout<<"new yNew: "<<yNew<<endl;
//        cout<<"new zNew: "<<zNew<<endl;
    //    cout<< "angles test: "<<anglesTest[0]<<" "<<anglesTest[1]<<" "<<anglesTest[2]<<endl;

        // Base= R*Cam
        MatrixXd B = XBase * (XCam.transpose()) + ZBase * (ZCam.transpose()) + YBase * (YCam.transpose());
        // Cam =R*Base, a Point A at cam frame, A_base = R*A
//        MatrixXd B = XCam * (XBase.transpose()) + ZCam * (ZBase.transpose()) + YCam * (YBase.transpose());
        JacobiSVD<MatrixXd> svd(B, ComputeFullU | ComputeFullV);
        Matrix3d M;
        const auto &U = svd.matrixU();
        const auto &V = svd.matrixV();
    //    M.diagonal()<< 1, 1, U.determinant()*V.determinant();
        RHumanCam = U * V.transpose();
        THumanCam = PNeck;
//        Vector3d anglesBC = 180 / M_PI * RHumanCam.eulerAngles(0, 1, 2);
    //    cout<<"Base -> Came XYZ: "<<angleBCX<< " " <<angleBCY <<" "<<angleBCZ <<endl;
    //    cout<<"Base -> Came eulerAngles XYZ: "<<anglesBC.transpose()<<endl;
        transformed = true;
}
    // get limbs length
    auto PWrist2 = Sk2.col(4);
    auto PElbow2 = Sk2.col(3);
    auto PShoulder2 = Sk2.col(2);
    auto PWrist1 = Sk1.col(4);
    auto PElbow1 = Sk1.col(3);
    auto PShoulder1 = Sk1.col(2);
    double upperL1 = (PShoulder1 - PElbow1).norm();
    double lowerL1 = (PWrist1 - PElbow1).norm();
    double upperL2 = (PShoulder2 - PElbow2).norm();
    double lowerL2 = (PWrist2 - PElbow2).norm();
//    cout<<"upper limb length: "<<upperL1<<" "<<upperL2<<endl;
//    cout<<"lower limb length: "<<lowerL1<<" "<<lowerL2<<endl;


    // translate sk1 from cam1 to cam2
    Sk1Cam2 = R12*(Sk1.colwise() - T12);
    // select target joints and plot test point on both frames
    for (int i=0; targetCount > i; i++) {
//        cout<<"tagets: "<<target[i+1]<<endl;
        // PTarget: P11, P21 (shoulder) P12, P22 (elbow), P13, P23 (wrist)
        PTarget.middleCols(2*i,2) << Sk1Cam2.col(target[i+1]), Sk2.col(target[i+1]);
        AccTarget.middleRows(2*i,2) << Acc1(target[i+1]), Acc2(target[i+1]);
        PCam1.col(i) << Sk1.col(target[i+1]);
        PCam2.col(i) << Sk2.col(target[i+1]);
        Vector3d Ptest = Sk1Cam2.col(target[i+1]);
        Vector3d pTestCam2 = K2*(Ptest/Ptest(2));
        MyFilledCircle(*f2, cv::Point(int(pTestCam2(0)),int(pTestCam2(1))),Scalar( 0, 255, 0 ));
        Vector3d Ptest1 = Sk1.col(target[i+1]);
        Vector3d pTestCam1 = K1*(Ptest1/Ptest1(2));
        MyFilledCircle(*f1, cv::Point(int(pTestCam1(0)),int(pTestCam1(1))),Scalar( 0, 255, 0 ));
    }

    L12.push_back((PTarget.col(3) - PTarget.col(1)).norm());
    L23.push_back((PTarget.col(5) - PTarget.col(3)).norm());

    // GNC-MC
    // length: contrain; lambda: loose the contrain; stepSize: convergence
    GNC.setupGNC(PTarget, AccTarget, L1, Lambda1, stepSize, 3);
    GNC.run(true, index);
//        PEstimated = GNC.getEstimation();
    // Get arm three points: P1 wrist, P2 elbow, P3 shoulder Vector3d
    tie(P1, P2, P3) = GNC.getThreePointsEstimation();
//    cout<<"Estimated limb length: "<<(P1-P2).norm()<<" "<<(P2-P3).norm()<<endl;
//    cout<<"target point: \n"<<target[0] << " "<<target[1] <<" " <<target[2]<< " " <<target[3]<<endl;

    // Get angles
    // Shoulder frame
//        Vector3d PShoulderCam2 = Sk2.col(2);
//        Vector3d PElbowCam2 = Sk2.col(3);
//        Vector3d PWristCam2 = Sk2.col(4);
    Vector3d PShoulderCam2 = P3;
    Vector3d PElbowCam2 = P2;
    Vector3d PWristCam2 = P1;
    Vector3d ZShoulder = PShoulderCam2 - PElbowCam2;
    Vector3d lowerArm = PWristCam2 - PElbowCam2;
    ZShoulder = ZShoulder/ZShoulder.norm();
    lowerArm = lowerArm/lowerArm.norm();
    Vector3d XShoulder = lowerArm.cross(ZShoulder);
    angleArm = getAngleOfTwoVectors(ZShoulder, lowerArm);
//    auto angleShoulderOld = getAngleOfTwoVectors(YShoulder, YShoulderOld);
//    auto angleShoulderNew = getAngleOfTwoVectors(-YShoulder, YShoulderOld);
//    if(angleShoulderNew < angleShoulderOld) {
//        YShoulder = -YShoulder;
//    }
//    else{
//        YShoulderOld = YShoulder;
//    }
    XShoulder = XShoulder/XShoulder.norm();
//    Vector3d XShoulder = YShoulder.cross(ZShoulder);
//    XShoulder = XShoulder/XShoulder.norm();

    Vector3d YShoulder = ZShoulder.cross(XShoulder);
    YShoulder = YShoulder/YShoulder.norm();

    if (countTF<=5){
        YShoulderF0 = YShoulder;
        ZShoulderF0 = ZShoulder;
        XShoulderF0 = XShoulder;
    }

//    auto XF0 = RShoulderF0Cam.transpose()*XShoulderF0;
//    auto YF0 = RShoulderF0Cam.transpose()*YShoulderF0;
//    auto ZF0 = RShoulderF0Cam.transpose()*ZShoulderF0;
//
//    auto XF1 = RShoulderF0Cam.transpose()*XShoulder;
//    auto YF1 = RShoulderF0Cam.transpose()*YShoulder;
//    auto ZF1 = RShoulderF0Cam.transpose()*ZShoulder;

//    cout<<"X Shoulder in Frame 0: "<<XF1<<endl;
//    cout<<"Y Shoulder in Frame 0: "<<YF1<<endl;
//    cout<<"Z Shoulder in Frame 0: "<<ZF1<<endl;

//    MatrixXd F0F1 = XF1*(XF0.transpose()) + YF1*(YF0.transpose()) + ZF1*(ZF0.transpose());
////    MatrixXd F0F1 = XF0*(XF1.transpose()) + YF0*(YF1.transpose()) + ZF0*(ZF1.transpose());
//    JacobiSVD<MatrixXd> svd_f0f1(F0F1, ComputeFullU | ComputeFullV);
//    const auto& U_f0f1 = svd_f0f1.matrixU();
//    const auto& V_f0f1 = svd_f0f1.matrixV();
//    RShoulderF0F1 = U_f0f1*V_f0f1.transpose();
//    //0:roll(X), 1: pitch (Y), 2:yaw (Z)
//    angles_f0f1 = Matrix3d(RShoulderF0F1).eulerAngles(0, 1, 2)*180/M_PI;
//    angles_f0f1 = Matrix3d(RShoulderF0F1).eulerAngles(2, 1, 0)*180/M_PI;
//    cout<<"F0_X: "<<XF0.transpose()<<endl;
//    cout<<"F1_X: "<<XF1.transpose()<<endl;
//    cout<<"R*F0_X"<<(RShoulderF0F1*XF0).transpose()<<endl;
//    cout<<"R*F0_X"<<(XF0.transpose()*RShoulderF0F1)<<endl;
//    if(countTF < 1){
//        angles_SB_old = angles_f0f1;
//    }
//    else{
//        auto angles_disc = (angles_f0f1 - angles_SB_old).cwiseAbs(); //array().abs()
//        cout<<angles_disc<<endl;
//        if((angles_disc[0]>180)){
//            angles_f0f1[0] += (angles_SB_old.array().sign()[0])*360.0;
//        }
//        else if((angles_disc[1]>180)){
//            angles_f0f1[1] += (angles_SB_old.array().sign()[1])*360.0;
//        }
//        else if((angles_disc[2]>180)){
//            angles_f0f1[2] += (angles_SB_old.array().sign()[2])*360.0;
//        }
//        angles_SB_old = angles_ShoulderBase;
//    }
    auto XBase_b = RHumanCam.transpose()*XBase;
    auto YBase_b = RHumanCam.transpose()*YBase;
    auto ZBase_b = RHumanCam.transpose()*ZBase;
    cout<<"XBase_b: "<<XBase_b.transpose()<<endl;
    cout<<"YBase_b: "<<YBase_b.transpose()<<endl;
    cout<<"ZBase_b: "<<ZBase_b.transpose()<<endl;

    auto XF1_b = RHumanCam.transpose()*XShoulder;
    auto YF1_b = RHumanCam.transpose()*YShoulder;
    auto ZF1_b = RHumanCam.transpose()*ZShoulder;

    cout<<"XF1_b: "<<XF1_b.transpose()<<endl;
    cout<<"YF1_b: "<<YF1_b.transpose()<<endl;
    cout<<"ZF1_b: "<<ZF1_b.transpose()<<endl;

    // Shoulder = R*Base -> S*B_T = R(B*B_T) -> (B*B_T)^(-1)*S*B_T = R
    MatrixXd BS = XF1_b*(XBase_b.transpose()) + YF1_b*(YBase_b.transpose()) + ZF1_b*(ZBase_b.transpose());
    JacobiSVD<MatrixXd> svd_BS(BS, ComputeFullU | ComputeFullV);
//    Matrix3d M_BS;
    const auto& U_BS = svd_BS.matrixU();
    const auto& V_BS = svd_BS.matrixV();
//    M.diagonal()<< 1, 1, U.determinant()*V.determinant();
//    auto RShoudlerBase = U_BS*M*V.transpose();
    RotationBaseShoulder = U_BS*V_BS.transpose();
    cout<<"R determinant: " <<RotationBaseShoulder.determinant()<<endl;
//    thetaX = 180/M_PI*atan2(RotationBaseShoulder(2,1), RotationBaseShoulder(2,2));
//    thetaY = 180/M_PI*atan2(-RotationBaseShoulder(2,0),
//                            sqrt(pow(RotationBaseShoulder(2,1),2) + pow(RotationBaseShoulder(2,2),2)));
//    thetaZ = 180/M_PI*atan2(RotationBaseShoulder(1,0), RotationBaseShoulder(0,0));
    angles_ShoulderBase = 180/M_PI*Matrix3d(RotationBaseShoulder).eulerAngles(0, 1, 2);
//    if(countTF < 1){
//        angles_SB_old = angles_ShoulderBase;
//    }
//    else{
//        auto angles_disc = (angles_ShoulderBase - angles_SB_old).cwiseAbs(); //array().abs()
//        cout<<angles_disc<<endl;
//        if((angles_disc[0]>180)){
//            angles_ShoulderBase[0] += (angles_SB_old.array().sign()[0])*360.0;
//        }
//        else if((angles_disc[1]>180)){
//            angles_ShoulderBase[1] += (angles_SB_old.array().sign()[1])*360.0;
//        }
//        else if((angles_disc[2]>180)){
//            angles_ShoulderBase[2] += (angles_SB_old.array().sign()[2])*360.0;
//        }
//        angles_SB_old = angles_ShoulderBase;
//    }
//    cout<<"angles from atan2: "<< thetaX <<" "<<thetaY <<" "<<thetaZ<<endl;
//    cout<<"roation matrix: \n"<<RShoulderF0F1<<endl;
//    cout<<"angles X Y Z: "<<angles_f0f1[2]<<" "<<angles_f0f1[1]<<" "<<angles_f0f1[0]<<endl;

//    cout<<"roation matrix: \n"<<RotationBaseShoulder<<endl;
    cout<<"angles X Y Z: "<<angles_ShoulderBase.transpose()<<endl;
//    cout<<"angle flexion: " << angleArm<<endl;

    countTF ++;

    // Visulization
    if(*isDisplay){
        // P1 on cam 1 Wrist
        PETranslated = (R12.transpose()*P1) + T12;
        PixelEstimated = K1*(PETranslated/PETranslated(2));
        MyFilledCircle(*f1, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));
        // P1 on cam 2
        PixelEstimated = K2*(P1/P1(2));
        MyFilledCircle(*f2, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));

        // P2 on cam 1 Elbow
        PETranslated = (R12.transpose()*P2) + T12;
        PixelEstimated = K1*(PETranslated/PETranslated(2));
        MyFilledCircle(*f1, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));
        // P2 on cam 2
        PixelEstimated = K2*(P2/P2(2));
        MyFilledCircle(*f2, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));

        // P3 on cam 1 Shoulder
        PETranslated = (R12.transpose()*P3) + T12;
        PixelEstimated = K1*(PETranslated/PETranslated(2));
        MyFilledCircle(*f1, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));
        // P3 on cam 2
        PixelEstimated = K2*(P3/P3(2));
        MyFilledCircle(*f2, cv::Point(PixelEstimated(0),PixelEstimated(1)),Scalar( 0, 0, 255 ));

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

        // Drawing coordinates
        // Shoulder frame
        double lineLength = 0.3;
        PETranslated= (R12.transpose()*P2) + T12;
        PixelEstimated = K1*(PETranslated/PETranslated(2));
        // Y axis on cam2
        Vector3d LineEnd = lineLength*YShoulder + P2;
        Vector3d PixelLineEnd = K2*(LineEnd/LineEnd(2));
        Vector3d PixelLineStart = K2*(P2/P2(2));
        line(*f2, cv::Point (int(PixelLineStart[0]), int(PixelLineStart[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(255, 0, 0), 5, LINE_8);
        // Y axis on cam1
        LineEnd = (R12.transpose()*LineEnd) + T12;
        PixelLineEnd = K1*(LineEnd/LineEnd(2));
        line(*f1, cv::Point (int(PixelEstimated[0]), int(PixelEstimated[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(255, 0, 0), 5, LINE_8);
        // Z axis on cam2
        LineEnd = lineLength*ZShoulder + P2;
        PixelLineEnd = K2*(LineEnd/LineEnd(2));
        line(*f2, cv::Point (int(PixelLineStart[0]), int(PixelLineStart[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(0, 255, 0), 5, LINE_8);
        // Z axis on cam1
        LineEnd = (R12.transpose()*LineEnd) + T12;
        PixelLineEnd = K1*(LineEnd/LineEnd(2));
        line(*f1, cv::Point (int(PixelEstimated[0]), int(PixelEstimated[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(0, 255, 0), 5, LINE_8);

        // X axis on cam2
        LineEnd = lineLength*XShoulder + P2;
        PixelLineEnd = K1*(LineEnd/LineEnd(2));
        line(*f2, cv::Point (int(PixelLineStart[0]), int(PixelLineStart[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(0, 0, 255), 5, LINE_8);
        // X axis on cam1
        LineEnd = (R12.transpose()*LineEnd) + T12;
        PixelLineEnd = K1*(LineEnd/LineEnd(2));
        line(*f1, cv::Point (int(PixelEstimated[0]), int(PixelEstimated[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(0, 0, 255), 5, LINE_8);

        // Base frame
        lineLength = 0.2;
        auto PBase = Sk2.col(1);
        auto PBaseCam1 = Sk1.col(1);

        // Y axis on cam2
        LineEnd = lineLength*YBase + PBase;
        PixelLineEnd = K2*(LineEnd/LineEnd(2));
        PixelLineStart = K2*(PBase/PBase(2));
        auto PixelLineStartCam1 = K1*(PBaseCam1/PBaseCam1(2));
        line(*f2, cv::Point (int(PixelLineStart[0]), int(PixelLineStart[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(100, 0, 0), 5, LINE_8);
        // Y axis on cam1
        LineEnd = (R12.transpose()*LineEnd) + T12;
        PixelLineEnd = K1*(LineEnd/LineEnd(2));
        line(*f1, cv::Point (int(PixelLineStartCam1[0]), int(PixelLineStartCam1[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(100, 0, 0), 5, LINE_8);
        // Z axis on cam2
        LineEnd = lineLength*ZBase + PBase;
        PixelLineEnd = K2*(LineEnd/LineEnd(2));
        line(*f2, cv::Point (int(PixelLineStart[0]), int(PixelLineStart[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(0, 100, 0), 5, LINE_8);
        // Z axis on cam1
        LineEnd = (R12.transpose()*LineEnd) + T12;
        PixelLineEnd = K1*(LineEnd/LineEnd(2));
        line(*f1, cv::Point (int(PixelLineStartCam1[0]), int(PixelLineStartCam1[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(0, 100, 0), 5, LINE_8);

        LineEnd = lineLength*XBase + Sk2.col(1);
        PixelLineEnd = K1*(LineEnd/LineEnd(2));
        line(*f2, cv::Point (int(PixelLineStart[0]), int(PixelLineStart[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(0, 0, 100), 5, LINE_8);
        LineEnd = (R12.transpose()*LineEnd) + T12;
        PixelLineEnd = K1*(LineEnd/LineEnd(2));
        line(*f1, cv::Point (int(PixelLineStartCam1[0]), int(PixelLineStartCam1[1])),
             cv::Point(int(PixelLineEnd[0]), int(PixelLineEnd[1])),
             Scalar(0, 0, 100), 5, LINE_8);

    }
}


inline tuple<Vector3d, Vector3d> SkeletonMerger::getEstimation() {
    return make_tuple(P1, P2);
}

inline tuple<Vector3d, Vector3d, Vector3d> SkeletonMerger::getThreeEstimation() {
    // P1 wrist, P2 elbow, P3 shoulder
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

    predictionTxt.open(fs::path(FLAGS_saveDir)/"prediction.txt");
    // Skeleton merger
    SkeletonMerger SM;
    TagDetector TD;
    // target: size, joints
//    int target[] = {3,7,6,5};
    int target[] = {3,4,3,2};
    double tagSize = 0.166; // 0.215
    double constrainLength1 = 0.24, constrainLength2 = 0.27, lambda1 = 0.0, lambda2 = 0.0, stepSize = 0.01;

//    SM.setTarget(target, constrainLength1, lambda1, stepSize, tagSize);
    SM.setThreeTargets(target, constrainLength1, constrainLength2, lambda1, lambda2, stepSize, tagSize);
    int index = 1;
    vector<double> length1, length2;
    Mat frame1, frame2, frameD1, frameD2, result;
    Matrix3d RCam2Human;
    Vector3d THuman2Cam;
    vector<double> angles_x, angles_y, angles_z;

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
            bool skDetected = false, isDisplay = true;
            SM.processing(&frame1, &frame2, &frameD1, &frameD2, &skDetected, &isDisplay);
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
        // real time camera
        context ctx;
        int WIDTH = 848, HEIGHT = 480;
        fs::path saveDir = FLAGS_saveDir;
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

            bool skDetected = false, isDisplay = false;
            SM.processing(&colorMat1, &colorMat2, &depthMat1, &depthMat2, &skDetected, &isDisplay);
//            if(!skDetected)
//                continue;
            Mat colorResult;
            MyTextOnImage(colorMat1,"Camera 1: red (approximation)");
            MyTextOnImage(colorMat2,"Camera 2");
            hconcat(colorMat1, colorMat2, colorResult);
            imshow("Two Color Cameras", colorResult);

            Vector3d P1, P2, P3, PWrist_base, PElbow_base, PShoulder_base;
//            tie(P1, P2) = SM.getEstimation();
            // P1 wrist, P2 elbow, P3 shoulder
            tie(P1, P2, P3) = SM.getThreeEstimation();
            bool isTransformed = SM.isTransformed();
            double l1 = (P1-P2).norm();
            double l2 = (P2-P3).norm();
//            cout<<"predicted lower arm length : "<<l1<<" upperer length: "<< l2 <<endl;
            length1.push_back(l1);

//            predictionTxt << P3(0) << " " << P3(1) << " " << P3(2)
//                          << " " << P2(0) << " " <<P2(1) << " " <<P2(2)
//                          << " " << P1(0) << " " <<P1(1) << " " <<P1(2)<<'\n';
            if(isTransformed){
                tie(RCam2Human, THuman2Cam) = SM.getHumanCamTF();
                // P1 wrist, P2 elbow, P3 shoulder
                PShoulder_base = RCam2Human*(P3 - P3);
                PElbow_base = RCam2Human*(P2 - P3);
                PWrist_base = RCam2Human*(P1 - P3);

//                cout << P3(0) << " " << P3(1) << " " << P3(2) << "\n"
//                     << P2(0) << " " <<P2(1) << " " <<P2(2) <<"\n"
//                     << P1(0) << " " <<P1(1) << " " <<P1(2)<<'\n';
//
//                cout << P3_base(0) << " " << P3_base(1) << " " << P3_base(2) << "\n"
//                << P2_base(0) << " " <<P2_base(1) << " " <<P2_base(2) <<"\n"
//                << P1_base(0) << " " <<P1_base(1) << " " <<P1_base(2)<<'\n';
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
            string colorSaveDir1 = saveDir / serials[0] / "color";
            string depthSaveDir1 = saveDir / serials[0] / "depth";
            string colorSaveDir2 = saveDir / serials[1] /"color2";
            string depthSaveDir2 = saveDir / serials[1] / "depth2";
            string colorOriginalDir1 = saveDir / serials[0] / "original";
            string colorOriginalDir2 = saveDir / serials[1] / "original";
            saveImages(colorSaveDir1, colorSaveName, colorSaveImages_ptr1);
            saveImages(depthSaveDir1, depthSaveName, depthSaveImages_ptr1);
            saveImages(colorSaveDir2, colorSaveName, colorSaveImages_ptr2);
            saveImages(depthSaveDir2, depthSaveName, depthSaveImages_ptr2);
            saveImages(colorOriginalDir1, colorSaveName, colorSaveOriginal_ptr1);
            saveImages(colorOriginalDir2, colorSaveName, colorSaveOriginal_ptr2);
        }
    }
    else{
        ifstream camParametersTxt(FLAGS_cameraParameters);
        vector<vector<string>> camsParametersVec;
        if (camParametersTxt.is_open())
        {
            string line;
            while ( getline (camParametersTxt, line) )
            {
                auto lineS = splitString(line, " ");
                camsParametersVec.push_back(lineS);
            }
            camParametersTxt.close();
        }

        string cam1Name = FLAGS_colorDir1;
        auto nameVec1 = splitString(cam1Name, "/");
        string cam2Name = FLAGS_colorDir2;
        auto nameVec2 = splitString(cam2Name, "/");
        if(nameVec1[nameVec1.size()-3]==camsParametersVec[0][0]){
            SM.setCamParameters(stod(camsParametersVec[0][1]), stod(camsParametersVec[0][2]),
                                stod(camsParametersVec[0][3]), stod(camsParametersVec[0][4]),
                                stod(camsParametersVec[1][1]), stod(camsParametersVec[1][2]),
                                stod(camsParametersVec[1][3]), stod(camsParametersVec[1][4]));
        }
        else if (nameVec2[nameVec2.size()-3]==camsParametersVec[0][0]){
            SM.setCamParameters(stod(camsParametersVec[1][1]), stod(camsParametersVec[1][2]),
                                stod(camsParametersVec[1][3]), stod(camsParametersVec[1][4]),
                                stod(camsParametersVec[0][1]), stod(camsParametersVec[0][2]),
                                stod(camsParametersVec[0][3]), stod(camsParametersVec[0][4]));
        }
        else
            throw std::invalid_argument( "the serial number of camera is not found on camParametersTxt file!" );

        std::vector<fs::path> filenames;
        for (const auto& entry : fs::directory_iterator{FLAGS_colorDir1}) {
            filenames.push_back(entry.path().filename());
            frameNumber ++;
        }
        cout<<"frameNumber: "<<frameNumber<<endl;
        predictionTxt<<"frameNumber: " << frameNumber<<'\n';
        /* sort filenames
        std::sort(filenames.begin(), filenames.end(), [](const auto& lhs, const auto& rhs)
        {
            string lhs_ = lhs.string();
            string rhs_ = rhs.string();
            string lhsn = lhs_.find("_")? lhs_.substr(lhs_.find("_")+1, lhs_.find(".")-lhs_.find("_")): lhs_;
            string rhsn = rhs_.find("_")? rhs_.substr(rhs_.find("_")+1, rhs_.find(".")-rhs_.find("_")): rhs_;
//            cout<< "filename: " << lhsn << " " << rhsn<<endl;
            return stoi(lhsn) < stoi(rhsn);
//            return lhs.string() < rhs.string();
        });
        */

        for (int frameIndex=0; frameIndex<frameNumber; frameIndex++) {
            string img = format("color_%d.png", frameIndex);
            string imgD = format("depth_%d.png", frameIndex);
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

            if (frame2.empty() || frame1.empty() || frameD2.empty() || frameD1.empty()) {
                cout<<"Couldn't read images"<<endl;
                break;
            }

            bool skDetected = false, isDisplay=true;
            SM.processing(&frame1, &frame2, &frameD1, &frameD2, &skDetected, &isDisplay);
            Mat colorResult;
            MyTextOnImage(frame1,"Camera 1: red (approximation)" );
            MyTextOnImage(frame2,"Camera 2 " + to_string(SM.angleArm));
            hconcat(frame1, frame2, colorResult);
            imshow("Two Color Cameras", colorResult);
            if(!skDetected)
                continue;
            Vector3d P1, P2, P3, PWrist_base, PElbow_base, PShoulder_base, PBase;
//            tie(P1, P2) = SM.getEstimation();

            tie(P1, P2, P3) = SM.getThreeEstimation();
            bool isTransformed = SM.isTransformed();
            double l1 = (P1-P2).norm();
            double l2 = (P2-P3).norm();
//            cout<<"predicted length1 : "<<l1<<" length2: "<< l2 <<endl;
            length1.push_back(l1);

            if(isTransformed){
                tie(RCam2Human, THuman2Cam) = SM.getHumanCamTF();
                // P1 wrist, P2 elbow, P3 shoulder
                if(index==1){
                    PBase = P3;
                }
                PShoulder_base = RCam2Human*(P3 - P3);
                PElbow_base = RCam2Human*(P2 - P3);
                PWrist_base = RCam2Human*(P1 - P3);

//                cout << P3(0) << " " << P3(1) << " " << P3(2) << "\n"
//                     << P2(0) << " " <<P2(1) << " " <<P2(2) <<"\n"
//                     << P1(0) << " " <<P1(1) << " " <<P1(2)<<'\n';
//
//                cout << PShoulder_base(0) << " " << PShoulder_base(1) << " " << PShoulder_base(2) << "\n"
//                     << PElbow_base(0) << " " <<PElbow_base(1) << " " <<PElbow_base(2) <<"\n"
//                     << PWrist_base(0) << " " <<PWrist_base(1) << " " <<PWrist_base(2)<<'\n';
            }

            angles_x.push_back(SM.angles_ShoulderBase[0]);
            angles_y.push_back(SM.angles_ShoulderBase[1]);
            angles_z.push_back(SM.angles_ShoulderBase[2]);
            if(index>=5) {
                predictionTxt << "frame: " << index << '\n';
                predictionTxt << "Joints (Shoulder-Elbow-Wrist in rowwise):\n"
                              << PShoulder_base(0) << " " << PShoulder_base(1) << " " << PShoulder_base(2) << '\n'
                              << PElbow_base(0) << " " << PElbow_base(1) << " " << PElbow_base(2) << '\n'
                              << PWrist_base(0) << " " << PWrist_base(1) << " " << PWrist_base(2) << '\n';
                //            predictionTxt <<"Angles predicted by SVD rotation matrix (Rz*Ry*Rx):\n" << SM.angleZ<<'\n';
                //            predictionTxt <<"Rotation matrix (Shoulder = R*Base):\n"
                //                          << SM.RotationBaseShoulder<<'\n';
                //            predictionTxt <<"Rotate angles XYZ (Shoulder = R*Base):\n"
                //                          << SM.angles_ShoulderBase.transpose()<<'\n';
//                predictionTxt << "Rotation matrix (Shoulder = R*Shoulder0):\n"
//                              << SM.RShoulderF0F1(0, 0) << " " << SM.RShoulderF0F1(0, 1) << " "
//                              << SM.RShoulderF0F1(0, 2) << '\n'
//                              << SM.RShoulderF0F1(1, 0) << " " << SM.RShoulderF0F1(1, 1) << " "
//                              << SM.RShoulderF0F1(1, 2) << '\n'
//                              << SM.RShoulderF0F1(2, 0) << " " << SM.RShoulderF0F1(2, 1) << " "
//                              << SM.RShoulderF0F1(2, 2) << '\n';
//                predictionTxt << "Rotate angles XYZ (Shoulder = R*Shoulder0):\n"
//                              << SM.angles_f0f1.transpose() << '\n';
                predictionTxt << "Rotation matrix (Shoulder = R*Shoulder0):\n"
                              << SM.RotationBaseShoulder(0, 0) << " " << SM.RotationBaseShoulder(0, 1) << " "
                              << SM.RotationBaseShoulder(0, 2) << '\n'
                              << SM.RotationBaseShoulder(1, 0) << " " << SM.RotationBaseShoulder(1, 1) << " "
                              << SM.RotationBaseShoulder(1, 2) << '\n'
                              << SM.RotationBaseShoulder(2, 0) << " " << SM.RotationBaseShoulder(2, 1) << " "
                              << SM.RotationBaseShoulder(2, 2) << '\n';
                predictionTxt << "Rotate angles XYZ (Shoulder = R*Shoulder0):\n"
                              << SM.angles_ShoulderBase.transpose() << '\n';


                predictionTxt << "Flexion angle: " << SM.angleArm << '\n' << '\n';
            }
            string saveDir = fs::path(FLAGS_saveDir) / "output/" ;
            if(!fs::exists(saveDir)){
                fs::create_directories(saveDir);
            }
            cout<<"savedir: "<<saveDir<<endl;
            imwrite(fs::path(saveDir) / img, colorResult);
            index ++;
            int key = waitKey(1); // key is an integer here
            if (key == 27)
                break;
        }
    }
    predictionTxt.close();

    // plot limb length over time
//    vector<double> constrain1 (frameNumber, constrainLength1);
//    plt::named_hist("Constrain length L_ew", constrain1);
//    plt::show();
//    plt::close();
    // plt::plot(angles_x);
    // plt::plot(angles_y);
    // plt::plot(angles_z);
    // plt::show();
//    if(!FLAGS_demo){
//        vector<double> constrain1 (frameNumber, constrainLength1);
//        plt::named_hist("Constrain length L_ew", constrain1);
//        plt::named_hist("Estimated length", length1);
//        plt::legend();
//        string imgDir = format("/media/dataset/translation/skeleton_fusion/results/cost/Length_over_time.png");
//        plt::save(imgDir);
//        plt::show();
//        plt::close();
//    }
    return 0;
}