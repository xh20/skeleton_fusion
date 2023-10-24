//
// Created by hao on 09.08.21.
//

#ifndef SKELETON_FUSION_SKELETON_FUSION_HPP
#define SKELETON_FUSION_SKELETON_FUSION_HPP

#endif //SKELETON_FUSION_SKELETON_FUSION_HPP

#include <iostream>
#include <eigen3/Eigen/Dense>

#include <openpose/headers.hpp>
#include <openpose/flags.hpp>
#include <opencv2/opencv.hpp>

using namespace op;
using namespace std;
using namespace cv;
using namespace Eigen;

class OpWrapper{
private:
    Mat input_img;
    Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
    shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
    void configureWrapper();
    bool display();
    void processKeypoints(const Mat& frameD, double fx, double fy, double cx, double cy);
    MatrixXd skeleton;
    VectorXd confidence;
public:
    OpWrapper();
    virtual ~OpWrapper();
    void run(Mat* frame, const Mat* frameD, const double* fx, const double* fy,
             const double* cx, const double* cy, const bool* noDisplay, MatrixXd*, VectorXd*);
    tuple<MatrixXd, VectorXd> getSkeleton();
};

//inline OpWrapper::OpWrapper()= default;

inline OpWrapper::OpWrapper(){
    opLog("Starting OpenPose demo...", op::Priority::High);
    const auto opTimer = op::getTimerInit();

    // Configuring OpenPose
    opLog("Configuring OpenPose...", op::Priority::High);
    configureWrapper();

    // Starting OpenPose
    op::opLog("Starting thread(s)...", op::Priority::High);
    opWrapper.start();
}

inline OpWrapper::~OpWrapper()= default;

inline void OpWrapper::configureWrapper()
{
    try
    {
//        FLAGS_hand = true;
//        FLAGS_hand_detector = 3;
        // Configuring OpenPose
        // logging_level
        op::checkBool(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog("Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                    " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;
        const string modelFolder = "/home/geriatronics/openpose/models";
        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
                poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
                FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
                op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
                (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(modelFolder),
                heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
                FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
                op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging};
//        const op::WrapperStructPose wrapperStructPose{
//                poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
//                FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
//                poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
//                FLAGS_part_to_show, op::String(FLAGS_model_folder), heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
//                (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
//                op::String(FLAGS_prototxt_path), op::String(FLAGS_caffemodel_path),
//                (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
                FLAGS_face, faceDetector, faceNetInputSize,
                op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
                (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
                FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
                op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
                (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
                FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
                FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
                op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
                FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
                op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
                op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
                op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
                op::String(FLAGS_udp_port)};
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

inline void OpWrapper::run(Mat* frame, const Mat* frameD, const double* fx, const double* fy,
        const double* cx, const double* cy, const bool* noDisplay, MatrixXd *sk, VectorXd * acc) {
    const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(*frame); //OP_CV2OPCONSTMAT
    datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
    if (datumProcessed != nullptr)
    {
//        frame = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData);
        processKeypoints(*frameD, *fx, *fy, *cx, *cy);
        *sk = skeleton;
        *acc = confidence;
        if (!noDisplay)
//        if (true)
        {
            const auto userWantsToExit = display();
            if (userWantsToExit)
            {
                op::opLog("User pressed Esc to exit demo.", op::Priority::High);
                exit(1);
            }
        }
    }
    else{
        op::opLog("Video could not be processed.", op::Priority::High);
    }
}

inline void OpWrapper::processKeypoints(const Mat& frameD, double fx, double fy, double cx, double cy)
{
    try
    {
        if (datumProcessed != nullptr && !datumProcessed->empty())
        {
            if(datumProcessed->at(0)->poseKeypoints.getCvMat().dims()!=0)
            {
                skeleton = MatrixXd::Zero(3,25);
                confidence = VectorXd::Zero(25);
                op::Array<float> keypoints = datumProcessed->at(0)->poseKeypoints;
                vector<int> dim = keypoints.getSize(); // dim: 1x25x3

                for(int i=0; i<dim[1]; i++){
                    auto x = keypoints.at({0,i,0});
                    auto y = keypoints.at({0,i,1});
                    auto a = keypoints.at({0,i,2});
//                    ushort depth = frameD.at<ushort>(cvRound(y),cvRound(x));
                    double depth = (frameD.at<ushort>(cvRound(y),cvRound(x))/1000.0);
                    if(x > 0.0 && y>0.0 && depth<=0.0){
                        depth = 4.000;
                        a = 0.0;
                    }
                    double X = ((x - cx)/fx)*depth;
                    double Y = ((y - cy)/fy)*depth;
                    skeleton.col(i) << X, Y, depth;
                    confidence(i) = a;
                }
//                cout<<"skeleton pixel: "<<keypoints.at({0,0,0})<<" "<<keypoints.at({0,1,0})<<endl;
            }
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

inline tuple<MatrixXd, VectorXd> OpWrapper::getSkeleton() {
    return make_tuple(skeleton, confidence);
}

inline bool OpWrapper::display()
{
    try
    {
        // User's displaying/saving/other processing here
        // datum.cvOutputData: rendered frame with pose or heatmaps
        // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumProcessed != nullptr && !datumProcessed->empty())
        {
            // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData); //OP_OP2CVCONSTMAT
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
        const auto key = (char)cv::waitKey(1);
        return (key == 27);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return true;
    }
}
