//
// Created by hao on 19.08.21.
//

#ifndef SKELETON_FUSION_GRADUATENONCONVEXITY_HPP
#define SKELETON_FUSION_GRADUATENONCONVEXITY_HPP

#endif //SKELETON_FUSION_GRADUATENONCONVEXITY_HPP
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <matplotlibcpp.h>

using namespace Eigen;
namespace plt = matplotlibcpp;


class GraduateNonConvexity{
private:
    // Parameters
    // Weights
    VectorXd W, A, costM, W2;
    // Constrain
    double L, L1, L2;
    // Cost
    double cost{}, c2, mu, lambda, lambda1, lambda2, delta, costOriginal{};
    // Estimate joint, measured joints and gradients
    Vector3d P, P11, P12, P13, P21, P22, P23, grad, P1, P2, P3, grad1, grad2, grad3;
//    VectorXd P1, P2;
    // is system ready to run
    bool isSetup, isTwoPoints, isThreePoints;

    // Functions
    // Update Weights
    void updateWeights();
    void updateTwoPointsWeights();
    void updateThreePointsWeights();
    // Update Grad
    void updateGrad();
    void updateTwoPointsGrad();
    void updateThreePointsGrad();
    // Update cost
    void updateCost();
    void updateTwoPointsCost();
    void updateThreePointsCost();
    // Update estimation
    void updateEstimation();
    void updateTwoPointsEstimation();
    void updateThreePointsEstimation();
    // Initialize Cost
    void initialCost();
    void initialTwoPointsCost();
    void initialThreePointsCost();

public:
    GraduateNonConvexity();
    ~GraduateNonConvexity();
    [[nodiscard]] Vector3d getEstimation() const;
    [[nodiscard]] tuple<Vector3d, Vector3d> getTwoPointsEstimation() const;
    [[nodiscard]] tuple<Vector3d, Vector3d, Vector3d> getThreePointsEstimation() const;
    [[nodiscard]] double getCost() const;
    void setupGNC(const MatrixXd &PTarget, const VectorXd& Acc, const double& length,
            const double& Lambda, const double& stepSize, const int& pointsNumber);
    void run(const bool& isVisualized, const int& frameID);
};

inline GraduateNonConvexity::GraduateNonConvexity():c2(1.0), mu(1000.0), L(0.22), lambda(0.5), delta(1.0),
                                                    L1(0.23), lambda1(0.5), L2(0.24), lambda2(0.5),
                                                    isSetup(false), isTwoPoints(false), isThreePoints(false)
{
    P.setZero();
    grad.setZero();
    P1.setZero();
    P2.setZero();
    P3.setZero();
    grad1.setZero();
    grad2.setZero();
    grad3.setZero();
}

inline GraduateNonConvexity::~GraduateNonConvexity()= default;

// P target: col 0-1: first joint in cam 1 and 2; col 2-3: second joint in cam 1 and 2
// Pij is j-th joint in i-th cam coordinate
// Acc: accuracy 0: first joint at cam1; 1: first joint at cam2; 2: second joint at cam1; 3: second joint at cam2;
inline void GraduateNonConvexity::setupGNC(const MatrixXd &PTarget, const VectorXd& Acc,
                                           const double& length, const double& Lambda, const double& stepSize, const int& pointsNumber) {
    // P11 21:
    // P12 22:
    // P13 23:
    P11 = PTarget.col(0);
    P21 = PTarget.col(1);
    P12 = PTarget.col(2);
    P22 = PTarget.col(3);
    P13 = PTarget.col(4);
    P23 = PTarget.col(5);
    L = length;
    lambda = Lambda;
    delta = stepSize;

    if(pointsNumber == 2){
        isTwoPoints = true;
        A = VectorXd::Ones(PTarget.cols()+1);
        A << Acc, 1;
//        A << Acc(0)/(Acc(0)+Acc(1)), Acc(1)/(Acc(0)+Acc(1)),
//                Acc(2)/(Acc(2)+Acc(3)), Acc(3)/(Acc(2)+Acc(3)), 1;
        W = VectorXd::Ones(A.size());
        costM = VectorXd::Zero(A.size());
    }
    else if(pointsNumber == 3)
    {
        isThreePoints = true;
        A = VectorXd::Ones(PTarget.cols()+2);
        A << Acc, 1, 1;
        W = VectorXd::Ones(A.size());
        costM = VectorXd::Zero(A.size());
    }
    else{
        A = VectorXd::Zero(Acc.size());
        A << Acc(0)/(Acc(0)+Acc(1)), Acc(1)/(Acc(0)+Acc(1)),
                Acc(2)/(Acc(2)+Acc(3)), Acc(3)/(Acc(2)+Acc(3));
        costM = VectorXd::Zero(A.size());
        W = VectorXd::Ones(A.size());
    }

    isSetup = true;
}

inline Vector3d GraduateNonConvexity::getEstimation() const{
    return P;
}

inline double GraduateNonConvexity::getCost() const {
    return cost;
}

inline void GraduateNonConvexity::updateGrad() {
    W2 = W.array().pow(2);
    grad = 2*W2(0)*(P-P11) + 2*W2(1)*(P-P21)
           + lambda*(2*W2(2)*(1-L/(P-P12).norm())*(P-P12) + 2*W2(3)*(1-L/(P-P22).norm())*(P-P22));
}

inline void GraduateNonConvexity::updateWeights() {
    W = (mu*c2*A).array()/(costM.array()+mu*c2).array();
}

inline void GraduateNonConvexity::updateCost() {
    // cost = W1^2*(P-P11)^2 + W2^2*(P-P12)^2 + W3^2*(|P-P12|-L) + W4^2*(|P-P22|-L)
    costM << (P-P11).squaredNorm(), (P-P21).squaredNorm(), ((P-P12).norm()-L), ((P-P22).norm()-L);
    cost = W2(0)*costM[0] + W2(1)*costM[1] + lambda*(W2(2)*costM[2] + W2(3)*costM[3]);
    costOriginal = cost;
    cost += mu*c2*(W - VectorXd::Ones(W.size())).squaredNorm();
}

inline void GraduateNonConvexity::initialCost() {
//    P =  A(0)*P11+A(1)*P21;
    P = P21;
    cost = (P-P11).squaredNorm() + (P-P21).squaredNorm()
            + lambda*(pow(((P-P12).norm() - L),2) + pow(((P-P22).norm() - L),2));
    costOriginal = cost;
    mu = 3000.0;
}

inline void GraduateNonConvexity::updateEstimation() {
    P = P + grad*delta;
}

// ************************************ Two Points ******************************************** //

inline void GraduateNonConvexity::initialTwoPointsCost() {
    P1 =  A(0)/(A(0)+A(1))*P11+A(1)/(A(0)+A(1))*P21;
    P2 =  A(2)/(A(2)+A(3))*P12+A(3)/(A(2)+A(3))*P22;
//    P1 = 0.5*P21 + 0.5*P11;
//    P2 = 0.5*P22 + 0.5*P12;
//    P1 = P11;
//    P2 = P12;
    costM << (P1-P11).squaredNorm(),(P1-P21).squaredNorm(),(P2-P12).squaredNorm(),(P2-P22).squaredNorm(),pow(((P1-P2).norm()-L),2);
    cost = (P1-P11).squaredNorm() + (P1-P21).squaredNorm() + (P2-P12).squaredNorm() + (P2-P22).squaredNorm()
           + lambda*(pow(((P1-P2).norm() - L),2));
    costOriginal = cost;
    mu = 3000.0;
}

inline void GraduateNonConvexity::updateTwoPointsGrad() {
    W2 = W.array().pow(2);
    grad1 = 2*W2(0)*(P1-P11) + 2*W2(1)*(P1-P21) + lambda*(2*W2(4)*(1-L/(P1-P2).norm())*(P2-P1));
    grad2 = 2*W2(2)*(P2-P12) + 2*W2(3)*(P2-P22) + lambda*(2*W2(4)*(1-L/(P1-P2).norm())*(P1-P2));
}

inline void GraduateNonConvexity::updateTwoPointsEstimation() {
    P1 = P1 + grad1*delta;
    P2 = P2 + grad2*delta;
}

inline void GraduateNonConvexity::updateTwoPointsCost() {
    // cost = W1^2*(P1-P11)^2 + W2^2*(P1-P21)^2 + W3^2*(P2-P12)^2 + W4^2*(P2-P22)^2 + W5^2*(|P1-P2|-L)
    costM << (P1-P11).squaredNorm(), (P1-P21).squaredNorm(), (P2-P12).squaredNorm(), (P2-P22).squaredNorm(), pow(((P1-P2).norm()-L),2);
    cost = W2(0)*costM(0) + W2(1)*costM(1) + W2(2)*costM(2) +
            W2(3)*costM(3) + lambda*(W2(4)*costM(4));
    costOriginal = cost;
//    cost += mu*c2*(W - VectorXd::Ones(W.size())).squaredNorm();
}

inline void GraduateNonConvexity::updateTwoPointsWeights() {
    W = (mu*c2*A).array()/(costM.array()+mu*c2).array();
    W(4)=1;
}

inline tuple<Vector3d, Vector3d> GraduateNonConvexity::getTwoPointsEstimation() const{
    return make_tuple(P1, P2);
}

// *********************************** Three Points **************************** //
inline void GraduateNonConvexity::initialThreePointsCost() {
    // Accuracy: 0 - P11, 1 - P21, 2 - P12, 3 - P22, 4 - P13, 5 - P23, 6 - constant,
    P1 = A(0)/(A(0)+A(1))*P11+A(1)/(A(0)+A(1))*P21;
    P2 = A(2)/(A(2)+A(3))*P12+A(3)/(A(2)+A(3))*P22;
    P3 = A(4)/(A(4)+A(5))*P13+A(5)/(A(4)+A(5))*P23;
//    P1 = 0.5*P21 + 0.5*P11;
//    P2 = 0.5*P22 + 0.5*P12;
//    P3 = 0.5*P23 + 0.5*P13;
//    P1 = P21;
//    P2 = P22;
//    P3 = P23;
    costM << (P1-P11).squaredNorm(), (P1-P21).squaredNorm(), (P2-P12).squaredNorm(), (P2-P22).squaredNorm(),
            (P3-P13).squaredNorm(), (P3-P23).squaredNorm(),
            pow(((P1-P2).norm()-L1),2), pow(((P2-P3).norm()-L2),2);
    cost = A(0)*costM(0) + A(1)*costM(1) + A(2)*costM(2) +A(3)*costM(3)
           + A(4)*costM(4) + A(5)*costM(5)
           + lambda1*(1*costM(6)) + lambda2*(1*costM(7));
    costOriginal = cost;
    mu = 3000.0;
}

inline void GraduateNonConvexity::updateThreePointsGrad() {
    W2 = W.array().pow(2);
    grad1 = 2*W2(0)*(P1-P11) + 2*W2(1)*(P1-P21) + lambda1*(2*W2(6)*(1-L1/(P1-P2).norm())*(P2-P1));
    grad2 = 2*W2(2)*(P2-P12) + 2*W2(3)*(P2-P22) + lambda1*(2*W2(6)*(1-L1/(P1-P2).norm())*(P1-P2)) +
            lambda2*(2*W2(7)*(1-L2/(P2-P3).norm())*(P3-P2));
    grad3 = 2*W2(4)*(P3-P13) + 2*W2(5)*(P3-P23) + lambda2*(2*W2(7)*(1-L2/(P2-P3).norm())*(P2-P3));
}

inline void GraduateNonConvexity::updateThreePointsEstimation() {
    P1 = P1 + grad1*delta;
    P2 = P2 + grad2*delta;
    P3 = P3 + grad3*delta;
}

inline void GraduateNonConvexity::updateThreePointsCost() {
    // cost = W1^2*(P1-P11)^2 + W2^2*(P1-P21)^2 + W3^2*(P2-P12)^2 + W4^2*(P2-P22)^2 + W5^2*(|P1-P2|-L)
    costM << (P1-P11).squaredNorm(), (P1-P21).squaredNorm(), (P2-P12).squaredNorm(), (P2-P22).squaredNorm(),
             (P3-P13).squaredNorm(), (P3-P23).squaredNorm(),
             pow(((P1-P2).norm()-L1),2), pow(((P2-P3).norm()-L2),2);
    cost = W2(0)*costM(0) + W2(1)*costM(1) + W2(2)*costM(2) + W2(3)*costM(3)
            + W2(4)*costM(4) + W2(5)*costM(5)
            + lambda1*(W2(6)*costM(6)) + lambda2*(W2(7)*costM(7));
    costOriginal = cost;
//    cost += mu*c2*pow(((W - VectorXd::Ones(W.size())).transpose()*A),2);
}

inline void GraduateNonConvexity::updateThreePointsWeights() {
    W = (mu*c2*A).array()/(costM.array()+mu*c2).array();
    W(6)=1;
    W(7)=1;
}

inline tuple<Vector3d, Vector3d, Vector3d> GraduateNonConvexity::getThreePointsEstimation() const{
    return make_tuple(P1, P2, P3);
}


// ******************************** RUN ***********************************//

inline void GraduateNonConvexity::run(const bool& isVisualized, const int& frameID) {
    if (!isSetup){
        printf("The system is not setup, please run setupGNC before run!");
        exit(-1);
    }
    if(isTwoPoints){
        vector<double> cost1 = {};
        vector<double> cost2 = {};
        vector<double> cost3 = {};
        initialTwoPointsCost();
        cost1.push_back(costM(0));  //costOriginal
        cost2.push_back(costM(1));  //costOriginal
        cost3.push_back(costM(4));  //costOriginal
        int index = 0;

        while (mu>1.0){
//            printf("Estimating single point at iteration: %d with mu %f\n", index, mu);
            updateTwoPointsGrad();
            updateTwoPointsEstimation();
            updateTwoPointsCost();
            updateTwoPointsWeights();
            cost1.push_back(costM(0));  //costOriginal
            cost2.push_back(costM(1));  //costOriginal
            cost3.push_back(costM(4));  //costOriginal
//        cout<<"updated weights: \n"<<W<<endl;
            mu = mu/1.2;
            index++;
            if(index%10==0)
                delta /=10;
//            cout<<"Estimated point: \n"<<P<<endl;
        }
    }
    else if(isThreePoints){
        vector<double> cost1 = {};
        vector<double> cost2 = {};
        vector<double> cost3 = {};
        initialThreePointsCost();

        cost1.push_back(costM(0));  //costOriginal
        cost2.push_back(costM(1));  //costOriginal
        cost3.push_back(costM(4));  //costOriginal

        int index = 0;

        while (mu>1.0) {
//            printf("Estimating single point at iteration: %d with mu %f\n", index, mu);
            updateThreePointsGrad();
            updateThreePointsEstimation();
            updateThreePointsCost();
            updateThreePointsWeights();
            cost1.push_back(costM(0));  //costOriginal
            cost2.push_back(costM(1));  //costOriginal
            cost3.push_back(costM(4));  //costOriginal
//        cout<<"updated weights: \n"<<W<<endl;
            mu = mu / 1.2;
            index++;
            if (index % 10 == 0)
                delta /= 10;
//            cout<<"Estimated point: \n"<<P<<endl;
        }
    }
    else{
        initialCost();
        int index = 0;
        vector<double> costV = {};
        while (mu>1.0){
//            printf("Estimating single point at iteration: %d with mu %f\n", index, mu);
            updateGrad();
            updateEstimation();
            updateCost();
            updateWeights();
            costV.push_back(cost);  //costOriginal
//        cout<<"updated weights: \n"<<W<<endl;
            mu = mu/1.2;
            index++;
            if(index%10==0)
                delta /=10;
//            cout<<"Estimated point: \n"<<P<<endl;
        }

//        plt::named_plot("distance to left camera measurement of P1 (upper)",cost1);
//        plt::named_plot("distance to right camera measurement of P1 (upper)", cost2);
//        plt::named_plot("cost of |P1-P2|-L", cost3);
//        plt::legend();
//        string imgDir = format("/media/dataset/translation/skeleton_fusion/results/cost/Cost%d.png", frameID);
//        plt::save(imgDir);
//        plt::show();
//        plt::close();
    }


}