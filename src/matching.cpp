#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath> 
using namespace std;
using namespace cv;

struct MotionVector {
    Point2i blockPosition;
    Point2i motion;      
};

Mat custom_absdiff(const Mat& img1, const Mat& img2) {

    Mat abs_diff = Mat::zeros(img1.size(), img1.type());

    int channels = img1.channels();
    int rows = img1.rows;
    int cols = img1.cols * channels;

    for(int y = 0; y < rows; y++) {
        const uchar* ptr1 = img1.ptr<unsigned char>(y);
        const uchar* ptr2 = img2.ptr<unsigned char>(y);
        unsigned char* ptr_diff = abs_diff.ptr<unsigned char>(y);

        for(int x = 0; x < cols; x++) {
            ptr_diff[x] = static_cast<unsigned char>(abs(ptr1[x] - ptr2[x]));
        }
    }

    return abs_diff;
}

int main() {
    string frameAPath = "/home/jejung/PycharmProjects/MOT15/MOT15/test/KITTI-16/img1/000001.jpg";
    string frameBPath = "/home/jejung/PycharmProjects/MOT15/MOT15/test/KITTI-16/img1/000002.jpg";

    Mat frameA = imread(frameAPath, IMREAD_GRAYSCALE);
    Mat frameB = imread(frameBPath, IMREAD_GRAYSCALE);

    if (frameA.empty() || frameB.empty()) {
        cerr << "이미지를 로드할 수 없습니다." << endl;
        return -1;
    }

    const int blockSize = 16;    
    const int searchRange = 8;  

    int rows = frameA.rows;
    int cols = frameA.cols;

    vector<MotionVector> motionVectors;

    Mat result;
    cvtColor(frameA, result, COLOR_GRAY2BGR);

    for (int y = 0; y <= rows - blockSize; y += blockSize) {
        for (int x = 0; x <= cols - blockSize; x += blockSize) {
            Rect currentBlockRect(x, y, blockSize, blockSize);
            Mat currentBlock = frameA(currentBlockRect);

            int minY = max(y - searchRange, 0);
            int maxY = min(y + searchRange, rows - blockSize);
            int minX = max(x - searchRange, 0);
            int maxX = min(x + searchRange, cols - blockSize);

            double minSSD = numeric_limits<double>::max();
            Point2i bestMatch(0, 0);

            for (int j = minY; j <= maxY; ++j) {
                for (int i = minX; i <= maxX; ++i) {
                    Rect candidateRect(i, j, blockSize, blockSize);
                    Mat candidateBlock = frameB(candidateRect);

                    Mat diff = custom_absdiff(currentBlock, candidateBlock);

                    Mat diff32F;
                    diff.convertTo(diff32F, CV_32F);
                    Mat squaredDiff = diff32F.mul(diff32F);
                    double ssd = sum(squaredDiff)[0];

                    if (ssd < minSSD) {
                        minSSD = ssd;
                        bestMatch = Point2i(i - x, j - y);
                    }
                }
            }

            MotionVector mv;
            mv.blockPosition = Point2i(x, y);
            mv.motion = bestMatch;
            motionVectors.push_back(mv);

            Rect movedBlockRect(x + mv.motion.x, y + mv.motion.y, blockSize, blockSize);
            rectangle(result, movedBlockRect, Scalar(0, 0, 255), 1);
        }
    }

    imshow("Frame A", frameA);
    imshow("Frame B", frameB);
    imshow("Motion Vectors", result);
    waitKey(0);
    return 0;
}
