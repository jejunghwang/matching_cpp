#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <sstream>
#include <unistd.h>

using namespace std;
using namespace cv;

Mat cur_frame;   // 현재 프레임
Mat prev_frame;  // 이전 프레임
map<int, Rect> object_rects; 

void drawRectangle(Mat& frame) {
    for (const auto& obj : object_rects) {
        rectangle(frame, obj.second, Scalar(0, 0, 255), 2);
    }
}


void difference(const Mat& img1, const Mat& img2, Mat& diff) {
    diff.create(img1.size(), img1.type());

    int channels = img1.channels();
    int rows = img1.rows;
    int cols = img1.cols * channels;

    for (int y = 0; y < rows; y++) {
        const uchar* ptr1 = img1.ptr<uchar>(y);
        const uchar* ptr2 = img2.ptr<uchar>(y);
        uchar* ptr_diff = diff.ptr<uchar>(y);

        for (int x = 0; x < cols; x++) {
            ptr_diff[x] = static_cast<uchar>(abs(ptr1[x] - ptr2[x]));
        }
    }
}


Point blockMatching(const Mat& prev_frame, const Mat& cur_frame, const Rect& search_window) {
    int box_width = prev_frame.cols;
    int box_height = prev_frame.rows;

    double minSSD = 999999999;
    Point bestMatchLoc = search_window.tl();

    int startY = search_window.y;
    int endY = search_window.y + search_window.height - box_height;
    int startX = search_window.x;
    int endX = search_window.x + search_window.width - box_width;

    endY = min(endY, cur_frame.rows - box_height);
    endX = min(endX, cur_frame.cols - box_width);

    for (int y = startY; y <= endY; y++) {
        for (int x = startX; x <= endX; x++) {
            Rect candidateRect(x, y, box_width, box_height);
            Mat candidateRegion = cur_frame(candidateRect);
            Mat diff;
            difference(prev_frame, candidateRegion, diff);
            diff.convertTo(diff, CV_32F);
            Mat squaredDiff = diff.mul(diff);
            double ssd = sum(squaredDiff)[0];

            if (ssd < minSSD) {
                minSSD = ssd;
                bestMatchLoc = Point(x, y);
            }
        }
    }

    return bestMatchLoc;
}

void keyInput(VideoCapture& src, int& cur_frame_num) {
    Mat temp;
    while (true) {
        int temp_key = waitKey(0);
        if (temp_key == ' ') break;  // 스페이스바 입력 시 종료
        else if (temp_key == 'p' || temp_key == 'P') { // 다음 프레임으로 이동
            src >> temp;
            if (temp.empty()) {   // 프레임이 없으면 현재 프레임 번호 유지
                src.set(CAP_PROP_POS_FRAMES, src.get(CAP_PROP_POS_FRAMES) - 1);
                continue;
            }
            cur_frame = temp;
            cur_frame_num = max(0, (int)src.get(CAP_PROP_POS_FRAMES) - 1);

            Mat display_frame = cur_frame.clone();
            drawRectangle(display_frame);

            imshow("Video Player", display_frame);
        }
        else if (temp_key == 'n' || temp_key == 'N') { // 이전 프레임으로 이동
            int new_frame_num = max(0, (int)src.get(CAP_PROP_POS_FRAMES) - 2);
            src.set(CAP_PROP_POS_FRAMES, new_frame_num);
            src >> temp;
            if (temp.empty()) { // 프레임이 없으면 다음 프레임으로 이동
                src.set(CAP_PROP_POS_FRAMES, new_frame_num + 1);
                continue;
            }
            cur_frame = temp;
            cur_frame_num = new_frame_num;

            Mat display_frame = cur_frame.clone();
            drawRectangle(display_frame);

            imshow("Video Player", display_frame);
        }
        else if (temp_key == 27) {  // ESC 키 입력 시 프로그램 종료
            cerr << "Program Terminated!\n";
            exit(0);
        }
    }
}

int main() {
    VideoCapture src("tracking_video.avi");  // 비디오 파일 열기
    if (!src.isOpened()) {
        cerr << "Video file cannot be opened\n";
        return 0;
    }

    namedWindow("Video Player");

    //////////////////////////////////////////////////////////////////////////

    FILE* fp = fopen("gt.txt", "r");
    if (fp == NULL) {
        cerr << "Failed to open ground truth file\n";
        return 1;
    }

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        int frame_n, id, conf, conf1;
        float box_left, box_top, width, height;

        sscanf(line, "%d,%d,%f,%f,%f,%f,%d,%d,%d,%d", &frame_n, &id, &box_left, &box_top, &width, &height, &conf, &conf1, &conf1, &conf1);
        if (frame_n == 1) {
            Rect rect(box_left, box_top, width, height);
            if (conf) {
                object_rects[id] = rect; // 객체 ID를 키로 저장
            }
        }
    }

    fclose(fp);

    //////////////////////////////////////////////////////////////////////////

    int cur_frame_num = 0;   // 현재 프레임 번호
    const int target_fps = 30;   // FPS 설정
    const double time_per_frame = 1000.0 / target_fps;  // 프레임당 시간 (ms)

    src.set(CAP_PROP_POS_FRAMES, 0);  // 첫 프레임으로 이동
    src >> cur_frame;

    if (cur_frame.empty()) {
        cerr << "Failed to read the first frame.\n";
        return -1;
    }

    prev_frame = cur_frame.clone();

    Mat display_frame = cur_frame.clone();
    drawRectangle(display_frame);
    imshow("Video Player", display_frame);

    // 초기 키 입력 처리
    keyInput(src, cur_frame_num);

    // 메인 루프
    while (true) {
        int64 start_time = getTickCount();

        src >> cur_frame; 
        if (cur_frame.empty()) break;

        cur_frame_num = max(0, (int)src.get(CAP_PROP_POS_FRAMES) - 1); 

        Mat display_frame = cur_frame.clone();

        map<int, Rect> new_object_rects;

        Mat gray_prev_frame, gray_cur_frame;
        cvtColor(prev_frame, gray_prev_frame, COLOR_BGR2GRAY);
        cvtColor(cur_frame, gray_cur_frame, COLOR_BGR2GRAY);

        for (auto& obj : object_rects) {
            int obj_id = obj.first;
            Rect prev_rect = obj.second;

            Rect valid_prev_rect = prev_rect & Rect(0, 0, prev_frame.cols, prev_frame.rows);
            if (valid_prev_rect.width <= 0 || valid_prev_rect.height <= 0)
                continue;

            Mat prev_roi = gray_prev_frame(valid_prev_rect);

            int search_area = 20; 
            Rect search_window(
                max(prev_rect.x - search_area, 0),
                max(prev_rect.y - search_area, 0),
                prev_rect.width + 2 * search_area,
                prev_rect.height + 2 * search_area
            );

            Rect valid_search_window = search_window & Rect(0, 0, cur_frame.cols, cur_frame.rows);
            if (valid_search_window.width <= 0 || valid_search_window.height <= 0)
                continue;

            Point matchLoc = blockMatching(prev_roi, gray_cur_frame, valid_search_window);
            Rect new_rect(matchLoc.x, matchLoc.y, prev_rect.width, prev_rect.height);

            new_object_rects[obj_id] = new_rect;
            rectangle(display_frame, new_rect, Scalar(0, 0, 255), 2);
        }

        object_rects = new_object_rects;

        imshow("Video Player", display_frame);
        int key = waitKey(32);
        if (key == ' ') {  // 스페이스바 입력 시 일시정지
            keyInput(src, cur_frame_num);
        }
        else if (key == 'p' || key == 'P') { // 다음 프레임으로 이동
            src >> cur_frame;
            if (cur_frame.empty()) { // 프레임이 없으면 종료
                break;
            }
            cur_frame_num = max(0, (int)src.get(CAP_PROP_POS_FRAMES) - 1);

            Mat display_frame = cur_frame.clone();
            drawRectangle(display_frame);

            imshow("Video Player", display_frame);
        }
        else if (key == 'n' || key == 'N') { // 이전 프레임으로 이동
            int new_frame_num = max(0, (int)src.get(CAP_PROP_POS_FRAMES) - 2);
            src.set(CAP_PROP_POS_FRAMES, new_frame_num);
            src >> cur_frame;
            if (cur_frame.empty()) {
                continue;
            }

               cur_frame_num = new_frame_num;

            Mat display_frame = cur_frame.clone();
            drawRectangle(display_frame);

            imshow("Video Player", display_frame);
        }
        else if (key == 27) {  // ESC 키 입력 시 종료
            cerr << "Program Terminated!\n";
            break;
        }

        int64 end_time = getTickCount();
        double spent_time = (end_time - start_time) * 1000.0 / getTickFrequency();  // 처리에 걸린 시간 계산
        if (spent_time < time_per_frame) {
            usleep((time_per_frame - spent_time) * 1000); // 남은 시간만큼 대기
        }

        // 이전 프레임 업데이트
        prev_frame = cur_frame.clone();
    }

    return 0;
}
