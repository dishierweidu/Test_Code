#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>

using namespace cv;
using namespace std;

Mat src;
void on_mouse(int event, int x, int y, int flags, void* ustc);

int main(int argc, char* argv)
{
    src = imread("233.jpg");

    namedWindow("input image", WINDOW_AUTOSIZE);
    setMouseCallback("input image", on_mouse, &src);

    while (1)
    {
        imshow("input image", src);
        waitKey(40);
    }
    return 0;
}

void on_mouse(int event, int x, int y, int flags, void* ustc)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        Mat p;
        p = *(Mat*)ustc;
        Point pt = Point(x, y);
        char temp[16];
        sprintf(temp,"(%d,%d)", pt.x, pt.y);
        putText(src, temp, pt, FONT_HERSHEY_COMPLEX,0.5, Scalar(255, 0, 0),2,8);
        printf("b=%d\t", p.at<Vec3b>(pt)[0]);
        printf("g=%d\t", p.at<Vec3b>(pt)[1]);
        printf("r=%d\n", p.at<Vec3b>(pt)[2]);
        circle(src, pt, 2, Scalar(0, 0, 255), 2, 8);
    }

}