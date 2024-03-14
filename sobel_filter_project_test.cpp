#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

#define REDi 54
#define GREENi 183
#define BLUEi 18

#define REDf 0.2126f
#define GREENf 0.7152f
#define BLUEf 0.0722f

#define ESC 27

#define GX1 {1, 0, -1, 2, -2, 1, 0, -1}
#define GY1 {1, 2, 1, 0, 0, -1, -2, -1}

#define GX2 {-1, 1, 0, -2, 2, -1, 1, 0}
#define GY2 {1, 1, 2, 0, 0, -1, -1, -2}

#define GX3 {0, -1, 1, -2, 2, 0, -1, 1}
#define GY3 {2, 1, 1, 0, 0, -2, -1, -1}

struct ThreadData
{
    cv::Mat* input;
    cv::Mat* output;
};

void to442_grayscale(cv::Mat *rgbImage);
void to442_sobel(ThreadData *data);

cv::Mat frame_split_stitch(cv::Mat& frame);
void frame_to_sobel(ThreadData *data);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file>" << std::endl;
        return 1;
    }
    
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
		std::cerr << "Error: Couldn't open the video file." << std::endl;
        return 1;
	}
		
	while (true) {
		
		cv::Mat frame;
		cap >> frame;
		if (frame.empty())
			break;
		
		cv::Mat sobelFiltered = frame_split_stitch(frame);
		
		cv::imshow("Processed Frame", sobelFiltered);
		
		int key = cv::waitKey(30);
		if (key == ESC)
			break;
	}
	
	cap.release();
	cv::destroyAllWindows();
}

cv::Mat frame_split_stitch(cv::Mat& frame)
{
	cv::Size sz = frame.size();
    int cols = sz.width;
    int rows = sz.height;
    //int midRow = rows / 2;
    //int midCol = cols / 2;

    int line1 = rows / 4;    
    int line2 = rows / 2;
    int line3 = line1 + line2;
    
    //printf("%d %d %d %d\n", line1, line2, line3, rows);
    //exit(0);
    
    /*cv::Rect quad1(0, 0, cols, line1 + 1); // strip1
    cv::Rect quad2(0, line1 - 1, cols, line2 - line1 + 2);
    cv::Rect quad3(0, line2 - 1, cols, line3 - line2 + 2);
    cv::Rect quad4(0, line3 - 1, cols, rows - line3 + 1);*/
    
    cv::Rect quad1(0, 0, cols, 271); // strip1
    cv::Rect quad2(0, 269, cols, 272);
    cv::Rect quad3(0, 539, cols, 272);
    cv::Rect quad4(0, 809, cols, 271);
    
    /*cv::Rect quad1(0, 0, cols, line1); // strip1
    cv::Rect quad2(0, line1, cols, line1);
    cv::Rect quad3(0, line2, cols, line1);
    cv::Rect quad4(0, line3, cols, line1);*/
    
    cv::Mat quad1Mat = frame(quad1);
    cv::Mat quad2Mat = frame(quad2);
    cv::Mat quad3Mat = frame(quad3);
    cv::Mat quad4Mat = frame(quad4);
    
    //printf("%d %d %d %d\n", quad1Mat.size().height, quad2Mat.size().height, quad3Mat.size().height, quad4Mat.size().height);
    //exit(0);
    
    cv::Mat sobelFiltered1(quad1Mat.size().height-2, quad1Mat.size().width-2, CV_8UC3);
    cv::Mat sobelFiltered2(quad2Mat.size().height-2, quad2Mat.size().width-2, CV_8UC3);
    cv::Mat sobelFiltered3(quad3Mat.size().height-2, quad3Mat.size().width-2, CV_8UC3);
    cv::Mat sobelFiltered4(quad4Mat.size().height-2, quad4Mat.size().width-2, CV_8UC3);
    
    pthread_t frame_to_sobel_1_thread, frame_to_sobel_2_thread, frame_to_sobel_3_thread, frame_to_sobel_4_thread;
    
    ThreadData data1 = { &quad1Mat, &sobelFiltered1 };
    ThreadData data2 = { &quad2Mat, &sobelFiltered2 };
    ThreadData data3 = { &quad3Mat, &sobelFiltered3 };
    ThreadData data4 = { &quad4Mat, &sobelFiltered4 };
    
    pthread_create(&frame_to_sobel_1_thread, NULL, (void* (*)(void*)) frame_to_sobel, &data1);
    pthread_create(&frame_to_sobel_2_thread, NULL, (void* (*)(void*)) frame_to_sobel, &data2);
    pthread_create(&frame_to_sobel_3_thread, NULL, (void* (*)(void*)) frame_to_sobel, &data3);
    pthread_create(&frame_to_sobel_4_thread, NULL, (void* (*)(void*)) frame_to_sobel, &data4);
    
    pthread_join(frame_to_sobel_1_thread, NULL);
    pthread_join(frame_to_sobel_2_thread, NULL);
    pthread_join(frame_to_sobel_3_thread, NULL);
    pthread_join(frame_to_sobel_4_thread, NULL);
    
    cv::Mat sobelFiltered, sobelFilteredTop, sobelFilteredBottom;
    
    cv::vconcat(sobelFiltered1, sobelFiltered2, sobelFilteredTop);
    cv::vconcat(sobelFiltered3, sobelFiltered4, sobelFilteredBottom);
    cv::vconcat(sobelFilteredTop, sobelFilteredBottom, sobelFiltered);
    
    //printf("%d\n", sobelFiltered.size().height);
    //exit(0);
    
    return sobelFiltered;
	}

void frame_to_sobel(ThreadData *data)
{
	to442_grayscale(data->input);
	to442_sobel(data);
}

void to442_grayscale(cv::Mat *rgbImage)
{	
    cv::Size sz = rgbImage->size();
    int imageWidth = sz.width;
    int imageHeight = sz.height;
    printf("%d %d\n", imageWidth, imageHeight);
    int i = 0; // height (row) index
    int j = 0; // width (column) index

    for (i = 0; i < imageHeight; i++)
    {
        for (j = 0; j < imageWidth; j+=8)
        {
			if (0 && j + 8 <= imageWidth)
			{
				uint8_t *pixel = rgbImage->ptr<uint8_t>(i, j); // Get a pointer to the pixel
				
				// Load four pixels into NEON vectors
				uint8x8x3_t pixels = vld3_u8(pixel);
				
				// Convert RGB to grayscale using NEON vectors
				uint16x8_t blue = vmull_u8(pixels.val[0], vdup_n_u8(BLUEi));
				uint16x8_t green = vmull_u8(pixels.val[1], vdup_n_u8(GREENi));
				uint16x8_t red = vmull_u8(pixels.val[2], vdup_n_u8(REDi));
				uint16x8_t gray = vaddq_u16(blue, vaddq_u16(green, red));
				
				// Shift right by 8 bits to divide by 256 (equivalent to multiplying by 1/256)
				gray = vshrq_n_u16(gray, 8);
				
				// Pack the 16-bit grayscale values to 8-bit
				uint8x8_t gray_u8 = vqmovn_u16(gray);
				
				pixels.val[0] = gray_u8;
				
				// Store the grayscale values back to memory
				vst3_u8((uint8_t*)pixel, pixels);
			}
            else
            {
				uint8_t *pixel = rgbImage->ptr<uint8_t>(j, i); // Vec<uchar, 3>

				// set pixel[0] to grayscale value
				*pixel = pixel[0] * BLUEf + pixel[1] * GREENf + pixel[2] * REDf;
				
				j -= 7;
			}
        }
    }
}


void to442_sobel(ThreadData *data) 
{
    cv::Size sz = data->input->size();
    int imageWidth = sz.width;
    int imageHeight = sz.height;
    //printf("%d %d %d %d\n", imageWidth, imageHeight, data->output->size().width, data->output->size().height);
    
    int16_t gx1[] = GX1;
    int16_t gy1[] = GY1;
    int16_t gx2[] = GX2;
    int16_t gy2[] = GY2;
    int16_t gx3[] = GX3;
    int16_t gy3[] = GY3;
    int16_t grayValues[8];

    int j = 0; // width (column) index
    int i = 0; // height (row) index
    
    int16x8_t grayValues_vector, gx_sum, gy_sum, sum_vector;
    int sum;
    
    // test code
    /*for (j = 1; j < imageHeight - 1; j++)
    {
		for (i = 1; i < imageWidth - 1; i++)
		{
			memset(data->output->ptr<uint8_t>(j - 1, i - 1), data->input->ptr<uint8_t>(j, i)[0], 3);
		}
	}
	
	for (i = 0; i < imageWidth - 2; i++)
		if (data->output->ptr<uint8_t>(0, i)[0] > 0)
			printf("success!\n");
	return;*/
    
    for (j = 1; j < imageHeight - 1; j++)
    {
		int index = 0;
		i = 1;
		for (int dj = -1; dj <= 1; ++dj)
		{
			for (int di = -1; di <= 1; ++di)
			{
				if (!(di == 0 && dj == 0))
				{
					int nj = j + dj;
					int ni = i + di;

					grayValues[index] = (int16_t) data->input->ptr<uint8_t>(nj, ni)[0];
					index++;
				}
			}      
		}
		
		grayValues_vector = vld1q_s16(grayValues);
		gx_sum = vmulq_s16(vld1q_s16(gx1), grayValues_vector);
		gy_sum = vmulq_s16(vld1q_s16(gy1), grayValues_vector);				
		sum_vector = vaddq_s16(gx_sum, gy_sum);				
		sum = abs((int) vaddvq_s16(sum_vector));							
		sum = std::min(sum, 255); // Clamp sum value to 255		
		memset(data->output->ptr<uint8_t>(j - 1, i - 1), sum, 3);
		
        for (i = 2; i < imageWidth - 3; i++)
        {
			//get next pixels into left side
			//do the math with gx2
			grayValues[0] = data->input->ptr<uint8_t>(j-1, i+1)[0];
			grayValues[3] = data->input->ptr<uint8_t>(j, i+1)[0];
			grayValues[4] = data->input->ptr<uint8_t>(j, i-1)[0];
			grayValues[5] = data->input->ptr<uint8_t>(j+1, i+1)[0];
			grayValues_vector = vld1q_s16(grayValues);
			gx_sum = vmulq_s16(vld1q_s16(gx2), grayValues_vector);
			gy_sum = vmulq_s16(vld1q_s16(gy2), grayValues_vector);				
			sum_vector = vaddq_s16(gx_sum, gy_sum);				
			sum = abs((int) vaddvq_s16(sum_vector));							
			sum = std::min(sum, 255); // Clamp sum value to 255			
			memset(data->output->ptr<uint8_t>(j - 1, i - 1), sum, 3);
			i++;
			
			//get next pixels into center
			//do the math with gx3
			grayValues[1] = data->input->ptr<uint8_t>(j-1, i+1)[0];
			grayValues[3] = data->input->ptr<uint8_t>(j, i+1)[0];
			grayValues[4] = data->input->ptr<uint8_t>(j, i-1)[0];
			grayValues[6] = data->input->ptr<uint8_t>(j+1, i+1)[0];
			grayValues_vector = vld1q_s16(grayValues);
			gx_sum = vmulq_s16(vld1q_s16(gx3), grayValues_vector);
			gy_sum = vmulq_s16(vld1q_s16(gy3), grayValues_vector);				
			sum_vector = vaddq_s16(gx_sum, gy_sum);				
			sum = abs((int) vaddvq_s16(sum_vector));							
			sum = std::min(sum, 255); // Clamp sum value to 255			
			memset(data->output->ptr<uint8_t>(j - 1, i - 1), sum, 3);
			i++;
			
			//get next pixels into right side
			//do the math with gx1
			grayValues[2] = data->input->ptr<uint8_t>(j-1, i+1)[0];
			grayValues[3] = data->input->ptr<uint8_t>(j, i+1)[0];
			grayValues[4] = data->input->ptr<uint8_t>(j, i-1)[0];
			grayValues[7] = data->input->ptr<uint8_t>(j+1, i+1)[0];
			grayValues_vector = vld1q_s16(grayValues);
			gx_sum = vmulq_s16(vld1q_s16(gx1), grayValues_vector);
			gy_sum = vmulq_s16(vld1q_s16(gy1), grayValues_vector);				
			sum_vector = vaddq_s16(gx_sum, gy_sum);				
			sum = abs((int) vaddvq_s16(sum_vector));							
			sum = std::min(sum, 255); // Clamp sum value to 255			
			memset(data->output->ptr<uint8_t>(j - 1, i - 1), sum, 3);
        }
        
        if (i == imageWidth - 3)
        {
			//get next pixels into left side
			//do the math with gx2
			grayValues[0] = data->input->ptr<uint8_t>(j-1, i+1)[0];
			grayValues[3] = data->input->ptr<uint8_t>(j, i+1)[0];
			grayValues[4] = data->input->ptr<uint8_t>(j, i-1)[0];
			grayValues[5] = data->input->ptr<uint8_t>(j+1, i+1)[0];
			grayValues_vector = vld1q_s16(grayValues);
			gx_sum = vmulq_s16(vld1q_s16(gx2), grayValues_vector);
			gy_sum = vmulq_s16(vld1q_s16(gy2), grayValues_vector);				
			sum_vector = vaddq_s16(gx_sum, gy_sum);				
			sum = abs((int) vaddvq_s16(sum_vector));							
			sum = std::min(sum, 255); // Clamp sum value to 255			
			memset(data->output->ptr<uint8_t>(j - 1, i - 1), sum, 3);
			i++;
		}
		
        if (i == imageWidth - 2)
        {
			//get next pixels into center
			//do the math with gx3
			grayValues[1] = data->input->ptr<uint8_t>(j-1, i+1)[0];
			grayValues[3] = data->input->ptr<uint8_t>(j, i+1)[0];
			grayValues[4] = data->input->ptr<uint8_t>(j, i-1)[0];
			grayValues[6] = data->input->ptr<uint8_t>(j+1, i+1)[0];
			grayValues_vector = vld1q_s16(grayValues);
			gx_sum = vmulq_s16(vld1q_s16(gx3), grayValues_vector);
			gy_sum = vmulq_s16(vld1q_s16(gy3), grayValues_vector);				
			sum_vector = vaddq_s16(gx_sum, gy_sum);				
			sum = abs((int) vaddvq_s16(sum_vector));							
			sum = std::min(sum, 255); // Clamp sum value to 255			
			memset(data->output->ptr<uint8_t>(j - 1, i - 1), sum, 3);
		}
    }
}
