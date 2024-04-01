#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//Number of threads in block

//#define BLOCK_SIZE 256
//#define GRID_SIZE 1

int BLOCK_SIZE = 256;
int GRID_SIZE = 1;


struct ImgData
{
    int width,height;
    unsigned long step;
};

struct Color
{
    unsigned char b,g,r;
    __device__ char GetDom()
    {
        int max = b;
        if(max<g)
        {
            max=g;
        }
        if(max<r)
        {
            return 'r';
        }
        if(max==g)
        {
            return 'g';
        }
        return 'b';
    }
};

struct Point
{
    int x,y;
    __device__ void GetNeighbours(int width,int height,Point* neighbours)
    {
        // |l|lu|u|ru|r|rd|d|ld (over engineered)
        // enum POS
        // {
        //     LEFT = 0b10000000,
        //     LEFTUP = 0b01000000,
        //     UP = 0b00100000,
        //     RIGHTUP = 0b00010000,
        //     RIGHT = 0b00001000,
        //     RIGHTDOWN = 0b00000100,
        //     DOWN = 0b00000010,
        //     LEFTDOWN = 0b00000001
        // };
        // unsigned char pos_flag = 0;

        if(x>0)
        {
            neighbours[0]={x-1,y};
            if(y>0)
            {
                neighbours[1]={x-1,y-1};
            }
            
            if(y<height-1)
            {
                neighbours[7]={x-1,y+1};
            }
        }

        if(x<width-1)
        {
            neighbours[4]={x+1,y};
            if(y>0)
            {
                neighbours[3]={x+1,y-1};
            }
            
            if(y<height-1)
            {
                neighbours[5]={x+1,y+1};
            }
        }

        if(y>0)
        {
            neighbours[2]={x,y-1};
        }
        
        if(y<height-1)
        {
            neighbours[6]={x,y+1};
        }
        

    }
};





__global__ void test_fun(unsigned char* input, ImgData data)
{
    //2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((xIndex<data.width) && (yIndex<data.height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * data.step + (3 * xIndex);
		
		//Location of gray pixel in output
		//const int gray_tid  = yIndex * grayWidthStep + xIndex;

		input[color_tid] = 255;  //B
		input[color_tid + 1] = 0; //G
		input[color_tid + 2] = 0;  //R

		//const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

		//output[gray_tid] = static_cast<unsigned char>(gray);
	}

}

//threadIdx.x - thread index
//blockDim.x - number of threads in block
//blockIdx.x - block index in grid
//gridDim.x - grid size


__global__ void test_fun2(unsigned char* input, ImgData data)
{
    //2D Index of current thread
	//const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	//const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int step = blockDim.x*gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;;
    //long int size = data.height*data.width;
    for(unsigned int i=tid;i<data.height*data.width;i+=step)
    {
        unsigned int color_i = i * 3;
        Color c;
        c.r = input[color_i + 2];
        c.g = input[color_i + 1];
        c.b = input[color_i];
        unsigned char max = c.r;
        if (c.g>max)
        {
            max = c.g;
        }
        if(c.b>max)
        {
            max = c.b;
        }

        if(max == c.r)
        {
            input[color_i] = 0;  //B
            input[color_i + 1] = 0;//G
            input[color_i + 2] = 255;//R
        }
        else if(max == c.g)
        {
            input[color_i] = 0;  //B
            input[color_i + 1] = 255;//G
            input[color_i + 2] = 0;//R
        }
        else if(max == c.b)
        {
            input[color_i] = 255;  //B
            input[color_i + 1] = 0;//G
            input[color_i + 2] = 0;//R
        }




    }

}


__global__ void DomOnly(Color* input,Color* output, ImgData data)
{
    //Step so threads won't overlap
    int step = blockDim.x * gridDim.x;
    //Starting possition for thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //long int size = data.height*data.width;
    for(int i=tid;i<data.height*data.width;i+=step)
    {
        
        switch (input[i].GetDom())
        {
        case 'r':
            output[i].r=input[i].r;
            output[i].g=0;
            output[i].b=0;
            break;
        
        case 'g':
            output[i].r=0;
            output[i].g=input[i].g;
            output[i].b=0;
            break;
        
        case 'b':
            output[i].r=0;
            output[i].g=0;
            output[i].b=input[i].b;
            break;
        
        default:
            break;
        }
    }

}


__global__ void MaxColor(Color* input,Color* output, ImgData data)
{
    //Step so threads won't overlap
    int step = blockDim.x * gridDim.x;
    //Starting possition for thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //long int size = data.height*data.width;
    for(int i=tid;i<data.height*data.width;i+=step)
    {
        Point pos = {.x=tid % data.width, .y=tid / data.width};
        Point neighbours[8];
        for(int j=0;j<8;j++)
        {
            neighbours[j]={-1,-1};
        }

        pos.GetNeighbours(data.width,data.height,neighbours);

        Color max_color={input[i].r,input[i].g,input[i].g};

        for(int j=0;j<8;j++)
        {
            if(neighbours[j].x==-1)
            {
                continue;
            }
            if(input[neighbours[j].y*data.width+neighbours[j].x].r>max_color.r)
            {
                max_color.r=input[neighbours[j].y*data.width+neighbours[j].x].r;
            }
            if(input[neighbours[j].y*data.width+neighbours[j].x].g>max_color.g)
            {
                max_color.g=input[neighbours[j].y*data.width+neighbours[j].x].g;
            }
            if(input[neighbours[j].y*data.width+neighbours[j].x].b>max_color.b)
            {
                max_color.b=input[neighbours[j].y*data.width+neighbours[j].x].b;
            }
        }        

        output[i].r=max_color.r;
        output[i].g=max_color.g;
        output[i].b=max_color.b;

    }

}

__global__ void MinColor(Color* input,Color* output, ImgData data)
{
    //Step so threads won't overlap
    int step = blockDim.x * gridDim.x;
    //Starting possition for thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //long int size = data.height*data.width;
    for(int i=tid;i<data.height*data.width;i+=step)
    {
        Point pos = {.x=tid % data.width, .y=tid / data.width};
        Point neighbours[8];
        for(int j=0;j<8;j++)
        {
            neighbours[j]={-1,-1};
        }

        pos.GetNeighbours(data.width,data.height,neighbours);

        Color min_color={input[i].r,input[i].g,input[i].g};

        for(int j=0;j<8;j++)
        {
            if(neighbours[j].x==-1)
            {
                continue;
            }
            if(input[neighbours[j].y*data.width+neighbours[j].x].r<min_color.r)
            {
                min_color.r=input[neighbours[j].y*data.width+neighbours[j].x].r;
            }
            if(input[neighbours[j].y*data.width+neighbours[j].x].g<min_color.g)
            {
                min_color.g=input[neighbours[j].y*data.width+neighbours[j].x].g;
            }
            if(input[neighbours[j].y*data.width+neighbours[j].x].b<min_color.b)
            {
                min_color.b=input[neighbours[j].y*data.width+neighbours[j].x].b;
            }
        }        

        output[i].r=min_color.r;
        output[i].g=min_color.g;
        output[i].b=min_color.b;

    }

}

void what_it_does(cv::Mat& input, cv::Mat& output)
{
    //Step stores number of bytes that one row takes, rows stores how many rows matrix has;
    int colorBytes = input.step * input.rows;

    unsigned char *d_input;
    unsigned char *test_input = (unsigned char *)malloc(colorBytes*sizeof(unsigned char));

	// Allocate device memory
    cudaMalloc<unsigned char>(&d_input,colorBytes);

	// Copy data from OpenCV input image to device memory
    cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice);
    
    //std::cout << input.row(0) << std::endl;
    if(false)
    {
        std::ofstream out("res.txt");
        for(int y=0;y<input.rows;y++)
        {
            for(int x=0;x<input.step;x+=3)
            {
                out << "#"<< std::hex << (int)(*(input.ptr()+2+x+y*input.step)); //R
                out << std::hex << (int)(*(input.ptr()+1+x+y*input.step));       //G
                out << std::hex << (int)(*(input.ptr()+x+y*input.step)) << " ";  //B
                
                test_input[x+y*input.step] = *(input.ptr()+x+y*input.step);      //B
                test_input[1+x+y*input.step] = *(input.ptr()+1+x+y*input.step);  //G
                test_input[2+x+y*input.step] = *(input.ptr()+2+x+y*input.step);  //R
            }
            out << std::endl;
        }
        out.close();
    }

	// Specify a reasonable block size
    int block_size=16;
    const dim3 block(block_size,block_size);
    
    //std::cout << block.x << std::endl;
    //std::cout << block.y << std::endl;
    //std::cout << block.z << std::endl;
    const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);
    
    //std::cout << grid.x << std::endl;
    //std::cout << grid.y << std::endl;
    //std::cout << grid.z << std::endl;

    ImgData data = {input.cols,input.rows,input.step};

    const dim3 donno1(BLOCK_SIZE);
    const dim3 donno2(GRID_SIZE);

    //test_fun<<<grid,block>>>(d_input,data);
    test_fun2<<<GRID_SIZE,BLOCK_SIZE>>>(d_input,data);
    //test_fun2<<<donno2,donno1>>>(d_input,data);

    cudaError res = cudaDeviceSynchronize();
    if(res)
    {
        
        std::cout << "CUDA: " << cudaGetErrorName(res) << std::endl;
    }
    cudaMemcpy(input.ptr(),d_input,colorBytes,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    
    //memcpy(input.ptr(),test_input,colorBytes);

    free(test_input);


}


void what_it_does2(cv::Mat& input, cv::Mat& output)
{
    //Step stores number of bytes that one row takes, rows stores how many rows matrix has;
    int colorBytes = input.step * input.rows;
    //int dimmensions = input.rows * input.cols; 


    Color *d_input;
    Color *d_output;

	// Allocate device memory
    cudaMalloc<Color>(&d_input,colorBytes);
    cudaMalloc<Color>(&d_output,colorBytes);

	// Copy data from OpenCV input image to device memory
    cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice);
    
    ImgData data = {input.cols,input.rows,input.step};

    //DomOnly<<<GRID_SIZE,BLOCK_SIZE>>>(d_input,d_output,data);
    //MaxColor<<<GRID_SIZE,BLOCK_SIZE>>>(d_input,d_output,data);
    MinColor<<<GRID_SIZE,BLOCK_SIZE>>>(d_input,d_output,data);

    cudaError res = cudaDeviceSynchronize();
    if(res)
    {
        std::cout << "CUDA: " << cudaGetErrorName(res) << std::endl;
    }
    
    cudaMemcpy(output.ptr(),d_output,colorBytes,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);


}



int main(int argc, char** argv)
{
    int cudaDevices;
    cudaGetDeviceCount(&cudaDevices);
    std::cout << "Cuda devices found: " << cudaDevices << std::endl;
    if(argc<2)
    {
        std::cout << "Usage " << argv[0] << " path/to/img" << std::endl;
    }
    if(argc>2)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop,0);
        //std::cout << "Max Threads per Block: " << dev_prop.maxThreadsPerBlock << std::endl;
        //std::cout << "Max Blocks: " << dev_prop.maxBlocksPerMultiProcessor << std::endl;
        //std::cout << "Max Grid Size: " << dev_prop.maxGridSize[0]  << std::endl;
        //std::cout << "Max Grid Size (2D): " << dev_prop.maxGridSize[1]  << std::endl;
        //std::cout << "Max Grid Size (3D): " << dev_prop.maxGridSize[2]  << std::endl;
        BLOCK_SIZE = std::atoi(argv[2]);
        if(BLOCK_SIZE>dev_prop.maxThreadsPerBlock)
        {
            std::cout<<"CUDA: Can't use " << BLOCK_SIZE << " threads in block, changed to " << dev_prop.maxThreadsPerBlock << " instead\n";
            BLOCK_SIZE = dev_prop.maxThreadsPerBlock;
        }
    }
    
	std::string imagePath = argv[1];

    cv::Mat input = cv::imread(imagePath,cv::IMREAD_COLOR);
	cv::Mat output(input.rows,input.cols,CV_8UC3);


    //cv::imshow("Before",input);
    //what_it_does(input,output);
    what_it_does2(input,output);
    //cv::imshow("After",output);

    // bool run=true;
    // while(run)
    // {
    //     int key = cv::waitKeyEx();
    //     // if(key >= 0)
    //     // {
    //     //    std::cout << key << std::endl;
        
    //     // }
    //     if(key == 1048603)
    //     {
    //         run = false;
    //     }
    // }

    cv::waitKey();

    //std::cout<< CUDA_VERSION << std::endl;
    //std::cout<< CUDART_VERSION << std::endl;

    return EXIT_SUCCESS;
}

