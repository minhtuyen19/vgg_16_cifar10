#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE  
#define _CRT_NONSTDC_NO_DEPRECATE


#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h>

void readfile(const char* filename, float a[], int* n) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Can't open file %s\n", filename);
        exit(1);
    }
    else {
        for (int i = 0; i <*n; i++) {
            fscanf(file, "%f", &a[i]);
        }
        fclose(file);
    }
}
void writefile1d(const char* filename, float a[], int n) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Can't open file %s\n", filename);
        exit(1);
    }
    else {
        for (int i = 0; i < n; i++) {
            fprintf(file, "%.32f", a[i]);
        }
        fclose(file);
    }
}
void xuat(float a[], int n) {
    for (int i = 0; i < n; i++) {
        printf("a[%d] = %0.10f ", i, a[i]);
    }
    printf("\n");
}


float mean[3] = { 0.49139968, 0.48215841, 0.44653091 };
float stdard[3] = { 0.24703223, 0.24348513, 0.26158784 };


void normalize(float* image, int height, int width, int channels, float* output)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float value = image[y * width * channels + x * channels + c];
                value /= 255.0;
                value -= mean[c];
                value /= stdard[c];
                output[y * width * channels + x * channels + c] = value;
            }
        }
    }
}
void transpose(float* input, int row, int col, int channels, float* output) {
    for (int c = 0; c < col; c++) { //col = 3
        for (int r = 0; r < row; r++) { //row = 32
            for (int ch = 0; ch < channels; ch++) {
                int input_idx = r * row * col + ch * col + c;
                int output_idx = c * row * channels + r * row + ch;
                output[output_idx] = input[input_idx];
            }
        }
    }
}

void relu(float* input, int n, float* out) {
	for (int i = 0; i < n; i++) {
		if (input[i] < 0) {
			input[i] = 0;
		}
		out[i] = input[i];
	}
}


//===============================Conv2d============================================================
    void conv2d(float* input, float* w, int row, int col, int frow, int fcol, int channel_inputs,
        int numofkernel, float* conv_results, float* bias, int padding) {

        int out_row = row - frow + 2 * padding + 1;
        int out_col = col - fcol + 2 * padding + 1;

        //Allocate memory for kernel
        float**** kernel_out = (float****)malloc(numofkernel * sizeof(float***));
            for (int k = 0; k < numofkernel; k++) {
                kernel_out[k] = (float***)malloc(channel_inputs * sizeof(float**));
                for (int l = 0; l < channel_inputs; l++) {
                    kernel_out[k][l] = (float**)malloc(frow * sizeof(float*));
                    for (int i = 0; i < frow; i++) {
                        kernel_out[k][l][i] = (float*)malloc(fcol * sizeof(float));
                        for (int j = 0; j < fcol; j++) {
                            int kernel_idx = k * channel_inputs * frow * fcol + l * frow * fcol + i * fcol + j;
                            kernel_out[k][l][i][j] = w[kernel_idx];
                        }
                    }
                }
            }

        //Add padding for input
        int padded_row = row + 2 * padding;
        int padded_col = col + 2 * padding;
       float* padded_input = (float*)calloc(padded_row * padded_col * channel_inputs, sizeof(float));


        for (int l = 0; l < channel_inputs; l++) {
            for (int i = 0; i < padded_row; i++) {
                for (int j = 0; j < padded_col; j++) {
                    int padded_idx = l * padded_row * padded_col + i * padded_col + j;
                    padded_input[padded_idx] = 0.0;
                }
            }
        }

        //Gan input vao vien padding
        for (int l = 0; l < channel_inputs; l++) {
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    int input_idx = l * row * col + i * col + j;
                    int padded_idx = l * padded_row * padded_col + (i + padding) * padded_col + j + padding;
                    padded_input[padded_idx] = input[input_idx];
                }
            }
        }

        //Calculate Convolution
        for (int k = 0; k < numofkernel; k++) {
            for (int i = 0; i < out_row; i++) {
                for (int j = 0; j < out_col; j++) {
                    float sum = 0.0;
                    for (int l = 0; l < channel_inputs; l++) {
                        for (int m = 0; m < frow; m++) {
                            for (int n = 0; n < fcol; n++) {
                                int input_idx = l * padded_row * padded_col + (i + m) * padded_col + j + n;
                                int kernel_idx = k * channel_inputs * frow * fcol + l * frow * fcol + m * fcol + n;
                                sum += padded_input[input_idx] * kernel_out[k][l][m][n];
                            }
                        }
                    }
                    int h = k * out_row * out_col + i * out_col + j;
                    conv_results[h] = sum + bias[k];
                }
            }
        }

    for (int k = 0; k < numofkernel; k++) {
        for (int l = 0; l < channel_inputs; l++) {
            for (int i = 0; i < frow; i++) {
                free(kernel_out[k][l][i]);
            }
            free(kernel_out[k][l]);
        }
        free(kernel_out[k]);
    }
    free(kernel_out);
}
//=====================================Batch_Normalization_first =========================================
void batch_norm(float* input, float* output, float* mean, float* variance, float* scale, float* shift, int batch_size, int channels, int row, int col) {
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < row * col; i++) {
                int idx = b * channels * row * col + c * row * col + i;
                float normalized_val = (input[idx] - mean[c]) / sqrt(variance[c] + 1e-3);
                output[idx] = scale[c] * normalized_val + shift[c];
            }
        }
    }
}
//==================================Maxpool2d=============================================================
//Maxpool cut out with padding = 1, stride = 2
void maxpool2d(float* input, int row, int col, int pool_size, int channel_size, int stride, float* output_max) {
    int output_row = ((row - pool_size) / stride) + 1;
    int output_col = ((col - pool_size) / stride) + 1;
    for (int k = 0; k < channel_size; k++) {
        for (int i = 0; i < output_row; i++) {
            for (int j = 0; j < output_col; j++) {
                float max_val = input[k * row * col + i * stride * col + j * stride];
                for (int m = 0; m < pool_size; m++) {
                    for (int n = 0; n < pool_size; n++) {
                        float curr_val = input[k * row * col + (i * stride + m) * col + j * stride + n];
                        if (curr_val > max_val) {
                            max_val = curr_val;
                        }
                    }
                }
                output_max[k * output_row * output_col + i * output_col + j] = max_val;
            }
        }
    }
}
//====================================Dense================================================================
void transposedense(float* input, int height, int width, int channels, float* output) {
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                output[h * width * channels + w * channels + c] = input[c * height * width + h * width + w];
            }
        }
    }
}

void dense(float* input, int n_input, int n_w, float* weights, float* bias, float* output) {

    float* dot = (float*)malloc(n_w * sizeof(float));

    for (int i = 0; i < n_w; i++) {
        dot[i] = 0;
        for (int j = 0; j < n_input; j++) {
            int h = i * n_input + j;
            dot[i] += input[j] * weights[h];
            //if (h == 448) {
            //	printf(" input[%d] = %f ", j, input[j]);
            //}
        }
        //printf("dot[%d] = %f ", i, dot[i]);
        output[i] = dot[i] + bias[i];
    }

    //for (int i = 0; i < n_w; i++) {
    //	printf("output[%i] = %f \n", i, output[i]);
    //}

    free(dot);
}

void softmax(float* input, int n_input, float* output) {
    float max_val = input[0];
    for (int i = 0; i < n_input; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    float exp_sum = 0.0;
    for (int i = 0; i < n_input; i++) {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        exp_sum += exp_val;
    }
    for (int i = 0; i < n_input; i++) {
        output[i] /= exp_sum;
    }
}

int argmax(float* arr, int size) {
    const char* a[] = { "airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck" };
    float max_val = arr[0];
    int max_idx = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    if (max_idx == 0) {
        printf("Gia tri du doan: %s\n", a[0]);
    }
    else if (max_idx == 1) {
        printf("Gia tri du doan: %s\n", a[1]);
    }
    else if (max_idx == 2) {
        printf("Gia tri du doan: %s\n", a[2]);
    }
    else if (max_idx == 3) {
        printf("Gia tri du doan: %s\n", a[3]);
    }
    else if (max_idx == 4) {
        printf("Gia tri du doan: %s\n", a[4]);
    }
    else if (max_idx == 5) {
        printf("Gia tri du doan: %s\n", a[5]);
    }
    else if (max_idx == 6) {
        printf("Gia tri du doan: %s\n", a[6]);
    }
    else if (max_idx == 7) {
        printf("Gia tri du doan: %s\n", a[7]);
    }
    else if (max_idx == 8) {
        printf("Gia tri du doan: %s\n", a[8]);
    }
    else if (max_idx == 9) {
        printf("Gia tri du doan: %s\n", a[9]);
    }
    else {
        return -1;
    }
    return max_idx;
}


static float w_0[3 * 3 * 3 * 64];
void main() {
    // Get the start time.
    time_t start_time = time(NULL);
    //========================Input_and_Normalization========================
    float* image = (float*)malloc((32 * 32 * 3) * sizeof(float));
    float* output = (float*)malloc((32 * 32 * 3) * sizeof(float));
    float* image_transpose = (float*)malloc((32 * 32 * 3) * sizeof(float));
    int n_image = 32 * 32 * 3;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/image_bird_0.txt", image, &n_image);
    // xuat(image, n_image);
    normalize(image, 32, 32, 3, output);
    //xuat(output, 32 * 32 * 3);
    transpose(output, 32, 3, 32, image_transpose);
    //writefile1d("Normalized_Input.txt", image_transpose, n_image);
    //xuat(image_transpose, (32 * 32 * 3));
    // 
 /*   float* input = (float*)malloc((32 * 32 * 3) * sizeof(float));
    int n_input = (32 * 32 * 3);
    readfile("ship_n.txt", input, n_input);*/
    //=========================Conv2d_0======================================
    //float* w_0 = (float*)malloc((3 * 3 * 3 * 64) * sizeof(float));
    float* out_0 = (float*)malloc((32 * 32 * 64) * sizeof(float));
    float* out_relu_0 = (float*)malloc((32 * 32 * 64) * sizeof(float));
    float bias_0[64];
    int nw_0 = (3 * 3 * 3 * 64);
    int nb_0 = 64;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_0.txt", w_0, &nw_0);
    // xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_0.txt", bias_0, &nb_0);
    //xuat(bias_0, 64);
    conv2d(image_transpose, w_0, 32, 32, 3, 3, 3, 64, out_0, bias_0, 1);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_0.txt", out_0, (32 * 32 * 64));
    relu(out_0, (32 * 32 * 64), out_relu_0);
    // xuat(out_relu_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_0.txt", out_relu_0, (32 * 32 * 64));
    //===========================Batch_Normalization_0========================
    float* out_batch_0 = (float*)malloc((32 * 32 * 64) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_0 = (32 * 32 * 64);
    float mean_0[64];
    float variance_0[64];
    float scale_0[64];
    float shift_0[64];
    int n_mean_0 = 64;
    int n_variance_0 = 64;
    int n_scale_0 = 64;
    int n_shift_0 = 64;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_0/Mean.txt", mean_0, &n_mean_0);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_0/Variance.txt", variance_0, &n_variance_0);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_0/Scale.txt", scale_0, &n_scale_0);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_0/Shift.txt", shift_0, &n_shift_0);
    batch_norm(out_relu_0, out_batch_0, mean_0, variance_0, scale_0, shift_0, 1, 64, 32, 32);

    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_0.txt", out_batch_0, (32 * 32 * 64));
    //xuat(out_batch_0, (32 * 32 * 32));

    //////===========================Conv2d_1=====================================
    float* w_1 = (float*)malloc((3 * 3 * 64 * 64) * sizeof(float));
    float* out_1 = (float*)malloc((32 * 32 * 64) * sizeof(float));
    float* out_relu_1 = (float*)malloc((32 * 32 * 64) * sizeof(float));
    float bias_1[64];
    int nw_1 = (3 * 3 * 64 * 64);
    int nb_1 = 64;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_1.txt", w_1, &nw_1);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_1.txt", bias_1, &nb_1);
    conv2d(out_batch_0, w_1, 32, 32, 3, 3, 64, 64, out_1, bias_1, 1);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_1.txt", out_1, (32 * 32 * 64));
    relu(out_1, (32 * 32 * 64), out_relu_1);
    //xuat(out_relu_1, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_1.txt", out_relu_1, (32 * 32 * 64));

    ////===========================Batch_Normalization_1========================
    float* out_batch_1 = (float*)malloc((32 * 32 * 64) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_1 = (32 * 32 * 64);
    float mean_1[64];
    float variance_1[64];
    float scale_1[64];
    float shift_1[64];
    int n_mean_1 = 64;
    int n_variance_1 = 64;
    int n_scale_1 = 64;
    int n_shift_1 = 64;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_1/Mean.txt", mean_1, &n_mean_1);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_1/Variance.txt", variance_1, &n_variance_1);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_1/Scale.txt", scale_1, &n_scale_1);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_1/Shift.txt", shift_1, &n_shift_1);
    batch_norm(out_relu_1, out_batch_1, mean_1, variance_1, scale_1, shift_1, 1, 64, 32, 32);
    //xuat(out_batch_1, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_1.txt", out_batch_1, (32 * 32 * 64));

    //////===========================Maxpooling2d_0===============================
    float out_maxpool_0[16 * 16 * 64];
    int l_pool_0 = 16 * 16 * 64;
    maxpool2d(out_batch_1, 32, 32, 2, 64, 2, out_maxpool_0);
    //xuat(out_maxpool_0, 16 * 16 * 64);
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_maxpool_0.txt", out_maxpool_0, (16 * 16 * 64));
    //////===========================Conv2d_2=====================================
    float* w_2 = (float*)malloc((3 * 3 * 64 * 128) * sizeof(float));
    float* out_2 = (float*)malloc((16 * 16 * 128) * sizeof(float));
    float* out_relu_2 = (float*)malloc((16 * 16 * 128) * sizeof(float));
    float bias_2[128];
    int nw_2 = (3 * 3 * 64 * 128);
    int nb_2 = 128;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_2.txt", w_2, &nw_2);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_2.txt", bias_2, &nb_2);
    conv2d(out_maxpool_0, w_2, 16, 16, 3, 3, 64, 128, out_2, bias_2, 1);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_2.txt", out_2, (16 * 16 * 128));
    relu(out_2, (16 * 16 * 128), out_relu_2);
    //xuat(out_relu_2, (16 * 16 * 128));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_2.txt", out_relu_2, (16 * 16 * 128));
    ////===========================Batch_Normalization_2========================
    float* out_batch_2 = (float*)malloc((16 * 16 * 128) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_2 = (16 * 16 * 128);
    float mean_2[128];
    float variance_2[128];
    float scale_2[128];
    float shift_2[128];
    int n_mean_2 = 128;
    int n_variance_2 = 128;
    int n_scale_2 = 128;
    int n_shift_2 = 128;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_2/Mean.txt", mean_2, &n_mean_2);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_2/Variance.txt", variance_2, &n_variance_2);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_2/Scale.txt", scale_2, &n_scale_2);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_2/Shift.txt", shift_2, &n_shift_2);
    batch_norm(out_relu_2, out_batch_2, mean_2, variance_2, scale_2, shift_2, 1, 128, 16, 16);
    //xuat(out_batch_2, (16 * 16 * 128));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_2.txt", out_batch_2, (16 * 16 * 128));
    //////===========================Conv2d_3=====================================
    float* w_3 = (float*)malloc((3 * 3 * 128 * 128) * sizeof(float));
    float* out_3 = (float*)malloc((16 * 16 * 128) * sizeof(float));
    float* out_relu_3 = (float*)malloc((16 * 16 * 128) * sizeof(float));
    float bias_3[128];
    int nw_3 = (3 * 3 * 128 * 128);
    int nb_3 = 128;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_3.txt", w_3, &nw_3);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_3.txt", bias_3, &nb_3);
    conv2d(out_batch_2, w_3, 16, 16, 3, 3, 128, 128, out_3, bias_3, 1);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_3.txt", out_3, (16 * 16 * 128));
    relu(out_3, (16 * 16 * 128), out_relu_3);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_3.txt", out_relu_3, (16 * 16 * 128));
    ////===========================Batch_Normalization_3========================
    float* out_batch_3 = (float*)malloc((16 * 16 * 128) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_3 = (16 * 16 * 128);
    float mean_3[128];
    float variance_3[128];
    float scale_3[128];
    float shift_3[128];
    int n_mean_3 = 128;
    int n_variance_3 = 128;
    int n_scale_3 = 128;
    int n_shift_3 = 128;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_3/Mean.txt", mean_3, &n_mean_3);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_3/Variance.txt", variance_3, &n_variance_3);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_3/Scale.txt", scale_3, &n_scale_3);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_3/Shift.txt", shift_3, &n_shift_3);
    batch_norm(out_relu_3, out_batch_3, mean_3, variance_3, scale_3, shift_3, 1, 128, 16, 16);
    //xuat(out_batch_3, (16 * 16 * 128));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_3.txt", out_batch_3, (16 * 16 * 128));
    ////////===========================Maxpooling2d_1=============================
    float out_maxpool_1[8 * 8 * 128];
    int l_pool_1 = 8 * 8 * 128;
    maxpool2d(out_batch_3, 16, 16, 2, 128, 2, out_maxpool_1);
    //xuat(out_maxpool_1, 8 * 8 * 128);
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_maxpool_1.txt", out_maxpool_1, (8 * 8 * 128));
    //////===========================Conv2d_4=====================================
    float* w_4 = (float*)malloc((3 * 3 * 128 * 256) * sizeof(float));
    float* out_4 = (float*)malloc((8 * 8 * 256) * sizeof(float));
    float* out_relu_4 = (float*)malloc((8 * 8 * 256) * sizeof(float));
    float bias_4[256];
    int nw_4 = (3 * 3 * 128 * 256);
    int nb_4 = 256;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_4.txt", w_4, &nw_4);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_4.txt", bias_4, &nb_4);
    conv2d(out_maxpool_1, w_4, 8, 8, 3, 3, 128, 256, out_4, bias_4, 1);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_4.txt", out_4, (8 * 8 * 256));
    relu(out_4, (8 * 8 * 256), out_relu_4);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_3.txt", out_relu_4, (8 * 8 * 256));

    ////===========================Batch_Normalization_4========================
    float* out_batch_4 = (float*)malloc((8 * 8 * 256) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_4 = (8 * 8 * 256);
    float mean_4[256];
    float variance_4[256];
    float scale_4[256];
    float shift_4[256];
    int n_mean_4 = 256;
    int n_variance_4 = 256;
    int n_scale_4 = 256;
    int n_shift_4 = 256;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_4/Mean.txt", mean_4, &n_mean_4);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_4/Variance.txt", variance_4, &n_variance_4);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_4/Scale.txt", scale_4, &n_scale_4);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_4/Shift.txt", shift_4, &n_shift_4);
    batch_norm(out_relu_4, out_batch_4, mean_4, variance_4, scale_4, shift_4, 1, 256, 8, 8);
    //xuat(out_batch_3, (16 * 16 * 128));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_4.txt", out_batch_4, (8 * 8 * 256));

    //////===========================Conv2d_5=====================================
    float* w_5 = (float*)malloc((3 * 3 * 256 * 256) * sizeof(float));
    float* out_5 = (float*)malloc((8 * 8 * 256) * sizeof(float));
    float* out_relu_5 = (float*)malloc((8 * 8 * 256) * sizeof(float));
    float bias_5[256];
    int nw_5 = (3 * 3 * 256 * 256);
    int nb_5 = 256;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_5.txt", w_5, &nw_5);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_5.txt", bias_5, &nb_5);
    conv2d(out_batch_4, w_5, 8, 8, 3, 3, 256, 256, out_5, bias_5, 1);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_5.txt", out_5, (8 * 8 * 256));
    relu(out_5, (8 * 8 * 256), out_relu_5);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_5.txt", out_relu_5, (8 * 8 * 256));

    ////===========================Batch_Normalization_5========================
    float* out_batch_5 = (float*)malloc((8 * 8 * 256) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_5 = (8 * 8 * 256);
    float mean_5[256];
    float variance_5[256];
    float scale_5[256];
    float shift_5[256];
    int n_mean_5 = 256;
    int n_variance_5 = 256;
    int n_scale_5 = 256;
    int n_shift_5 = 256;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_5/Mean.txt", mean_5, &n_mean_5);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_5/Variance.txt", variance_5, &n_variance_5);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_5/Scale.txt", scale_5, &n_scale_5);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_5/Shift.txt", shift_5, &n_shift_5);
    batch_norm(out_relu_5, out_batch_5, mean_5, variance_5, scale_5, shift_5, 1, 256, 8, 8);
    //xuat(out_batch_3, (16 * 16 * 128));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_5.txt", out_batch_5, (8 * 8 * 256));

    //////===========================Conv2d_6=====================================
    float* w_6 = (float*)malloc((3 * 3 * 256 * 256) * sizeof(float));
    float* out_6 = (float*)malloc((8 * 8 * 256) * sizeof(float));
    float* out_relu_6 = (float*)malloc((8 * 8 * 256) * sizeof(float));
    float bias_6[256];
    int nw_6 = (3 * 3 * 256 * 256);
    int nb_6 = 256;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_6.txt", w_6, &nw_6);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_6.txt", bias_6, &nb_6);
    conv2d(out_batch_5, w_6, 8, 8, 3, 3, 256, 256, out_6, bias_6, 1);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_6.txt", out_6, (8 * 8 * 256));
    relu(out_6, (8 * 8 * 256), out_relu_6);
    //xuat(out_0, (32 * 32 * 64));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_6.txt", out_relu_6, (8 * 8 * 256));

    ////===========================Batch_Normalization_6========================
    float* out_batch_6 = (float*)malloc((8 * 8 * 256) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_6 = (8 * 8 * 256);
    float mean_6[256];
    float variance_6[256];
    float scale_6[256];
    float shift_6[256];
    int n_mean_6 = 256;
    int n_variance_6 = 256;
    int n_scale_6 = 256;
    int n_shift_6 = 256;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_6/Mean.txt", mean_6, &n_mean_6);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_6/Variance.txt", variance_6, &n_variance_6);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_6/Scale.txt", scale_6, &n_scale_6);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_6/Shift.txt", shift_6, &n_shift_6);
    batch_norm(out_relu_6, out_batch_6, mean_6, variance_6, scale_6, shift_6, 1, 256, 8, 8);
    //xuat(out_batch_3, (16 * 16 * 128));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_6.txt", out_batch_6, (8 * 8 * 256));

    //////===========================Maxpooling2d_2===============================
    float out_maxpool_2[4 * 4 * 256];
    int l_pool_2 = 4 * 4 * 256;
    maxpool2d(out_batch_6, 8, 8, 2, 256, 2, out_maxpool_2);
    //xuat(out_maxpool_2, 4 * 4 * 256);
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_maxpool_2.txt", out_maxpool_2, (4 * 4 * 256));

    //////===========================Conv2d_7=====================================
    float* w_7 = (float*)malloc((3 * 3 * 256 * 512) * sizeof(float));
    float* out_7 = (float*)malloc((4 * 4 * 512) * sizeof(float));
    float* out_relu_7 = (float*)malloc((4 * 4 * 512) * sizeof(float));
    float bias_7[512];
    int nw_7 = (3 * 3 * 256 * 512);
    int nb_7 = 512;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_7.txt", w_7, &nw_7);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_7.txt", bias_7, &nb_7);
    conv2d(out_maxpool_2, w_7, 4, 4, 3, 3, 256, 512, out_7, bias_7, 1);
    //xuat(out_7, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_7.txt", out_7, (4 * 4 * 512));
    relu(out_7, (4 * 4 * 512), out_relu_7);
    //xuat(out_relu_7, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_7.txt", out_relu_7, (4 * 4 * 512));

    ////===========================Batch_Normalization_7========================
    float* out_batch_7 = (float*)malloc((4 * 4 * 512) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_7 = (4 * 4 * 512);
    float mean_7[512];
    float variance_7[512];
    float scale_7[512];
    float shift_7[512];
    int n_mean_7 = 512;
    int n_variance_7 = 512;
    int n_scale_7 = 512;
    int n_shift_7 = 512;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_7/Mean.txt", mean_7, &n_mean_7);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_7/Variance.txt", variance_7, &n_variance_7);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_7/Scale.txt", scale_7, &n_scale_7);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_7/Shift.txt", shift_7, &n_shift_7);
    batch_norm(out_relu_7, out_batch_7, mean_7, variance_7, scale_7, shift_7, 1, 512, 4, 4);
    //xuat(out_batch_7, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_7.txt", out_batch_7, (4 * 4 * 512));


    ////===========================Conv2d_8=====================================
    float* w_8 = (float*)malloc((3 * 3 * 512 * 512) * sizeof(float));
    float* out_8 = (float*)malloc((4 * 4 * 512) * sizeof(float));
    float* out_relu_8 = (float*)malloc((4 * 4 * 512) * sizeof(float));
    float bias_8[512];
    int nw_8 = (3 * 3 * 512 * 512);
    int nb_8 = 512;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_8.txt", w_8, &nw_8);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_8.txt", bias_8, &nb_8);
    conv2d(out_batch_7, w_8, 4, 4, 3, 3, 512, 512, out_8, bias_8, 1);
    //xuat(out_8, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_8.txt", out_8, (4 * 4 * 512));
    relu(out_8, (4 * 4 * 512), out_relu_8);
    //xuat(out_relu_8, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_8.txt", out_relu_8, (4 * 4 * 512));

    ////===========================Batch_Normalization_8========================
    float* out_batch_8 = (float*)malloc((4 * 4 * 512) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_8 = (4 * 4 * 512);
    float mean_8[512];
    float variance_8[512];
    float scale_8[512];
    float shift_8[512];
    int n_mean_8 = 512;
    int n_variance_8 = 512;
    int n_scale_8 = 512;
    int n_shift_8 = 512;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_8/Mean.txt", mean_8, &n_mean_8);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_8/Variance.txt", variance_8, &n_variance_8);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_8/Scale.txt", scale_8, &n_scale_8);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_8/Shift.txt", shift_8, &n_shift_8);
    batch_norm(out_relu_8, out_batch_8, mean_8, variance_8, scale_8, shift_8, 1, 512, 4, 4);
    //xuat(out_batch_8, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_8.txt", out_batch_8, (4 * 4 * 512));
    //////===========================Conv2d_9=====================================
    float* w_9 = (float*)malloc((3 * 3 * 512 * 512) * sizeof(float));
    float* out_9 = (float*)malloc((4 * 4 * 512) * sizeof(float));
    float* out_relu_9 = (float*)malloc((4 * 4 * 512) * sizeof(float));
    float bias_9[512];
    int nw_9 = (3 * 3 * 512 * 512);
    int nb_9 = 512;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_9.txt", w_9, &nw_9);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_9.txt", bias_9, &nb_9);
    conv2d(out_batch_8, w_9, 4, 4, 3, 3, 512, 512, out_9, bias_9, 1);
    //xuat(out_8, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_9.txt", out_9, (4 * 4 * 512));
    relu(out_9, (4 * 4 * 512), out_relu_9);
    //xuat(out_relu_8, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_9.txt", out_relu_9, (4 * 4 * 512));

    ////===========================Batch_Normalization_9========================
    float* out_batch_9 = (float*)malloc((4 * 4 * 512) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_9 = (4 * 4 * 512);
    float mean_9[512];
    float variance_9[512];
    float scale_9[512];
    float shift_9[512];
    int n_mean_9 = 512;
    int n_variance_9 = 512;
    int n_scale_9 = 512;
    int n_shift_9 = 512;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_9/Mean.txt", mean_9, &n_mean_9);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_9/Variance.txt", variance_9, &n_variance_9);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_9/Scale.txt", scale_9, &n_scale_9);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_9/Shift.txt", shift_9, &n_shift_9);
    batch_norm(out_relu_9, out_batch_9, mean_9, variance_9, scale_9, shift_9, 1, 512, 4, 4);
    //xuat(out_batch_9, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_9.txt", out_batch_9, (4 * 4 * 512));
    //////===========================Maxpooling2d_3===============================
    float out_maxpool_3[2 * 2 * 512];
    int l_pool_3 = 2 * 2 * 512;
    maxpool2d(out_batch_9, 4, 4, 2, 512, 2, out_maxpool_3);
    //xuat(out_maxpool_3, 2 * 2 * 512);
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_maxpool_3.txt", out_maxpool_3, (2 * 2 * 512));

    //////===========================Conv2d_10=====================================
    float* w_10 = (float*)malloc((3 * 3 * 512 * 512) * sizeof(float));
    float* out_10 = (float*)malloc((2 * 2 * 512) * sizeof(float));
    float* out_relu_10 = (float*)malloc((2 * 2 * 512) * sizeof(float));
    float bias_10[512];
    int nw_10 = (3 * 3 * 512 * 512);
    int nb_10 = 512;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_10.txt", w_10, &nw_10);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_10.txt", bias_10, &nb_10);
    conv2d(out_maxpool_3, w_10, 2, 2, 3, 3, 512, 512, out_10, bias_10, 1);
    //xuat(out_8, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_10.txt", out_10, (2 * 2 * 512));
    relu(out_10, (2 * 2 * 512), out_relu_10);
    //xuat(out_relu_10, (2 * 2 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_10.txt", out_relu_10, (2 * 2 * 512));

    ////===========================Batch_Normalization_10========================
    float* out_batch_10 = (float*)malloc((2 * 2 * 512) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_10 = (2 * 2 * 512);
    float mean_10[512];
    float variance_10[512];
    float scale_10[512];
    float shift_10[512];
    int n_mean_10 = 512;
    int n_variance_10 = 512;
    int n_scale_10 = 512;
    int n_shift_10 = 512;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_10/Mean.txt", mean_10, &n_mean_10);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_10/Variance.txt", variance_10, &n_variance_10);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_10/Scale.txt", scale_10, &n_scale_10);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_10/Shift.txt", shift_10, &n_shift_10);
    batch_norm(out_relu_10, out_batch_10, mean_10, variance_10, scale_10, shift_10, 1, 512, 2, 2);
    //xuat(out_batch_10, (2 * 2 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_10.txt", out_batch_10, (2 * 2 * 512));

    //////===========================Conv2d_11=====================================
    float* w_11 = (float*)malloc((3 * 3 * 512 * 512) * sizeof(float));
    float* out_11 = (float*)malloc((2 * 2 * 512) * sizeof(float));
    float* out_relu_11 = (float*)malloc((2 * 2 * 512) * sizeof(float));
    float bias_11[512];
    int nw_11 = (3 * 3 * 512 * 512);
    int nb_11 = 512;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_11.txt", w_11, &nw_11);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_11.txt", bias_11, &nb_11);
    conv2d(out_batch_10, w_11, 2, 2, 3, 3, 512, 512, out_11, bias_11, 1);
    //xuat(out_8, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_11.txt", out_11, (2 * 2 * 512));
    relu(out_11, (2 * 2 * 512), out_relu_11);
    //xuat(out_relu_11, (2 * 2 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_11.txt", out_relu_11, (2 * 2 * 512));

    ////===========================Batch_Normalization_11========================
    float* out_batch_11 = (float*)malloc((2 * 2 * 512) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_11 = (2 * 2 * 512);
    float mean_11[512];
    float variance_11[512];
    float scale_11[512];
    float shift_11[512];
    int n_mean_11 = 512;
    int n_variance_11 = 512;
    int n_scale_11 = 512;
    int n_shift_11 = 512;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_11/Mean.txt", mean_11, &n_mean_11);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_11/Variance.txt", variance_11, &n_variance_11);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_11/Scale.txt", scale_11, &n_scale_11);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_11/Shift.txt", shift_11, &n_shift_11);
    batch_norm(out_relu_11, out_batch_11, mean_11, variance_11, scale_11, shift_11, 1, 512, 2, 2);
    //xuat(out_batch_11, (2 * 2 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_11.txt", out_batch_11, (2 * 2 * 512));

    ////===========================Conv2d_12=====================================
    float* w_12 = (float*)malloc((3 * 3 * 512 * 512) * sizeof(float));
    float* out_12 = (float*)malloc((2 * 2 * 512) * sizeof(float));
    float* out_relu_12 = (float*)malloc((2 * 2 * 512) * sizeof(float));
    float bias_12[512];
    int nw_12 = (3 * 3 * 512 * 512);
    int nb_12 = 512;
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/W_12.txt", w_12, &nw_12);
    //xuat(w_0, (3 * 3 * 3 * 64));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_12.txt", bias_12, &nb_12);
    conv2d(out_batch_11, w_12, 2, 2, 3, 3, 512, 512, out_12, bias_12, 1);
    //xuat(out_8, (4 * 4 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_12.txt", out_12, (2 * 2 * 512));
    relu(out_12, (2 * 2 * 512), out_relu_12);
    //xuat(out_relu_12, (2 * 2 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_relu_12.txt", out_relu_12, (2 * 2 * 512));

    ////===========================Batch_Normalization_12========================
    float* out_batch_12 = (float*)malloc((2 * 2 * 512) * sizeof(float));

    //float* input_batch_0 = (float*)malloc((32 * 32 * 32) * sizeof(float));
    int n_batch_12 = (2 * 2 * 512);
    float mean_12[512];
    float variance_12[512];
    float scale_12[512];
    float shift_12[512];
    int n_mean_12 = 512;
    int n_variance_12 = 512;
    int n_scale_12 = 512;
    int n_shift_12 = 512;
    //readfile ("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_relu_0.txt", input_batch_0, n_batch_0);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_12/Mean.txt", mean_12, &n_mean_12);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_12/Variance.txt", variance_12, &n_variance_12);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_12/Scale.txt", scale_12, &n_scale_12);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_12/Shift.txt", shift_12, &n_shift_12);
    batch_norm(out_relu_12, out_batch_12, mean_12, variance_12, scale_12, shift_12, 1, 512, 2, 2);
    //xuat(out_batch_12, (2 * 2 * 512));
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_12.txt", out_batch_12, (2 * 2 * 512));
    ////===========================Maxpooling2d_4===============================
    float out_maxpool_4[1 * 1 * 512];
    int l_pool_4 = 1 * 1 * 512;
    maxpool2d(out_batch_12, 2, 2, 2, 512, 2, out_maxpool_4);
    //xuat(out_maxpool_4, 1 * 1 * 512);
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_maxpool_4.txt", out_maxpool_4, (1 * 1 * 512));

    ////======================================****LAST_COVER****===========================================//
    ////===============================Transpose============================
    float out_transpose[1 * 1 * 512];
    transposedense(out_maxpool_4, 1, 1, 512, out_transpose);
    // xuat(out_transpose, 512);
    //============================Dense512=============================
    float* w_512 = (float*)malloc((512*512)* sizeof(float));
    float bias_512[512];
    int nw_512 = 512*512;
    int nb_512 = 512;
    int l_dense_512 = 1 * 1 * 512;
    int l_w_512 = 512;
    float out_dense_512[512];
    float out_relu_512[512];
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/dense_512.txt", w_512, &nw_512);
    //xuat(w_128, nw_128);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_dense_512.txt", bias_512, &nb_512);
    dense(out_transpose, l_dense_512, l_w_512, w_512, bias_512, out_dense_512);
    //xuat(out_dense_128, 128);
    relu(out_dense_512, 512, out_relu_512);
    //xuat(out_relu_512, 512);
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_dense_512.txt", out_relu_512, 512);

    //////=============================Batch_Normalization_13================

    float out_batch_13[512];
    int n_batch_13 = 512;
    float mean_13[512];
    float variance_13[512];
    float scale_13[512];
    float shift_13[512];
    int n_mean_13 = 512;
    int n_variance_13 = 512;
    int n_scale_13 = 512;
    int n_shift_13 = 512;
    //readfile("D:/ASSET/NAM4HK2/C_Code/OUT_CIFAR10/out_dense_128.txt", out_re_128, n_batch_6);
    //xuat(input_batch_0, (32 * 32 * 32));
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_13/Mean.txt", mean_13, &n_mean_13);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_13/Variance.txt", variance_13, &n_variance_13);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_13/Scale.txt", scale_13, &n_scale_13);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Batch_Normalization_13/Shift.txt", shift_13, &n_shift_13);
    batch_norm(out_relu_512, out_batch_13, mean_13, variance_13, scale_13, shift_13, 1, 512, 1, 1);
    // xuat(out_relu_512, 512);
    //writefile1d("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/Output_each_layers/Out_batch_13.txt", out_batch_13, (512));

    ////============================Dense_10================================
    float w_dense_10[5120];
    float bias_dense_10[10];
    int nw_dense_10 = 5120;
    int nb_dense_10 = 10;
    int l_dense_10 = 512;
    int l_w_10 = 10;
    float out_dense_10[10];
    float out_softmax_10[10];
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/dense_10.txt", w_dense_10, &nw_dense_10);
    readfile("D:/ASSET/NAM4HK2/C_Code/Cifar10_Vgg16/Cifar10_vgg16/Cifar10_vgg16/bias_dense_10.txt", bias_dense_10, &nb_dense_10);
    dense(out_batch_13, l_dense_10, l_w_10, w_dense_10, bias_dense_10, out_dense_10);
    softmax(out_dense_10, 10, out_softmax_10);
    xuat(out_softmax_10, 10);
    argmax(out_softmax_10, 10);

    //===========================Free_Input==================================
    free(image);
    free(output);
    free(image_transpose);
   
    //===========================Free_Conv2d_0===============================
    //free(w_0);
    free(out_0);
    free(out_relu_0);

    //===========================Free_BatchNormalization_0===================
    free(out_batch_0);

    //===========================Free_Conv2d_1===============================
    free(w_1);
    free(out_1);
    free(out_relu_1);
    //===========================Free_BatchNormalization_1===================
    free(out_batch_1);

    //===========================Free_Conv2d_2===============================
    free(w_2);
    free(out_2);
    free(out_relu_2);
    //===========================Free_BatchNormalization_2===================
    free(out_batch_2);
    //===========================Free_Conv2d_3===============================
    free(w_3);
    free(out_3);
    free(out_relu_3);
    //===========================Free_BatchNormalization_3===================
    free(out_batch_3);
    //===========================Free_Conv2d_4===============================
    free(w_4);
    free(out_4);
    free(out_relu_4);
    //===========================Free_BatchNormalization_4===================
    free(out_batch_4);
    //===========================Free_Conv2d_5===============================
    free(w_5);
    free(out_5);
    free(out_relu_5);
    //===========================Free_BatchNormalization_5===================
    free(out_batch_5);
    //===========================Free_Conv2d_6===============================
    free(w_6);
    free(out_6);
    free(out_relu_6);
    //===========================Free_BatchNormalization_6===================
    free(out_batch_6);
    //===========================Free_Conv2d_7===============================
    free(w_7);
    free(out_7);
    free(out_relu_7);
    //===========================Free_BatchNormalization_7===================
    free(out_batch_7);
    //===========================Free_Conv2d_8===============================
    free(w_8);
    free(out_8);
    free(out_relu_8);
    //===========================Free_BatchNormalization_8===================
    free(out_batch_8);
    //===========================Free_Conv2d_9===============================
    free(w_9);
    free(out_9);
    free(out_relu_9);
    //===========================Free_BatchNormalization_9===================
    free(out_batch_9);
    //===========================Free_Conv2d_10==============================
    free(w_10);
    free(out_10);
    free(out_relu_10);
    //===========================Free_BatchNormalization_10===================
    free(out_batch_10);

    //===========================Free_Conv2d_11===============================
    free(w_11);
    free(out_11);
    free(out_relu_11);
    //===========================Free_BatchNormalization_11===================
    free(out_batch_11);
    //===========================Free_Conv2d_12===============================
    free(w_12);
    free(out_12);
    free(out_relu_12);
    //===========================Free_BatchNormalization_12===================
    free(out_batch_12);
    //===========================Free_Dense512===================
    free(w_512);
     //Get the end time.
    time_t end_time = time(NULL);
    // Calculate the execution time.
    double execution_time = end_time - start_time;

    // Print the execution time.
    printf("Execution time: %.2f seconds\n", execution_time);
}




//int Sliding_Window(float* input, int col, int row, int fcol, int frow, float output[5][34][3][3]) {
//    // int out_col = col - frow + 1;
//    // int out_row = row - frow + 1;
//    for (int i = 0; i < row; i++) {
//        for (int j = 0; j < col; j++) {
//            for (int m = 0; m < frow; m++) {
//                for (int n = 0; n < fcol; n++) {
//                    int input_idx = (i + m) * col + j + n;
//                    output[i][j][m][n] = input[input_idx];
//                }
//            }
//        }
//    }
//    for (int i = 0; i < row; i++) {
//        for (int j = 0; j < col; j++) {
//            for (int m = 0; m < frow; m++) {
//                for (int n = 0; n < frow; n++) {
//                    printf("Matrix[%d][%d][%d][%d] = %f ", i, j, m, n, output[i][j][m][n]);
//                }
//            }
//        }
//    }
//    return 0;
//}


//void floatToHex(float* array, int size, FILE* file) {
//    for (int i = 0; i < size; i++) {
//        unsigned int* hex = (unsigned int*)&array[i];
//        fprintf(file, "%08X\n", array[i], *hex);
//    }
//}
//
//
//int main()
//{
//    //float output[5][34][3][3];
//
//    float Input[5 * 34] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//    0, 1.92354, 1.85504, 1.85504, 1.83791, 1.82079, 1.83791, 1.83791, 1.85504, 1.88929, 1.92354, 1.92354, 1.92354, 1.94066, 1.95779, 1.99204, 1.99204, 1.99204, 1.99204, 2.00916, 2.00916, 2.00916, 2.00916, 2.00916, 2.00916, 2.00916, 2.00916, 2.00916, 1.97491, 1.97491, 1.99204, 1.99204, 2.02629, 0,
//    0, 1.95779, 1.87216, 1.87216, 1.85504, 1.82079, 1.85504, 1.87216, 1.88929, 1.92354, 1.94066, 1.94066, 1.94066, 1.94066, 1.95779, 1.97491, 1.97491, 1.97491, 1.97491, 1.99204, 1.99204, 1.99204, 1.99204, 1.99204, 1.99204, 1.99204, 1.99204, 1.99204, 1.95779, 1.97491, 1.99204, 2.00916, 2.04341, 0,
//    0, 1.99204, 1.92354, 1.90641, 1.88929, 1.88929, 1.90641, 1.92354, 1.92354, 1.90641, 1.92354, 1.94066, 1.94066, 1.97491, 1.97491, 1.97491, 1.97491, 1.97491, 1.97491, 1.97491, 1.97491, 1.97491, 1.99204, 1.97491, 1.97491, 1.99204, 1.99204, 1.99204, 1.97491, 1.97491, 2.00916, 2.04341, 2.07766, 0,
//    0, 1.99204, 1.92354, 1.94066, 1.94066, 1.94066, 1.92354, 1.92354, 1.90641, 1.88929, 1.90641, 1.92354, 1.92354, 1.94066, 1.94066, 1.94066, 1.94066, 1.94066, 1.94066, 1.94066, 1.94066, 1.95779, 1.97491, 1.95779, 1.95779, 1.97491, 1.97491, 1.97491, 1.97491, 1.97491, 2.00916, 2.04341, 2.07766, 0 };
//    float W_0[3 * 3] = { -3.16983014e-02,  2.28820518e-02,  3.38687301e-02,
//                    4.47681360e-02,  9.72806364e-02,  2.34394465e-02,
//                    -2.30437238e-03, -5.98642230e-02, -8.50279108e-02 };
//    //float bias_0 = 0.81120044;
//
//    //Sliding_Window(Input, 34, 5, 3, 3, output);
//
//    FILE* fileInput = fopen("input_hex.txt", "w");
//    if (fileInput == NULL) {
//        printf("Failed to open 'input_hex.txt' file.\n");
//        return 1;
//    }
//
//    FILE* fileW0 = fopen("w0_hex.txt", "w");
//    if (fileW0 == NULL) {
//        printf("Failed to open 'w0_hex.txt' file.\n");
//        fclose(fileInput);
//        return 1;
//    }
//
//    printf("Writing floating-point arrays to 'input.txt' and 'w0.txt'...\n");
//
//    fprintf(fileInput, "Input array (float to hex):\n");
//    floatToHex(Input, sizeof(Input) / sizeof(float), fileInput);
//
//    fprintf(fileW0, "W_0 array (float to hex):\n");
//    floatToHex(W_0, sizeof(W_0) / sizeof(float), fileW0);
//
//    fclose(fileInput);
//    fclose(fileW0);
//
//    printf("Conversion completed. Check 'input_hex.txt' and 'w0_hex.txt' for the results.\n");
//
//    return 0;
//}
