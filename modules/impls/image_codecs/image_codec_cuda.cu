#include "image_codec.h"
#include "nvjpeg.h"
#include <fstream>

cudaStream_t stream;
nvjpegHandle_t nv_handle;
nvjpegEncoderState_t nv_enc_state;
nvjpegEncoderParams_t nv_enc_params;

/// @brief for debug
nvjpegStatus_t last_status = (nvjpegStatus_t)-1;
cudaError_t last_error = (cudaError_t)-1;

void cuda_log(nvjpegStatus_t status)
{
    last_status = status;
}

void cuda_log(cudaError_t status)
{
    last_error = status;
}

image_codec::image_codec()
{
    //THREAD SAFE
    //cuda stream that stores order of operations on GPU
    cuda_log(cudaStreamCreate(&stream));
    //library handle
    cuda_log(nvjpegCreateSimple(&nv_handle));

    //NOT THREAD SAFE
    //nvjpeg encoding
    cuda_log(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
    cuda_log(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));

    // set the highest quality
    cuda_log(nvjpegEncoderParamsSetQuality(nv_enc_params, 100, stream));

    //use the best type of JPEG encoding
    cuda_log(nvjpegEncoderParamsSetEncoding(nv_enc_params, nvjpegJpegEncoding_t::NVJPEG_ENCODING_LOSSLESS_HUFFMAN, stream));
}

void image_codec::encode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth)
{
    // code taken from example: https://docs.nvidia.com/cuda/nvjpeg/index.html#nvjpeg-encode-examples

    nvjpegImage_t nv_image;
    //Pitch represents bytes per row
    size_t pitch_0_size = img_matrix->width;

    if (colorScheme == ImageColorScheme::IMAGE_RGB)
    {
        // This has to be done, default params are not sufficient
        // source: https://stackoverflow.com/questions/65929613/nvjpeg-encode-packed-bgr
        cuda_log(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream));

        pitch_0_size *= 3;
    }
    else
    {
        cuda_log(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_GRAY, stream));
    }

    // Fill nv_image with image data, by copying data from matrix to GPU
    // docs about nv_image: https://docs.nvidia.com/cuda/nvjpeg/index.html#nvjpeg-encode-examples
    cuda_log(cudaMalloc((void **)&(nv_image.channel[0]), pitch_0_size * img_matrix->height));
    cuda_log(cudaMemcpy(nv_image.channel[0], img_matrix->array.data(), pitch_0_size * img_matrix->height, cudaMemcpyHostToDevice));
    
    nv_image.pitch[0] = pitch_0_size;

    // Compress image
    if (colorScheme == ImageColorScheme::IMAGE_RGB)
    {
        cuda_log(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
            &nv_image, nvjpegInputFormat_t::NVJPEG_INPUT_RGBI, img_matrix->width, img_matrix->height, stream));   
    }
    else
    {
        cuda_log(nvjpegEncodeYUV(nv_handle, nv_enc_state, nv_enc_params,
            &nv_image, nvjpegChromaSubsampling_t::NVJPEG_CSS_GRAY, img_matrix->width, img_matrix->height, stream));
    }

    // get compressed stream size
    size_t length = 0;
    cuda_log(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));
    // get stream itself
    cuda_log(cudaStreamSynchronize(stream));
    img_source->clear();
    img_source->resize(length);
    cuda_log(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, img_source->data(), &length, 0));

    cuda_log(cudaStreamSynchronize(stream));

    //clean up
    cuda_log(cudaFree(nv_image.channel[0]));
}
        
void image_codec::decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth)
{
    
}
        
void image_codec::load_image_file(std::vector<unsigned char>* img_buff, std::string image_filepath)
{
    std::ifstream input_file(image_filepath+".jpeg", std::ios::out | std::ios::binary);
    input_file.read((char *)img_buff->data(), img_buff->size());
    input_file.close();
}
        
void image_codec::save_image_file(std::vector<unsigned char>* img_buff, std::string image_filepath)
{
    std::ofstream output_file(image_filepath+".jpeg", std::ios::out | std::ios::binary);
    output_file.write((char *)img_buff->data(), img_buff->size());
    output_file.close();
}

image_codec::~image_codec()
{
    if (nv_enc_params != nullptr)
    {
        cuda_log(nvjpegEncoderParamsDestroy(nv_enc_params));
        nv_enc_params = nullptr;
    }
    
    if (nv_enc_state != nullptr)
    {
        cuda_log(nvjpegEncoderStateDestroy(nv_enc_state));
        nv_enc_state = nullptr;
    }

    if (nv_handle != nullptr)
    {
        cuda_log(nvjpegDestroy(nv_handle));
        nv_handle = nullptr;
    }

    if (stream != nullptr)
    {
        cuda_log(cudaStreamDestroy(stream));
        stream = nullptr;
    }
}
