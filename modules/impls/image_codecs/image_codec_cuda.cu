#include "image_codec.h"
#include "nvjpeg.h"
#include <fstream>

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

nvjpegInputFormat_t colorSchemeToNvOutputFormat(ImageColorScheme colorScheme)
{
    switch (colorScheme)
    {
    case ImageColorScheme::IMAGE_RGB :
        
        return nvjpegInputFormat_t::NVJPEG_INPUT_RGBI;
    default:
        return nvjpegInputFormat_t::NVJPEG_INPUT_RGBI;
    }
}

void encode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth)
{
    // code taken from example: https://docs.nvidia.com/cuda/nvjpeg/index.html#nvjpeg-encode-examples
    nvjpegInputFormat_t format = colorSchemeToNvOutputFormat(colorScheme);

    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    cudaStream_t stream;

    cuda_log(cudaStreamCreate(&stream));  // Add this before using 'stream'

    // initialize nvjpeg structures
    cuda_log(nvjpegCreateSimple(&nv_handle));
    cuda_log(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
    cuda_log(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));

    // This has to be done, default params are not sufficient
    // source: https://stackoverflow.com/questions/65929613/nvjpeg-encode-packed-bgr
    cuda_log(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream)); 

    nvjpegImage_t nv_image;

    // Fill nv_image with image data, by copying data from matrix to GPU
    // docs about nv_image: https://docs.nvidia.com/cuda/nvjpeg/index.html#nvjpeg-encode-examples
    cuda_log(cudaMalloc((void **)&(nv_image.channel[0]), img_matrix->height * img_matrix->width * 3));
    cuda_log(cudaMemcpy(nv_image.channel[0], img_matrix->array.data(), img_matrix->height * img_matrix->width * 3, cudaMemcpyHostToDevice));
    
    //Pitch represents bytes per row
    nv_image.pitch[0] = img_matrix->width *3;

    // Compress image
    cuda_log(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
        &nv_image, format, img_matrix->width, img_matrix->height, stream));

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
    cuda_log(cudaStreamDestroy(stream));
    cuda_log(cudaFree(nv_image.channel[0]));
}

void decode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth)
{

}

void load_image_file(std::vector<unsigned char>* png_buffer, std::string image_filepath)
{

}

void save_image_file(std::vector<unsigned char>* img_buff, std::string image_filepath)
{
    std::ofstream output_file(image_filepath, std::ios::out | std::ios::binary);
    output_file.write((char *)img_buff->data(), img_buff->size());
    output_file.close();
}
