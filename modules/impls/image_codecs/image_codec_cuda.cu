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

void encode(std::vector<unsigned char>* img_source, matrix* img_matrix, ImageColorScheme colorScheme, unsigned bit_depth)
{
    // code taken from example: https://docs.nvidia.com/cuda/nvjpeg/index.html#nvjpeg-encode-examples
    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    cudaStream_t stream;

    cuda_log(cudaStreamCreate(&stream));  // Add this before using 'stream'

    // initialize nvjpeg structures
    cuda_log(nvjpegCreateSimple(&nv_handle));
    cuda_log(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
    cuda_log(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));

    // set the highest quality
    cuda_log(nvjpegEncoderParamsSetQuality(nv_enc_params, 100, stream));

    //use the best type of JPEG encoding
    cuda_log(nvjpegEncoderParamsSetEncoding(nv_enc_params, nvjpegJpegEncoding_t::NVJPEG_ENCODING_LOSSLESS_HUFFMAN, stream));

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
