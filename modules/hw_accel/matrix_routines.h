#ifdef CUDA_IMPL

/// @brief Fill interlaced matrix with single value.
/// @param m interlaced
/// @param components the supplied value, split into components. Order of components is going to be preserved in the resulting array
void fillInterlaced(matrix* m, unsigned char* components);

#endif
