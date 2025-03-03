#ifdef CUDA_IMPL

/// @brief Fill interlaced matrix with single value.
/// @param arr interlaced matri
/// @param n_arr matrix size
/// @param components the supplied value, split into components. Order of components is going to be preserved in the resulting array
/// @param n_comps number of components
void fillInterlaced(unsigned char *arr, const unsigned int n_arr, unsigned char* components, const unsigned int n_comps);

#endif
