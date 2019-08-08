
#define ERROR(msg) \
    fprintf(stderr, "ERROR: %s\n", msg); \
    fprintf(stderr, "Line %i of function %s in file %s\n", __LINE__, __func__, __FILE__); \
    exit(EXIT_FAILURE);

