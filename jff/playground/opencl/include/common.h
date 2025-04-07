#ifndef COMMON_H
#define COMMON_H

#define CL_CHECK(err) do \
{ \
    if(err) { \
        std::cout << "CL_CHECK ERROR: code <" << err << ">: "; \
        switch(err) \
        { \
            case -11: \
                std::cout << "BUILD_PROGRAM_FAILURE"; \
                break; \
            case -3: \
                std::cout << "COMPILER_NOT_AVAILABLE"; \
                break; \
            case -2: \
                std::cout << "DEVICE_NOT_AVAILABLE"; \
                break; \
            case -1: \
                std::cout << "DEVICE_NOT_FOUND"; \
                break; \
            case -14: \
                std::cout << "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; \
                break; \
            case -9: \
                std::cout << "IMAGE_FORMAT_MISMATCH"; \
                break; \
            case -10: \
                std::cout << "IMAGE_FORMAT_NOT_SUPPORTED"; \
                break; \
            case -49: \
                std::cout << "INVALID_ARG_INDEX"; \
                break; \
            case -51: \
                std::cout << "INVALID_ARG_SIZE"; \
                break; \
            case -50: \
                std::cout << "INVALID_ARG_VALUE"; \
                break; \
            case -42: \
                std::cout << "INVALID_BINARY"; \
                break; \
            case -61: \
                std::cout << "INVALID_BUFFER_SIZE"; \
                break; \
            case -43: \
                std::cout << "INVALID_BUILD_OPTIONS"; \
                break; \
            case -36: \
                std::cout << "INVALID_COMMAND_QUEUE"; \
                break; \
            case -34: \
                std::cout << "INVALID_CONTEXT"; \
                break; \
            case -33: \
                std::cout << "INVALID_DEVICE"; \
                break; \
            case -31: \
                std::cout << "INVALID_DEVICE_TYPE"; \
                break; \
            case -58: \
                std::cout << "INVALID_EVENT"; \
                break; \
            case -57: \
                std::cout << "INVALID_EVENT_WAIT_LIST"; \
                break; \
            case -56: \
                std::cout << "INVALID_GLOBAL_OFFSET"; \
                break; \
            case -63: \
                std::cout << "INVALID_GLOBAL_WORK_SIZE"; \
                break; \
            case -60: \
                std::cout << "INVALID_GL_OBJECT"; \
                break; \
            case -37: \
                std::cout << "INVALID_HOST_PTR"; \
                break; \
            case -39: \
                std::cout << "INVALID_IMAGE_FORMAT_DESCRIPTOR"; \
                break; \
            case -40: \
                std::cout << "INVALID_IMAGE_SIZE"; \
                break; \
            case -48: \
                std::cout << "INVALID_KERNEL"; \
                break; \
            case -52: \
                std::cout << "INVALID_KERNEL_ARGS"; \
                break; \
            case -47: \
                std::cout << "INVALID_KERNEL_DEFINITION"; \
                break; \
            case -46: \
                std::cout << "INVALID_KERNEL_NAME"; \
                break; \
            case -38: \
                std::cout << "INVALID_MEM_OBJECT"; \
                break; \
            case -62: \
                std::cout << "INVALID_MIP_LEVEL"; \
                break; \
            case -59: \
                std::cout << "INVALID_OPERATION"; \
                break; \
            case -32: \
                std::cout << "INVALID_PLATFORM"; \
                break; \
            case -44: \
                std::cout << "INVALID_PROGRAM"; \
                break; \
            case -45: \
                std::cout << "INVALID_PROGRAM_EXECUTABLE"; \
                break; \
            case -64: \
                std::cout << "INVALID_PROPERTY"; \
                break; \
            case -35: \
                std::cout << "INVALID_QUEUE_PROPERTIES"; \
                break; \
            case -41: \
                std::cout << "INVALID_SAMPLER"; \
                break; \
            case -30: \
                std::cout << "INVALID_VALUE"; \
                break; \
            case -53: \
                std::cout << "INVALID_WORK_DIMENSION"; \
                break; \
            case -54: \
                std::cout << "INVALID_WORK_GROUP_SIZE"; \
                break; \
            case -55: \
                std::cout << "INVALID_WORK_ITEM_SIZE"; \
                break; \
            case -12: \
                std::cout << "MAP_FAILURE"; \
                break; \
            case -8: \
                std::cout << "MEM_COPY_OVERLAP"; \
                break; \
            case -4: \
                std::cout << "MEM_OBJECT_ALLOCATION_FAILURE"; \
                break; \
            case -13: \
                std::cout << "MISALIGNED_SUB_BUFFER_OFFSET"; \
                break; \
            case -6: \
                std::cout << "OUT_OF_HOST_MEMORY"; \
                break; \
            case -5: \
                std::cout << "OUT_OF_RESOURCES"; \
                break; \
            case -7: \
                std::cout << "PROFILING_INFO_NOT_AVAILABLE"; \
                break; \
            case 0: \
                std::cout << "SUCCESS"; \
                break; \
            default: \
                std::cout << "Unknown error code"; \
        } \
        std::cout << std::endl; \
    } \
} while(0)

#endif // COMMON_H