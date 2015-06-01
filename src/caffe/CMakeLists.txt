# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
set(Caffe_LINKER_LIBS proto ${Caffe_LINKER_LIBS}) # note, crucial to prepend!
caffe_default_properties(proto)

if( MSVC )
  include(${PROJECT_SOURCE_DIR}/cmake/Modules/FindWindowsSDK.cmake)
  get_windowssdk_library_dirs("${WINDOWSSDK_PREFERRED_DIR}" WINDOWSSDK_LIB_DIR)
  find_library(IMGHLP   ImageHlp ${WINDOWSSDK_LIB_DIR} NO_DEFAULT_PATH)
  if (NOT GCC_LIB_DIR)
    message(FATAL_ERROR "Please specify directory where libgcc.a can be found (-DGCC_LIB_DIR=\"...\")\nIt is needed when linking libopenblas.a with MSVC")
  endif()
  find_library(LIBGCC gcc ${GCC_LIB_DIR} NO_DEFAULT_PATH)
  set(Caffe_LINKER_LIBS ${IMGHLP} ${LIBGCC} ${Caffe_LINKER_LIBS})
endif()

# --[ Caffe library

if(CAFFE_PLAYER)
  caffe_source_group("Include"        GLOB "${PROJECT_SOURCE_DIR}/include/caffe/*.h*")
  caffe_source_group("Include\\Util"  GLOB "${PROJECT_SOURCE_DIR}/include/caffe/util/*.h*")
  caffe_source_group("Include"        GLOB "${PROJECT_BINARY_DIR}/caffe_config.h*")
  caffe_source_group("Source"         GLOB "${PROJECT_SOURCE_DIR}/src/caffe/*.cpp")
  caffe_source_group("Source\\Util"   GLOB "${PROJECT_SOURCE_DIR}/src/caffe/util/*.cpp")
  caffe_source_group("Source\\Layers" GLOB "${PROJECT_SOURCE_DIR}/src/caffe/layers/*.cpp")
  caffe_source_group("Source\\Proto"  GLOB "${PROJECT_SOURCE_DIR}/src/caffe/proto/*.proto")
  
  set(srcs ${srcs} 
    "${PROJECT_SOURCE_DIR}/src/caffe/blob.cpp" 
    "${PROJECT_SOURCE_DIR}/src/caffe/common.cpp" 
    "${PROJECT_SOURCE_DIR}/src/caffe/layer_factory.cpp" 
    "${PROJECT_SOURCE_DIR}/src/caffe/net.cpp" 
    "${PROJECT_SOURCE_DIR}/src/caffe/solver.cpp" 
    "${PROJECT_SOURCE_DIR}/src/caffe/syncedmem.cpp"
    
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/accuracy_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/base_conv_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/bnll_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/concat_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/conv_layer.cpp" 
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/dropout_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/eltwise_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/euclidean_loss_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/flatten_layer.cpp"    
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/hinge_loss_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/im2col_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/infogain_loss_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/inner_product_layer.cpp"    
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/loss_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/lrn_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/multinomial_logistic_loss_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/neuron_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/pooling_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/power_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/relu_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/sigmoid_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/softmax_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/softmax_loss_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/split_layer.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/layers/tanh_layer.cpp"
    
    "${PROJECT_SOURCE_DIR}/src/caffe/proto/caffe.proto"
    
    "${PROJECT_SOURCE_DIR}/src/caffe/util/im2col.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/util/insert_splits.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/util/io.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/util/math_functions.cpp"
    "${PROJECT_SOURCE_DIR}/src/caffe/util/upgrade_proto.cpp"
  )
  
  file(GLOB_RECURSE hdrs ${PROJECT_SOURCE_DIR}/include/caffe/*.h*)
  list(APPEND srcs ${proto_srcs} ${hdrs})

  caffe_convert_absolute_paths(srcs)
  
  add_definitions(-DCAFFE_PLAYER)
  
  add_library(caffe_player ${srcs})
  target_link_libraries(caffe_player proto ${Caffe_LINKER_LIBS})
  
  if(CAFFE_PLAYER_OUTPUT_DIR)
    set_target_properties(caffe_player PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${CAFFE_PLAYER_OUTPUT_DIR}
      LIBRARY_OUTPUT_DIRECTORY ${CAFFE_PLAYER_OUTPUT_DIR}
    )
  else()
    caffe_default_properties(caffe_player)
  endif()
else()
  # creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
  caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})
  
  if(HAVE_CUDA)
    caffe_cuda_compile(cuda_objs ${cuda})
    list(APPEND srcs ${cuda_objs} ${cuda})
  endif()
  
  add_library(caffe ${srcs})
  target_link_libraries(caffe proto ${Caffe_LINKER_LIBS})
  caffe_default_properties(caffe)
endif()



if(NOT CAFFE_PLAYER)
# ---[ Tests
 add_subdirectory(test)
endif()

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION include)
install(FILES ${proto_hdrs} DESTINATION include/caffe/proto)
if(CAFFE_PLAYER)
  install(TARGETS caffe_player proto EXPORT CaffeTargets DESTINATION lib)
else()  
  install(TARGETS caffe proto EXPORT CaffeTargets DESTINATION lib)
endif()

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)


