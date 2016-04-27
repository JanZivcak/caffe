#ifndef CAFFE_PLAYER
  #include <boost/thread.hpp>
#endif
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
#ifndef CAFFE_PLAYER
  forward_mutex_.reset(new boost::mutex());
#endif
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
#ifdef CAFFE_PLAYER
    LOG(FATAL) << "Locking is unsupported in Caffe Player";
#else
    forward_mutex_->lock();
#endif
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
#ifdef CAFFE_PLAYER
    LOG(FATAL) << "Unlocking is unsupported in Caffe Player";
#else
    forward_mutex_->unlock();
#endif
  }
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
