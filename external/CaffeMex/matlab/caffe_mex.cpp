//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

// @brief solver in API'name means it's for solver use
// otherwise it's for stand-alone net use

#include <string>
#include <vector>
#include <thread>
#include <omp.h>
#include "mex.h"

#include "caffe/caffe.hpp"
#include "caffe/util/path.h"
#include "caffe/util/directory.h"
#include "caffe/util/signal_handler.h"
#undef ERROR

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs
using namespace caffe;  // NOLINT(build/namespaces)


// in new caffe, one solver contains net_ for train and test_net_ for test
static shared_ptr< caffe::P2PSync<float> > SyncSolver_;

// for different stand-alone net instances running in one machine
static vector< shared_ptr<Net<float> > > net_;

// all solvers and nets use same GPU group
// TODO (MultiGPU):different solver or net can have different GPU Group
static vector < int > gpu_group;

// unique hash value for different solver or net
static vector <double> init_key(1, -2);


// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.

//auxiliary functions

// Enum indicates which blob memory to use
inline void mxCHECK(bool expr, const char* msg) {
  if (!expr) {
    mexErrMsgTxt(msg);
  }
}
inline void mxERROR(const char* msg) { mexErrMsgTxt(msg); }


static mxArray* do_forward(const mxArray* const bottom, int model_idx) {
  vector<Blob<float>*>& input_blobs = const_cast<vector<Blob<float>*>&>(net_[model_idx]->input_blobs());
  CHECK_EQ(static_cast<unsigned int>(mxGetNumberOfElements(bottom)),
      input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    if (!mxIsEmpty(elem)){
      CHECK(mxIsSingle(elem))
        << "MatCaffe require single-precision float point data";
      CHECK_EQ(mxGetNumberOfElements(elem), input_blobs[i]->count())
        << "MatCaffe input size does not match the input size of the network";
      const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
      switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(input_blobs[i]->count(), data_ptr,
              input_blobs[i]->mutable_cpu_data());
          break;
        case Caffe::GPU:
          caffe_copy(input_blobs[i]->count(), data_ptr,
              input_blobs[i]->mutable_gpu_data());
          break;
        default:
          LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }
  const vector<Blob<float>*>& output_blobs = net_[model_idx]->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = { output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num() };
    mxArray* mx_blob = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
      case Caffe::CPU:
        caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),
            data_ptr);
        break;
      case Caffe::GPU:
        caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),
            data_ptr);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_backward(const mxArray* const top_diff, int model_idx) {
  vector<Blob<float>*>& output_blobs = const_cast<vector<Blob<float>*>&>(net_[model_idx]->output_blobs());
  vector<Blob<float>*>& input_blobs = const_cast<vector<Blob<float>*>&>(net_[model_idx]->input_blobs());
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(top_diff)[0]),
      output_blobs.size());
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
      reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
      case Caffe::CPU:
        caffe_copy(output_blobs[i]->count(), data_ptr,
            output_blobs[i]->mutable_cpu_diff());
        break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
        output_blobs[i]->mutable_gpu_diff());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net_[model_idx]->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = { input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num() };
    mxArray* mx_blob = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_get_weights(boost::shared_ptr<Net<float> > net) {
  const vector<boost::shared_ptr<Layer<float> > >& layers = net->layers();
  const vector<string>& layer_names = net->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<boost::shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = { num_layers, 1 };
    const char* fnames[2] = { "weights", "layer_names" };
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<boost::shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = { static_cast<mwSize>(layer_blobs.size()), 1 };
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
          mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = { layer_blobs[j]->width(), layer_blobs[j]->height(),
          layer_blobs[j]->channels(), layer_blobs[j]->num() };

        mxArray* mx_weights =
          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
            weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
            weights_ptr);
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
      }
    }
  }

  return mx_layers;
}

static mxArray* do_get_response(boost::shared_ptr<Net<float> > net, string blob_name) {
  const boost::shared_ptr<Blob<float> > blob = net->blob_by_name(blob_name);

  mxArray *mx_blob = NULL;
  if (blob == NULL){
    mx_blob = mxCreateDoubleMatrix(0, 0, mxREAL);
    return mx_blob;
  }

  // copy blob into output
  {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = { blob->width(), blob->height(),
      blob->channels(), blob->num() };

    mx_blob =
      mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    float* response_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));

    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(blob->count(), blob->cpu_data(),
        response_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(blob->count(), blob->gpu_data(),
        response_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }

  return mx_blob;
}


void do_set_input_size(boost::shared_ptr<Net<float> > net, const mxArray *input_dims){

  int size_num = mxGetNumberOfElements(input_dims);
  if (size_num == 0 || size_num % 4 != 0)
  {
    char message[PATH_MAX];
    sprintf(message, "caffe_mex : set_input_size :: invalid prhs[0] with %d elements.\n", size_num);
    mexErrMsgTxt(message);
  }

  if (size_num / 4 != net->input_blobs().size())
  {
    char message[PATH_MAX];
    sprintf(message, "caffe_mex : set_input_size :: invalid prhs[0] with %d elements for %d input_blobs.\n", size_num, int(net->input_blobs().size()));
    mexErrMsgTxt(message);
  }

  if (net->layers().size() <= 0)
    mexErrMsgTxt("caffe_mex : set_input_size :: no layer loaded.\n");

  if (net->input_blobs().size() <= 0)
    mexErrMsgTxt("caffe_mex : set_input_size :: first layer has no input.\n");

  if (!mxIsDouble(input_dims))
    mexErrMsgTxt("caffe_mex : set_input_size :: prhs[0] must be double.\n");

  double *pSize = mxGetPr(input_dims);

  for (int i = 0; i < net->input_blobs().size(); ++i)
  {
    net->input_blobs()[i]->Reshape(pSize[3], pSize[2], pSize[1], pSize[0]);
    pSize += 4;
  }
  net->Reshape();
}


// function set_input_size
// input:  input_size[[width, height, channel, num]s for inputs], model_idx[opt, default = 0]
// output:
void set_input_size(MEX_ARGS){
  if (nrhs != 1 && nrhs != 2)
  {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  /// prhs[0]     [width, height, channel, num]s for inputs
  /// prhs[1](opt)  model_idx 

  int model_idx = 0;
  if (nrhs > 1)
    model_idx = (int)mxGetScalar(prhs[1]);

  do_set_input_size(net_[model_idx], prhs[0]);
}

// function set_input_size_solver
// input:  input_size[cell of ( [width, height, channel, num]s for inputs ) for multi-gpus], is_train[opt, default = true], is_test[opt, default = true]
// output:  
void set_input_size_solver(MEX_ARGS){

  if (nrhs != 1 && nrhs != 2 && nrhs != 3)
  {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  bool is_train = true, is_test = true;

  if (nrhs > 1)
    is_train = (bool)mxGetScalar(prhs[1]);
  if (nrhs > 2)
    is_test = (bool)mxGetScalar(prhs[2]);

#pragma omp parallel for num_threads(int(gpu_group.size()))
  for (int ID = 0; ID < gpu_group.size(); ++ID)
  {
    const mxArray* const input = mxGetCell(prhs[0], 0);

    if (is_train)
    {
      if (ID == 0)
        do_set_input_size(SyncSolver_->solver()->net(), input);
      else
        do_set_input_size(SyncSolver_->workers()[ID]->solver()->net(), input);
    }
    if (is_test)
    {
      if (ID == 0)
        do_set_input_size(SyncSolver_->solver()->test_nets()[0], input);
      else
        do_set_input_size(SyncSolver_->workers()[ID]->solver()->test_nets()[0], input);
    }
  }
}

// function get_weights
// input:  model_idx[opt, default = 0]
// output:  weights
static void get_weights(MEX_ARGS) {
  //  get_weights [model_idx]

  int model_idx = 0;
  if (nrhs > 0)
    model_idx = (int)mxGetScalar(prhs[0]);

  if (model_idx >= net_.size())
    mexErrMsgTxt("caffe_mex : Un-inited net");

  plhs[0] = do_get_weights(net_[model_idx]);
}

// function get_response
// input:  blob_name model_idx[opt, default = 0]
// output: response
static void get_response(MEX_ARGS) {
  //  get_response blob_name [model_idx]

  if (nrhs != 1 && nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  char* blob_name = mxArrayToString(prhs[0]);
  int model_idx = 0;
  if (nrhs > 1)
    model_idx = (int)mxGetScalar(prhs[1]);

  if (model_idx >= net_.size())
    mexErrMsgTxt("caffe_mex : Un-inited net");

  plhs[0] = do_get_response(net_[model_idx], string(blob_name));
}

// function set_mode_cpu
// input:  
// output: 
static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

// function set_mode_gpu
// input:  
// output: 
static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

// function set_device
// input: device_id
// output:
static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

// function set_random_seed
// input: random_seed
// output:
static void set_random_seed(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  int random_seed = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::set_random_seed(random_seed);
}


// function get_init_key
// input: model_idx[opt, default = 0]
// output: init_key
static void get_init_key(MEX_ARGS) {

  int model_idx = 0;
  if (nrhs > 0)
    model_idx = (int)mxGetScalar(prhs[0]);

  if (model_idx >= net_.size())
    mexErrMsgTxt("caffe_mex : Un-inited net");

  plhs[0] = mxCreateDoubleScalar(init_key[model_idx]);
}

static void glog_failure_handler(){
  static bool is_mex_failure = false;
  if (!is_mex_failure)
  {
    is_mex_failure = true;
    ::google::FlushLogFiles(0);
    mexErrMsgTxt("glog check error, please check log and clear mex");
  }
}

static void protobuf_log_handler(::google::protobuf::LogLevel level, const char* filename, int line,
  const std::string& message)
{
  const int max_err_length = 512;
  char err_message[max_err_length];
  sprintf(err_message, "Protobuf : %s . at %s Line %d", message.c_str(), filename, line);
  LOG(INFO) << err_message;
  ::google::FlushLogFiles(0);
  mexErrMsgTxt(err_message);
}

static bool is_log_inited = false;

// Usage: caffe_('init_log', log_base_filename)
static void init_log(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsChar(prhs[0]),
    "Usage: caffe_('init_log', log_dir)");
  if (is_log_inited)
    ::google::ShutdownGoogleLogging();
  char* log_base_filename = mxArrayToString(prhs[0]);
  ::google::SetLogDestination(0, log_base_filename);
  mxFree(log_base_filename);
  ::google::protobuf::SetLogHandler(&protobuf_log_handler);
  ::google::InitGoogleLogging("caffe_mex");
  ::google::InstallFailureFunction(&glog_failure_handler);

  is_log_inited = true;
}


void initGlog() {
  if (is_log_inited) return;
#ifdef _WIN32
  string log_dir = ".\\log\\";
  CDirectory::CreateDirectory(CPath::GetDirectoryName(string(log_dir)).c_str());
#else
  string log_dir = "./log/";
  mkdir(log_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH);
#endif

  std::string now_time = boost::posix_time::to_iso_extended_string(boost::posix_time::second_clock::local_time());
  now_time[13] = '-';
  now_time[16] = '-';
  string log_file = log_dir + "INFO" + now_time + ".txt";
  const char* log_base_filename = log_file.c_str();
  ::google::SetLogDestination(0, log_base_filename);
  ::google::protobuf::SetLogHandler(&protobuf_log_handler);
  ::google::InitGoogleLogging("caffe_mex");
  ::google::InstallFailureFunction(&glog_failure_handler);

  is_log_inited = true;
}

// function init
// input: param_file model_file[opt] model_idx[opt, default = 0]
// output: init_key
static void init(MEX_ARGS) {
  // init param_file [model_file] [model_idx]

  if (nrhs != 1 && nrhs != 2 && nrhs != 3) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }


  int model_idx = 0;
  if (nrhs > 2)
    model_idx = (int)mxGetScalar(prhs[2]);

  if (net_.size() <= model_idx)
  {
    net_.resize(model_idx + 1);
  }

  char* param_file = mxArrayToString(prhs[0]);
  net_[model_idx].reset(new Net<float>(string(param_file),TRAIN));
  mxFree(param_file);

  if (nrhs > 1)
  {
    char* model_file = mxArrayToString(prhs[1]);
    if (!string(model_file).empty())
      net_[model_idx]->CopyTrainedLayersFrom(string(model_file));
    mxFree(model_file);
  }

  init_key[model_idx] = (int)caffe_rng_rand();  // NOLINT(caffe/random_fn)
  if (nlhs == 1) {
	  plhs[0] = mxCreateDoubleScalar(init_key[model_idx]);
  }

}

// function release
// input:  model_idx[opt, default = 0]
// output: 
static void release(MEX_ARGS) {
  int model_idx = 0;
  if (nrhs > 0)
    model_idx = (int)mxGetScalar(prhs[0]);

  if (model_idx >= net_.size()){
    mexPrintf("caffe_mex : Un-inited net \n");
		return ;
	}

  if (net_[model_idx]) {
    net_[model_idx].reset();
	init_key[model_idx] = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }

    for(int i=0;i < net_.size();++i)
    {
      if(net_[i].get())
        return;
    }
    if (is_log_inited && !SyncSolver_.get())
    {
      is_log_inited = false;
      ::google::ShutdownGoogleLogging();
    }
}

// function set_phase
// input:  phase, model_idx[opt, default = 0]
// output: 
static void set_phase(MEX_ARGS) {
  int model_idx = 0;

	if ( nrhs < 1 ){
  	LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
	}	

	char * tmp = mxArrayToString(prhs[0]);
	Phase phase;
	if( strcmp(tmp ,"train") == 0){
		phase = TRAIN;	
	}else if( strcmp(tmp ,"test") == 0){
		phase = TEST;
	}else{
	  mexPrintf("caffe_mex : Unknown phase \n");
		mxFree(tmp);
		return ;
	}

	if( nrhs > 1)
		model_idx = (int) mxGetScalar(prhs[1]);

	if(model_idx >= net_.size())
		 mexErrMsgTxt("caffe_mex : error model_idx");

	vector<boost::shared_ptr<Layer<float> > >& layers = net_[model_idx]->getlayers();
	for(int i = 0 ; i < layers.size(); ++i){
		layers[i] -> SetPhase(phase);
	}

	mxFree(tmp);
}

// function forward
// input:  input_blob, model_idx[opt, default = 0]
// output: 
static void forward(MEX_ARGS) {
  if (nrhs != 1 && nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  int model_idx = 0;
  if (nrhs > 1)
    model_idx = (int)mxGetScalar(prhs[1]);

  if (model_idx >= net_.size())
    mexErrMsgTxt("caffe_mex : Un-inited net");

  plhs[0] = do_forward(prhs[0], model_idx);
}

// function backward
// input:  input_blob, model_idx[opt, default = 0]
// output: 
static void backward(MEX_ARGS) {
  if (nrhs != 1 && nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  int model_idx = 0;
  if (nrhs > 1)
    model_idx = (int)mxGetScalar(prhs[1]);

  if (model_idx >= net_.size())
    mexErrMsgTxt("caffe_mex : Un-inited net");

  plhs[0] = do_backward(prhs[0], model_idx);
}

// function is_initialized
// input:  model_idx[opt, default = 0]
// output: is_initialized
static void is_initialized(MEX_ARGS) {
  int model_idx = 0;
  if (nrhs > 0)
    model_idx = (int)mxGetScalar(prhs[0]);


  if (net_.size() <= model_idx || !net_[model_idx].get()) {
    plhs[0] = mxCreateDoubleScalar(0);
  }
  else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

// function read_mean
// input:  mean_file
// output: blob
static void read_mean(MEX_ARGS) {
  if (nrhs != 1) {
    mexErrMsgTxt("caffe_mex : Usage: caffe('read_mean', 'path_to_binary_mean_file'");
    return;
  }
  const string& mean_file = mxArrayToString(prhs[0]);
  Blob<float> data_mean;
  LOG(INFO) << "Loading mean file from" << mean_file;
  BlobProto blob_proto;
  bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
  if (!result) {
    mexErrMsgTxt("caffe_mex : Couldn't read the file");
    return;
  }
  data_mean.FromProto(blob_proto);
  mwSize dims[4] = { data_mean.width(), data_mean.height(),
    data_mean.channels(), data_mean.num() };
  mxArray* mx_blob = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
  caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
  mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
    " format and channels are also BGR!");
  plhs[0] = mx_blob;
}

// function init_solver
// input:  solver_file model_file log_file[opt]
// output: 
static void init_solver(MEX_ARGS) {
//ignore log file
  if (nrhs < 1 || nrhs > 3) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (gpu_group.empty())
    mexErrMsgTxt("please call set_device_solver first");
  
  char* solver_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);

  LOG(INFO) << "Loading from " << solver_file;
  // read protobuf
  SolverParameter solver_param;
  if (!ReadProtoFromTextFile(solver_file, &solver_param))
    mexErrMsgTxt("caffe_mex: ReadProtoFromTextFile Error!");

  //set net param and set multi GPU signal
  solver_param.set_device_id(gpu_group[0]);
  Caffe::SetDevice(gpu_group[0]);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_solver_count(gpu_group.size());

  caffe::SignalHandler signal_handler(caffe::SolverAction::NONE, caffe::SolverAction::NONE);

  //Caffe::set_root_solver(true); // init root solver's test net
  shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
  //Caffe::set_root_solver(false);
  solver->SetActionFunction(signal_handler.GetActionFunction());

  // copy layer from file 
  if (model_file != NULL && !string(model_file).empty()){
    LOG(INFO) << "Recovery from " << model_file;
    solver->net()->CopyTrainedLayersFrom(model_file);
  }
  SyncSolver_.reset(new caffe::P2PSync<float>(solver, nullptr, solver->param()));
  SyncSolver_->pre_run(gpu_group); // init root solver and worker solver

  //device bind
#pragma omp parallel num_threads(int(gpu_group.size())) 
  {
  int ID = omp_get_thread_num();
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpu_group.size());
  if(ID == 0)
    gpu_group[ID] = SyncSolver_->solver()->param().device_id();
  else
    gpu_group[ID] = SyncSolver_->workers()[ID]->solver()->param().device_id();
    Caffe::SetDevice(gpu_group[ID]);
  //int device;
  //CUDA_CHECK(cudaGetDevice(&device));
  //if(ID == 0)
  //  printf("cudaGetDev = %d solverDev = %d\n",device , SyncSolver_->solver()->param().device_id());
  //else
  //  printf("cudaGetDev = %d solverDev = %d\n",device,  SyncSolver_->workers()[ID]->solver()->param().device_id());
  }

  LOG(INFO) << "Starting Optimization";
  mxFree(model_file);
  mxFree(solver_file);
}

// function recovery_solver
// input:  solver_file model_file log_file[opt]
// output: 
static void recovery_solver(MEX_ARGS) {
//ignore log
  if (nrhs < 1 || nrhs > 3) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (gpu_group.empty())
    mexErrMsgTxt("please call set_device_solver first");

  
  char* solver_file = mxArrayToString(prhs[0]);
  LOG(INFO) << "Loading from " << solver_file;
  SolverParameter solver_param;
  ReadProtoFromTextFile(solver_file, &solver_param);

  //set net param and set multi GPU signal
  solver_param.set_device_id(gpu_group[0]);
  Caffe::SetDevice(gpu_group[0]);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_solver_count(gpu_group.size());

  caffe::SignalHandler signal_handler(caffe::SolverAction::NONE, caffe::SolverAction::NONE);

  shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
  solver->SetActionFunction(signal_handler.GetActionFunction());

  char* model_file = mxArrayToString(prhs[1]);
  LOG(INFO) << "Starting Optimization";
  LOG(INFO) << "Restoring previous solver status from " << model_file;
  printf("Resuming form %s\n", model_file);
  solver->Restore(model_file);

  SyncSolver_.reset(new P2PSync<float>(solver, nullptr, solver->param()));
  SyncSolver_->pre_run(gpu_group);

  //device bind
#pragma omp parallel for num_threads(int(gpu_group.size()))
  for (int ID = 0; ID < gpu_group.size(); ++ID)
  {
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpu_group.size());
    Caffe::SetDevice(gpu_group[ID]);
  }

  mxFree(model_file);
  mxFree(solver_file);
}

// function release_solver
// input:  
// output: 
static void release_solver(MEX_ARGS) {
  SyncSolver_.reset();
  gpu_group.clear();
  init_key[0] = -2;
  if (is_log_inited)
  {
    is_log_inited = false;
	  ::google::ShutdownGoogleLogging();
  }
}

// function do_train
// input:  bottom[cell of (cell of blobs) for multi-gpus]
// output: top[struct of output]
static mxArray* do_train(const mxArray* const bottom) {

  if (!SyncSolver_.get())
  {
    mexPrintf("caffe_mex : No solver inited!\n");
    return mxCreateDoubleMatrix(0, 0, mxREAL);
  }

  if (mxGetNumberOfElements(bottom) != gpu_group.size())
    mexErrMsgTxt("caffe_mex : do_train:input size should be equal to selected gpu number.\n");

	vector<boost::shared_ptr<Layer<float> > >& layers = SyncSolver_->solver()->net()->getlayers();
	for(int i = 0 ; i < layers.size(); ++i){
		layers[i]->SetPhase(TRAIN);
	}

	for(int i = 1 ; i < int(gpu_group.size()); ++i){
		vector<boost::shared_ptr<Layer<float> > >& layers = SyncSolver_->workers()[i]->solver()->net()->getlayers();
		for(int j = 0; j < layers.size(); ++j){
			layers[j]->SetPhase(TRAIN);
		}
	}

  mxArray* mx_out = NULL;
//#pragma omp parallel num_threads(int(gpu_group.size()))
//  for (int ID = 0; ID < gpu_group.size(); ++ID)
//  {
#pragma omp parallel num_threads(int(gpu_group.size())) 
  {
  int ID = omp_get_thread_num();
    const mxArray* const input = mxGetCell(bottom, ID);

    // train phase, use net()
    // copy data to blob, when run solver->step, data will be copyed to GPU
    vector<Blob<float>*>& input_blobs =
      ID == 0 ? 
      const_cast<vector<Blob<float>*>&>(SyncSolver_->solver()->net()->input_blobs())
      :
      const_cast<vector<Blob<float>*>&>(SyncSolver_->workers()[ID]->solver()->net()->input_blobs());
    
    for (unsigned int i = 0; i < input_blobs.size(); ++i) {
      const mxArray* const elem = mxGetCell(input, i);

      CHECK(mxIsSingle(elem))
        << "MatCaffe require single-precision float point data";
      int dim_num = (int)mxGetNumberOfDimensions(elem);
      const mwSize *dims = mxGetDimensions(elem);
      input_blobs[i]->Reshape(dim_num >= 4 ? dims[3] : 1, dim_num >= 3 ? dims[2] : 1, dim_num >= 2 ? dims[1] : 1, dim_num >= 1 ? dims[0] : 1);
      const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
      switch (Caffe::mode()) {
      case Caffe::CPU:
        caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_cpu_data());
        break;
      case Caffe::GPU:
        caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_gpu_data());
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
    
    float loss;
    vector<string> output_names;
    vector<vector<float> > results;
    vector<float> weights;
//  printf("card %d step once...\n",ID);
    if (ID == 0)
      SyncSolver_->solver()->StepOneIter(loss, output_names, results, weights);
    else
      SyncSolver_->workers()[ID]->solver()->StepOneIter(loss, output_names, results, weights);
//  printf("card %d out solver!\n",ID);

    if (ID == 0)
    {
      {
        const mwSize dims[2] = { int(output_names.size()), 1 };
        const char* fnames[3] = { "output_name", "results", "weight" };
        mx_out = mxCreateStructArray(2, dims, 3, fnames);
      }
      mxArray* mx_result = NULL;
      for (int i = 0; i < (int)output_names.size(); ++i){
        mxSetField(mx_out, i, "output_name", mxCreateString(output_names[i].c_str()));
        mx_result = mxCreateDoubleMatrix((int)results[i].size(), 1, mxREAL);
        double *p_mx_result = mxGetPr(mx_result);
        for (int j = 0; j < (int)results[i].size(); ++j){
          p_mx_result[j] = results[i][j];
        }
        mxSetField(mx_out, i, "results", mx_result);
        mxSetField(mx_out, i, "weight", mxCreateDoubleScalar(weights[i]));
      }
    }
  }
  return mx_out;
}

static void train(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  plhs[0] = do_train(prhs[0]);
}


// function do_test
// input:  bottom[cell of (cell of blobs) for multi-gpus]
// output: top[struct of output]
static mxArray* do_test(const mxArray* const bottom) {

  if (!SyncSolver_.get())
  {
    mexPrintf("caffe_mex : No solver inited!\n");
    return mxCreateDoubleMatrix(0, 0, mxREAL);
  }


  if (mxGetNumberOfElements(bottom) != gpu_group.size())
    mexErrMsgTxt("caffe_mex : do_test:input size should be equal to selected gpu number.\n");

	vector<boost::shared_ptr<Layer<float> > >& layers = SyncSolver_->solver()->net()->getlayers();
	for(int i = 0 ; i < layers.size(); ++i){
		layers[i]->SetPhase(TEST);
	}

	for(int i = 1 ; i < int(gpu_group.size()); ++i){
		vector<boost::shared_ptr<Layer<float> > >& layers = SyncSolver_->workers()[i]->solver()->net()->getlayers();
		for(int j = 0; j < layers.size(); ++j){
			layers[j]->SetPhase(TEST);
		}
	} 



  mxArray* mx_out = NULL;
  int cols = int(SyncSolver_->solver()->net()->output_blob_indices().size());
  mx_out = mxCreateCellMatrix(int(gpu_group.size()), cols);

  vector<string> output_names;
  for (int j = 0; j < cols; ++j) {
    const string& output_name =
      SyncSolver_->solver()->net()->blob_names()[SyncSolver_->solver()->net()->output_blob_indices()[j]];
    output_names.push_back(output_name);
  }
  //#pragma omp parallel for num_threads(int(gpu_group.size()))
  //  for (int ID = 0; ID < gpu_group.size(); ++ID)
  //  {
#pragma omp parallel num_threads(int(gpu_group.size())) 
  {
  int ID = omp_get_thread_num();
  //printf("ID = %d\n",ID);
    const mxArray* const input = mxGetCell(bottom, ID);
    vector<Blob<float>*>& input_blobs = 
      ID == 0 ?
      const_cast<vector<Blob<float>*>&>(SyncSolver_->solver()->net()->input_blobs())
      :
      const_cast<vector<Blob<float>*>&>(SyncSolver_->workers()[ID]->solver()->net()->input_blobs());

    for (unsigned int i = 0; i < input_blobs.size(); ++i) {
      const mxArray* const elem = mxGetCell(input, i);

      CHECK(mxIsSingle(elem))
        << "MatCaffe require single-precision float point data";
      int dim_num = (int)mxGetNumberOfDimensions(elem);
      const mwSize *dims = mxGetDimensions(elem);
      input_blobs[i]->Reshape(dim_num >= 4 ? dims[3] : 1, dim_num >= 3 ? dims[2] : 1, dim_num >= 2 ? dims[1] : 1, dim_num >= 1 ? dims[0] : 1);

      const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
      switch (Caffe::mode()) {
      case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_gpu_data());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  //printf("%d forwarding...\n",ID);

  const vector<Blob<float>*>& result =
    ID == 0 ?
    SyncSolver_->solver()->ForwardOnce()
    :
    SyncSolver_->workers()[ID]->solver()->ForwardOnce();

  //printf("%d forward done!\n",ID);


  if (ID == 0)
  {
    vector<vector<float> > results;
    vector<float> weights;

    results.resize(result.size());
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      const float loss_weight =
        ID == 0 ?
        SyncSolver_->solver()->net()->blob_loss_weights()[SyncSolver_->solver()->net()->output_blob_indices()[j]]
        :
        SyncSolver_->workers()[ID]->solver()->net()->blob_loss_weights()[SyncSolver_->workers()[ID]->solver()->net()->output_blob_indices()[j]];
      weights.push_back(loss_weight);
      for (int k = 0; k < result[j]->count(); ++k) {
        results[j].push_back(result_vec[k]);
      }
    }
    const mwSize dims[2] = { (int)output_names.size(), 1 };
    const char* fnames[3] = { "output_name", "results", "weight" };
    mx_out = mxCreateStructArray(2, dims, 3, fnames);
    mxArray* mx_result = NULL;
    for (int i = 0; i < (int)output_names.size(); ++i){
      mxSetField(mx_out, i, "output_name", mxCreateString(output_names[i].c_str()));
      mx_result = mxCreateDoubleMatrix((int)results[i].size(), 1, mxREAL);
      double *p_mx_result = mxGetPr(mx_result);
      for (int j = 0; j < (int)results[i].size(); ++j){
        p_mx_result[j] = results[i][j];
      }
      mxSetField(mx_out, i, "results", mx_result);
      mxSetField(mx_out, i, "weight", mxCreateDoubleScalar(weights[i]));
    }
  }

  //printf("card %d exit!\n",ID);
  }
  return mx_out;
}

static void test(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  plhs[0] = do_test(prhs[0]);
}

// function get_solver_max_iter
// input:  
// output: max_iter
static void get_solver_max_iter(MEX_ARGS) {
  if (!SyncSolver_.get())
  {
    mexPrintf("No solver inited!\n");
    plhs[0] = mxCreateDoubleScalar(-1);
    return;
  }

  plhs[0] = mxCreateDoubleScalar(SyncSolver_->solver()->max_iter());
}


// function get_solver_iter
// input:  
// output: iter
static void get_solver_iter(MEX_ARGS) {
  if (!SyncSolver_.get())
  {
    mexPrintf("No solver inited!\n");
    plhs[0] = mxCreateDoubleScalar(-1);
    return;
  }

  plhs[0] = mxCreateDoubleScalar(SyncSolver_->solver()->iter());
}

// function set_device_solver
// input:  device_ids
// output: 
static void set_device_solver(MEX_ARGS) {

  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (SyncSolver_.get())
    mexErrMsgTxt("caffe::set_device_solver solver_devices have already be set. If you want to change it, please call release_solver first");


  if (!mxIsDouble(prhs[0]))
    mexErrMsgTxt("caffe::set_device_solver device_ids only supports double");

  int iDeviceNum = 0;
  CUDA_CHECK(::cudaGetDeviceCount(&iDeviceNum));
  if (iDeviceNum <= 0)
    mexErrMsgTxt("caffe::set_device_solver Do not find CUDA devices.");

  const double* const data_ptr =
    reinterpret_cast<const double* const>(mxGetPr(prhs[0]));

  gpu_group.clear();
  const int ele_num = mxGetNumberOfElements(prhs[0]);
  for (int i = 0; i < ele_num; ++i)
  {
    int device_id = int(data_ptr[i]);
    if (device_id >= iDeviceNum)
      mexErrMsgTxt("caffe::set_device_solver device_id should in [0, gpuDeviceCount-1]");
    gpu_group.push_back(device_id);
  }
}

// function snapshot
// input:  file_name
// output: 
static void snapshot(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (!SyncSolver_.get())
  {
    mexPrintf("No solver inited!\n");
    return;
  }

  char* filename = mxArrayToString(prhs[0]);
  string filenameStr = filename;
  SyncSolver_->solver()->Snapshot(filenameStr);

}

static void do_set_weights(boost::shared_ptr<Net<float> > net, const mxArray* mx_layers) {
  //LOG(INFO) << "do_set_weights in\n";
  const vector<boost::shared_ptr<Layer<float> > >& layers = net->layers();
  const vector<string>& layer_names = net->layer_names();

  unsigned int input_layer_num = mxGetNumberOfElements(mx_layers);
  for (unsigned int i = 0; i < input_layer_num; ++i)
  {
    // Step 1: get input layer information
    mxArray *mx_layer_name = mxGetField(mx_layers, i, "layer_names");
    if (mx_layer_name == NULL)
    {
      mexPrintf("layer %d has no field ""layer_names"", ignore\n", i);
      continue;
    }
    char *layer_name = mxArrayToString(mx_layer_name);
    mxArray *mx_weights_cell = mxGetField(mx_layers, i, "weights");
    if (mx_weights_cell == NULL)
    {
      mexPrintf("layer %d has no field ""weights"", ignore\n", i);
      continue;
    }
    if (!mxIsCell(mx_weights_cell))
    {
      mexPrintf("layer %d field ""weights"" is not cell, ignore\n", i);
      continue;
    }
    unsigned int weight_blob_num = mxGetNumberOfElements(mx_weights_cell);

    // Step 2: scan model layers, and try to set layer
    string prev_layer_name = "";
    for (unsigned int j = 0; j < layers.size(); ++j) {

      vector<boost::shared_ptr<Blob<float> > >& layer_blobs = layers[j]->blobs();
      if (layer_blobs.size() == 0)
        continue;

      if (layer_names[j] != string(layer_name))
        continue;

      if (weight_blob_num != layer_blobs.size())
      {
        mexPrintf("%s has % blobs, while model layer has %d blobs, ignore\n", layer_name, weight_blob_num, layer_blobs.size());
        continue;
      }

      //LOG(INFO) << "start processing " << string(layer_name) << "\n";


      for (unsigned int k = 0; k < layer_blobs.size(); ++k) {
        bool setted = false;
        mxArray *mx_weights = mxGetCell(mx_weights_cell, k);
#ifdef WIN32
        const size_t* input_blob_dims = mxGetDimensions(mx_weights);
#else
        const int* input_blob_dims = mxGetDimensions(mx_weights);
#endif // WIN32
        int dim_num = mxGetNumberOfDimensions(mx_weights);
        size_t input_dims[4] = { 1, 1, 1, 1 };


        for (int idim = 0; idim < dim_num; ++idim)
        {
          input_dims[idim] = input_blob_dims[idim];
        }


        //LOG(INFO) << "input_dims = " << input_dims[0] << " "
        //  << input_dims[1] << " "
        //  << input_dims[2] << " "
        //  << input_dims[3] << "\n";
        //LOG(INFO) << "layer_bolbs[" << k << "] = " << layer_blobs[k]->width() << " " <<
        //  layer_blobs[k]->height() << " " <<
        //  layer_blobs[k]->channels() << " " <<
        //  layer_blobs[k]->num() << "\n";

        if (layer_blobs[k]->width() != (int)input_dims[0] || layer_blobs[k]->height() != (int)input_dims[1] || layer_blobs[k]->channels() != (int)input_dims[2] || layer_blobs[k]->num() != (int)input_dims[3])
        {
          mexPrintf("%s blobs %d dims don't match, ignore\n", layer_name, k);
          continue;
        }

        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
          case Caffe::CPU:
            caffe_copy(layer_blobs[k]->count(), weights_ptr, layer_blobs[k]->mutable_cpu_data());
            setted = true;
            break;
          case Caffe::GPU:
            caffe_copy(layer_blobs[k]->count(), weights_ptr, layer_blobs[k]->mutable_gpu_data());
            setted = true;
            break;
          default:
            LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
        //LOG(INFO) << "k = " << k << " setted = " << (int)setted << "\n";
        if (setted)
          LOG(INFO) << "Copied weights for " << layer_name << " blob " << k << "\n";
      }

    }
    mxFree(layer_name);
  }
}

// function get_weights_solver
// input:  gpu_id[opt, default = 0]
// output: weights
static void get_weights_solver(MEX_ARGS) {
  if (nrhs != 0 && nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (!SyncSolver_.get())
  {
    mexPrintf("No solver inited!\n");
    plhs[0] = mxCreateDoubleScalar(-1);
    return;
  }

  plhs[0] = do_get_weights(SyncSolver_->solver()->net());
}

// function get_response_solver
// input:  blob_name 
// output: response
static void get_response_solver(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (!SyncSolver_.get())
  {
    mexPrintf("No solver inited!\n");
    plhs[0] = mxCreateDoubleScalar(-1);
    return;
  }

  char* blob_name = mxArrayToString(prhs[0]);

  mxArray* top;
  top = mxCreateCellMatrix(int(gpu_group.size()), 1);
  mxArray* response = do_get_response(SyncSolver_->solver()->net(), string(blob_name));
  mxSetCell(top, 0, response);
  for (int i = 1; i < gpu_group.size(); ++i)
  {
    mxArray* response = do_get_response(SyncSolver_->workers()[i]->solver()->net(), string(blob_name));
    mxSetCell(top, i, response);
  }


  plhs[0] = top;
}

// function set_weights_solver
// input:  weights
// output: 
static void set_weights_solver(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (!SyncSolver_.get())
  {
    mexPrintf("No solver inited!\n");
    plhs[0] = mxCreateDoubleScalar(-1);
    return;
  }
#pragma omp parallel for num_threads(int(gpu_group.size()))
  for (int ID = 0; ID < gpu_group.size();++ID)
  {
    if (ID == 0)
    {
      LOG(INFO) << "processing solver net" << "\n";
      do_set_weights(SyncSolver_->solver()->net(), prhs[0]);
      //LOG(INFO) << "processing solver test_net" << "\n";
      if (!SyncSolver_->solver()->test_nets().empty())
        do_set_weights(SyncSolver_->solver()->test_nets()[0], prhs[0]);
      LOG(INFO) << "solver done!" << "\n";
    }
    else
    {
      LOG(INFO) << "processing worker net" << "\n";
      do_set_weights(SyncSolver_->workers()[ID]->solver()->net(), prhs[0]);
      if (!SyncSolver_->workers()[ID]->solver()->test_nets().empty())
        do_set_weights(SyncSolver_->workers()[ID]->solver()->test_nets()[0], prhs[0]);
      LOG(INFO) << "worker done!\n";
    }
  }
}


/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void(*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward", forward },
  { "backward", backward },
  { "init", init }, // do
  { "is_initialized", is_initialized }, // do
  { "set_mode_cpu", set_mode_cpu }, // do
  { "set_mode_gpu", set_mode_gpu }, // do
  { "set_device", set_device }, // do
  { "set_input_size", set_input_size },
  { "get_response", get_response }, // do
  { "get_weights", get_weights }, // do
  { "get_init_key", get_init_key }, // do
  { "release", release }, //do
  { "read_mean", read_mean }, // do
  { "set_random_seed", set_random_seed }, //do
	{ "set_phase", set_phase},
  // for solver
  { "init_solver", init_solver }, // done!
  { "recovery_solver", recovery_solver }, // done!
  { "release_solver", release_solver }, // done!
  { "get_solver_iter", get_solver_iter }, //done!
  { "get_solver_max_iter", get_solver_max_iter }, //done!
  { "get_weights_solver", get_weights_solver }, // done!
  { "get_response_solver", get_response_solver }, // done!
  { "set_weights_solver", set_weights_solver },  //done
  { "set_input_size_solver", set_input_size_solver },
  { "set_device_solver", set_device_solver }, // done !
  { "train", train }, // done!
  { "test", test }, // done!
  { "snapshot", snapshot }, // done!
  { "init_log", init_log },
  // The end.
  { "END", NULL },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  if (init_key[0] == -2) {
    init_key[0] = static_cast<double>(caffe_rng_rand());
    initGlog();
  }
  //mexLock();  // Avoid clearing the mex file.
  mxCHECK(nrhs > 0, "Usage: caffe_(api_command, arg1, arg2, ...)");
  {// Handle input command
    char* cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs - 1, prhs + 1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      ostringstream error_msg;
      error_msg << "Unknown command '" << cmd << "'";
      mxERROR(error_msg.str().c_str());
    }
    mxFree(cmd);
  }
}
