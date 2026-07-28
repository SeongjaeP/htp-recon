// qnn_min_runtime.c — a minimal inference runtime, written from scratch
//
// Why this exists: before reasoning about what the compiler decides, it helps to be able to run
// a graph and get bytes out. The SDK ships an API tracing layer that records every call an
// application makes and replays it as generated C. Reading that trace gives the call sequence;
// this file is the sequence rewritten by hand, minimally, with the generated scaffolding removed.
//
// The point of rewriting rather than reusing the generated code is that a hand-written version
// makes the structure legible: a backend is a shared object exporting one entry point that
// returns a function-pointer table, and everything else is calls through that table. Swapping
// which .so is loaded switches between the CPU reference and the NPU with no code change — which
// is exactly what makes a byte-for-byte comparison between them possible.
//
// Sequence:
//   dlopen(backend) -> QnnInterface_getProviders -> function-pointer table
//   logCreate -> backendCreate -> deviceCreate -> contextCreate
//   dlopen(model)   -> QnnModel_composeGraphs        (creates tensors and adds nodes)
//   graphFinalize   -> load input .raw -> graphExecute -> write output .raw
//
// Result: output from the NPU backend is bit-identical to the CPU reference (0 bytes differ).
//
// Requires the vendor SDK headers to compile, so it is not part of the default build.
// Point CXXFLAGS at the SDK include directory and build it separately.
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "QnnInterface.h"
#include "QnnTypeMacros.hpp"     // tensor accessors that work across tensor struct versions
#include "QnnWrapperUtils.hpp"   // GraphInfo_t, GraphInfoPtr_t, ModelError_t

#define CHECK(err, msg) do { \
  if ((err) != QNN_SUCCESS) { fprintf(stderr, "[FAIL] %s (err=0x%lx)\n", msg, (unsigned long)(err)); return 1; } \
} while(0)

// Entry point exported by a compiled model .so; signature taken from the SDK's own examples.
typedef qnn_wrapper_api::ModelError_t (*ComposeGraphsFn_t)(
    Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE, Qnn_ContextHandle_t,
    const qnn_wrapper_api::GraphConfigInfo_t**, uint32_t,
    qnn_wrapper_api::GraphInfoPtr_t**, uint32_t*, bool,
    QnnLog_Callback_t, QnnLog_Level_t);
typedef qnn_wrapper_api::ModelError_t (*FreeGraphsFn_t)(
    qnn_wrapper_api::GraphInfoPtr_t**, uint32_t);

static size_t dtype_size(Qnn_DataType_t dt) {
  switch (dt) {
    case QNN_DATATYPE_FLOAT_32: return 4;
    case QNN_DATATYPE_FLOAT_16: return 2;
    case QNN_DATATYPE_UINT_32:  return 4;
    case QNN_DATATYPE_INT_32:   return 4;
    case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_SFIXED_POINT_8: return 1;
    default: return 4;
  }
}
static size_t tensor_elems(const Qnn_Tensor_t* t) {
  uint32_t r = QNN_TENSOR_GET_RANK(t);
  const uint32_t* d = QNN_TENSOR_GET_DIMENSIONS(t);
  size_t n = 1; for (uint32_t i=0;i<r;i++) n *= d[i];
  return n;
}
static size_t tensor_bytes(const Qnn_Tensor_t* t) {
  return tensor_elems(t) * dtype_size(QNN_TENSOR_GET_DATA_TYPE(t));
}

static void logcb(const char* fmt, QnnLog_Level_t lvl, uint64_t ts, va_list ap) {
  (void)lvl;(void)ts; vfprintf(stderr, fmt, ap); fprintf(stderr, "\n");
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr,
      "usage: %s <backend.so> <model.so> <input.raw> <output_dir> [label]\n",
      argv[0]);
    return 2;
  }
  const char* backend_path = argv[1];
  const char* model_path   = argv[2];
  const char* input_path   = argv[3];
  const char* out_dir      = argv[4];
  const char* label        = (argc>5)?argv[5]:"run";

  // ---- 1) load the backend and fetch its function-pointer table -----------
  void* beh = dlopen(backend_path, RTLD_NOW|RTLD_LOCAL);
  if (!beh) { fprintf(stderr,"dlopen backend: %s\n", dlerror()); return 1; }
  typedef Qnn_ErrorHandle_t (*GetProvidersFn_t)(const QnnInterface_t***, uint32_t*);
  GetProvidersFn_t getProviders = (GetProvidersFn_t)dlsym(beh, "QnnInterface_getProviders");
  if (!getProviders) { fprintf(stderr,"dlsym getProviders: %s\n", dlerror()); return 1; }

  const QnnInterface_t** providers = NULL; uint32_t nprov = 0;
  CHECK(getProviders(&providers, &nprov), "getProviders");
  if (nprov == 0) { fprintf(stderr,"no providers\n"); return 1; }
  QNN_INTERFACE_VER_TYPE ifc = providers[0]->QNN_INTERFACE_VER_NAME;  // the vtable
  fprintf(stderr,"[%s] backend=%s  provider=%s  #prov=%u\n",
          label, backend_path, providers[0]->providerName, nprov);

  // ---- 2) log -> backend -> device -> context ------------------------------
  Qnn_LogHandle_t log = NULL;
  ifc.logCreate(logcb, QNN_LOG_LEVEL_WARN, &log);   // non-fatal if it fails

  Qnn_BackendHandle_t backend = NULL;
  CHECK(ifc.backendCreate(log, NULL, &backend), "backendCreate");

  Qnn_DeviceHandle_t device = NULL;
  ifc.deviceCreate(log, NULL, &device);             // the CPU backend accepts a NULL device

  Qnn_ContextHandle_t context = NULL;
  CHECK(ifc.contextCreate(backend, device, NULL, &context), "contextCreate");

  // ---- 3) load the model .so and let it assemble tensors and nodes ---------
  void* mh = dlopen(model_path, RTLD_NOW|RTLD_LOCAL);
  if (!mh) { fprintf(stderr,"dlopen model: %s\n", dlerror()); return 1; }
  ComposeGraphsFn_t compose = (ComposeGraphsFn_t)dlsym(mh, "QnnModel_composeGraphs");
  FreeGraphsFn_t    freeg   = (FreeGraphsFn_t)dlsym(mh, "QnnModel_freeGraphsInfo");
  if (!compose) { fprintf(stderr,"dlsym composeGraphs: %s\n", dlerror()); return 1; }

  qnn_wrapper_api::GraphInfoPtr_t* graphs = NULL; uint32_t ngraphs = 0;
  qnn_wrapper_api::ModelError_t merr =
      compose(backend, ifc, context, NULL, 0, &graphs, &ngraphs, false, logcb, QNN_LOG_LEVEL_WARN);
  if (merr != qnn_wrapper_api::MODEL_NO_ERROR) { fprintf(stderr,"composeGraphs err=%d\n", merr); return 1; }
  fprintf(stderr,"[%s] composed %u graph(s)\n", label, ngraphs);

  qnn_wrapper_api::GraphInfo_t* gi = graphs[0];
  fprintf(stderr,"[%s] graph='%s'  in=%u out=%u\n", label, gi->graphName, gi->numInputTensors, gi->numOutputTensors);

  // ---- 4) finalize --------------------------------------------------------
  // This is where the closed compiler runs: lowering, scheduling, allocation. Everything the
  // rest of this repository reconstructs happens inside this one call.
  CHECK(ifc.graphFinalize(gi->graph, NULL, NULL), "graphFinalize");

  // ---- 5) bind buffers ----------------------------------------------------
  // Single input assumed.
  Qnn_Tensor_t* in = &gi->inputTensors[0];
  size_t in_bytes = tensor_bytes(in);
  void* in_buf = malloc(in_bytes);
  FILE* fi = fopen(input_path,"rb");
  if (!fi) { fprintf(stderr,"open input %s\n", input_path); return 1; }
  size_t rd = fread(in_buf, 1, in_bytes, fi); fclose(fi);
  fprintf(stderr,"[%s] input '%s' bytes=%zu read=%zu\n", label, QNN_TENSOR_GET_NAME(in), in_bytes, rd);
  { Qnn_ClientBuffer_t cb = { in_buf, (uint32_t)in_bytes };
    QNN_TENSOR_SET_MEM_TYPE(in, QNN_TENSORMEMTYPE_RAW);
    QNN_TENSOR_SET_CLIENT_BUF(in, cb); }

  Qnn_Tensor_t* out = &gi->outputTensors[0];
  size_t out_bytes = tensor_bytes(out);
  void* out_buf = calloc(1, out_bytes);
  { Qnn_ClientBuffer_t cb = { out_buf, (uint32_t)out_bytes };
    QNN_TENSOR_SET_MEM_TYPE(out, QNN_TENSORMEMTYPE_RAW);
    QNN_TENSOR_SET_CLIENT_BUF(out, cb); }
  fprintf(stderr,"[%s] output '%s' bytes=%zu (elems=%zu)\n", label, QNN_TENSOR_GET_NAME(out), out_bytes, tensor_elems(out));

  // ---- 6) execute ---------------------------------------------------------
  CHECK(ifc.graphExecute(gi->graph, gi->inputTensors, gi->numInputTensors,
                         gi->outputTensors, gi->numOutputTensors, NULL, NULL), "graphExecute");
  fprintf(stderr,"[%s] graphExecute OK\n", label);

  // ---- 7) write the output so it can be byte-compared ---------------------
  char path[1024];
  { char cmd[1100]; snprintf(cmd,sizeof cmd,"mkdir -p '%s'",out_dir); if (system(cmd)!=0){} }
  snprintf(path,sizeof path,"%s/%s.raw", out_dir, label);
  FILE* fo = fopen(path,"wb");
  if (!fo) { fprintf(stderr,"open out %s\n", path); return 1; }
  fwrite(out_buf,1,out_bytes,fo); fclose(fo);
  fprintf(stderr,"[%s] wrote %s\n", label, path);

  // ---- cleanup -----------------------------------------------------------
  if (freeg) freeg(&graphs, ngraphs);
  ifc.contextFree(context, NULL);
  ifc.backendFree(backend);
  if (log) ifc.logFree(log);
  free(in_buf); free(out_buf);
  dlclose(mh); dlclose(beh);
  return 0;
}
