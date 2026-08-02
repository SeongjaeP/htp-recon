// minicc.c — a mini NPU compiler, end to end
//
//   graph -> [lowering] -> [pre-schedule] -> [allocation] -> [final parallel schedule]
//
// The stage ORDER matters and is not the obvious one. The vendor documents it: scheduling and
// allocation both happen inside graph preparation, and the pass that maximises parallelism runs
// AFTER allocation, because it "must obey the restrictions the allocator introduced by
// allocating some blocks at overlapping address ranges".
//
//   lowering        observed 19-op expansion of a 6-op graph, plus the vendor's open-source
//                   MLIR backend, which fixes tile height at 8 (the 2 KB block shape)
//   pre-schedule    a sequential order that keeps on-chip pressure low; this is what fixes
//                   every tensor's lifetime, so allocation depends on it
//   allocation      lifetimes -> in-place linking -> first-fit offsets, and as a side effect,
//                   ordering constraints wherever two tensors were given the same bytes
//   final schedule  parallelise across engines, respecting both real dependencies and the
//                   constraints allocation created
//
// See docs/03-tiling-and-crouton.md, docs/04-scheduling.md, docs/05-memory-allocation.md.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 1024
#define MAXNAME 96
#define MAXTEN 8192
#define TILE_H 8      // crouton height: the hardware's 2 KB block is 8 rows tall
#define ALIGN_LARGE 2048
#define ALIGN_SMALL 128

// ── Resource model ──
// Taken from the device-side execution kernel's diagnostic strings:
//   "Invalid number of HMX thread config: ... Maximum of 1 is allowed"  -> one matrix engine
//   "Invalid number of HVX thread config: ... Must be <= %d"            -> several vector units
//   "Started %d vec workers, %d matrix workers, %d eltwise workers"     -> pools are per engine
//
// A fourth engine (HLX) exists in the kernel but is gated behind a newer architecture:
//   "Detected DSP arch: %u >= v85. Setting num_hlx_ctx to %u"
// The target here reports architecture 81, so HLX is inactive and only three resources are
// modelled. Extending to v85+ means adding R_HLX and mapping softmax onto it.
typedef enum { R_HMX=0, R_HVX=1, R_DMA=2, R_NUM=3 } Resource;
static const char* RNAME[R_NUM] = {"HMX","HVX","DMA"};
#define MAXCAP 8
static int CAP[R_NUM] = { 1, 4, 1 };   // HMX=1 is a hardware limit; HVX is configurable

#define MAXIN 512   // Concat takes one input per tile, so this grows with H
typedef struct {
  int id; char name[MAXNAME]; char type[MAXNAME];
  char ins[MAXIN][MAXNAME]; int nin;
  char outs[2][MAXNAME]; int nout;
  int indeg; int deps[MAXN]; int ndep; int scheduled; int depth;
  Resource res;                        // which engine this op runs on
} Node;
static Node g[MAXN]; static int N=0;

typedef struct { char name[MAXNAME]; long size; } Ten;
static Ten ten[MAXTEN]; static int NT=0;

// ── Anti-dependencies: ordering constraints that ALLOCATION creates ──
//
// The vendor's documentation is explicit about this. When two blocks are given overlapping
// address ranges — legal, because their lifetimes were disjoint under the schedule that was
// current at allocation time — then "two ops that could have been rearranged in any order can
// no longer be swapped, because doing so would cause some blocks of data that were allocated in
// an overlapping manner to be needed at the same time". It adds that the allocator tries to
// limit this, "since these new restrictions can constrain available parallelism".
//
// This is the classic phase-ordering problem: register allocation introduces false dependencies
// that constrain instruction scheduling. Tensors instead of registers, but the same shape.
//
// An edge u -> v means "u must complete before v starts", over and above the real data
// dependencies. It is why the final scheduler has to run AFTER allocation and obey its output.
#define MAXANTI 16384
static int anti_u[MAXANTI], anti_v[MAXANTI];
static int n_anti = 0;

static void add_anti(int u, int v) {
  if (u < 0 || v < 0 || u == v || n_anti >= MAXANTI) return;
  for (int d = 0; d < g[v].ndep; d++) if (g[v].deps[d] == u) return;   // already a real dep
  for (int k = 0; k < n_anti; k++) if (anti_u[k]==u && anti_v[k]==v) return;
  anti_u[n_anti] = u; anti_v[n_anti] = v; n_anti++;
}

static long ceil_div(long a,long b){ return (a+b-1)/b; }
static long align_size(long n){ long a=(n>=ALIGN_LARGE)?ALIGN_LARGE:ALIGN_SMALL; return (n+a-1)&~(a-1); }

static void set_tensor(const char* nm,long bytes){
  for(int i=0;i<NT;i++) if(strcmp(ten[i].name,nm)==0){ ten[i].size=bytes; return; }
  strncpy(ten[NT].name,nm,MAXNAME-1); ten[NT].size=bytes; NT++;
}
static long size_of(const char* nm){
  for(int i=0;i<NT;i++) if(strcmp(ten[i].name,nm)==0) return ten[i].size;
  return 0;
}

// op type -> engine. Matched exactly, not by substring: depthwise convolution runs on the
// vector engine, so a substring test for "Conv" would misfile it onto the matrix engine.
static Resource resource_of(const char* type) {
  if(strcmp(type, "ConvLayer")==0) return R_HMX;
  if(strcmp(type, "MatMul")==0) return R_HMX;
  if(strcmp(type, "Transpose")==0) return R_HVX;
  if(strcmp(type, "EltwiseAdd")==0) return R_HVX;
  if(strcmp(type, "Relu")==0) return R_HVX;
  if(strcmp(type, "SlicePad")==0) return R_HVX;
  if(strcmp(type, "InputSlice")==0) return R_DMA;
  if(strcmp(type, "flat_from_vtcm")==0) return R_DMA;
  if(strcmp(type, "Concat")==0) return R_DMA;
  fprintf(stderr,"unknown op type: %s\n",type); exit(1);
}

static int add_node(const char* name,const char* type){
  Node* n=&g[N]; memset(n,0,sizeof(*n));
  n->id=N; strncpy(n->name,name,MAXNAME-1); strncpy(n->type,type,MAXNAME-1);
  n->res = resource_of(type);          // tag the resource at construction time
  return N++;
}
static void add_in(int nid,const char* t){
  if(g[nid].nin>=MAXIN){ fprintf(stderr,"too many inputs\n"); exit(1); }
  strncpy(g[nid].ins[g[nid].nin++],t,MAXNAME-1);
}
static void add_out(int nid,const char* t){ strncpy(g[nid].outs[g[nid].nout++],t,MAXNAME-1); }

// ── Stage 1: lowering — one high-level Conv into a tiled low-level graph ──
// The pattern reproduced here is the one the vendor compiler emits:
//   InputSlice -> Transpose -> (SlicePad -> flat_from_vtcm -> ConvLayer) x tiles -> Concat
static void lower_conv(const char* prefix, const char* graph_input,
                       int H,int W,int OC,int Cin,int ebytes, char* final_out){
  int tiles = (int)ceil_div(H, TILE_H);
  char nm[MAXNAME], t_in[MAXNAME], t_trans[MAXNAME];

  // InputSlice
  snprintf(nm,sizeof nm,"%s_islice",prefix); int nIS=add_node(nm,"InputSlice");
  add_in(nIS,graph_input);
  snprintf(t_in,sizeof t_in,"%s_islice_out",prefix); add_out(nIS,t_in);
  set_tensor(t_in, align_size((long)H*W*Cin*ebytes));

  // Transpose into the layout the engines address (NHWC)
  snprintf(nm,sizeof nm,"%s_transpose",prefix); int nTR=add_node(nm,"Transpose");
  add_in(nTR,t_in);
  snprintf(t_trans,sizeof t_trans,"%s_nhwc",prefix); add_out(nTR,t_trans);
  set_tensor(t_trans, align_size((long)H*W*Cin*ebytes));

  // Per tile: SlicePad -> flat_from_vtcm -> ConvLayer -> Relu
  static char tile_outs[MAXIN][MAXNAME]; // Concat's inputs, one per tile
  if(tiles>MAXIN){ fprintf(stderr,"too many tiles\n"); exit(1); }
  for(int t=0;t<tiles;t++){
    int th = (t<tiles-1)? TILE_H : (H-(tiles-1)*TILE_H); // last tile may be a short remainder
    long halo_in = align_size((long)(th+2)*(W+2)*Cin*ebytes); // +2 rows/cols = 3x3 kernel halo
    long tile_out= align_size((long)th*W*OC*ebytes);

    // SlicePad — cuts an overlapping slice; a plain slice would be one row short at each edge
    snprintf(nm,sizeof nm,"%s_slicepad%d",prefix,t); int nSP=add_node(nm,"SlicePad");
    add_in(nSP,t_trans);
    char sp_out[MAXNAME]; snprintf(sp_out,sizeof sp_out,"%s_sp%d",prefix,t); add_out(nSP,sp_out);
    set_tensor(sp_out, halo_in);

    // flat_from_vtcm — stage the slice into on-chip memory
    snprintf(nm,sizeof nm,"%s_vtcm%d",prefix,t); int nVT=add_node(nm,"flat_from_vtcm");
    add_in(nVT,sp_out);
    char vt_out[MAXNAME]; snprintf(vt_out,sizeof vt_out,"%s_vt%d",prefix,t); add_out(nVT,vt_out);
    set_tensor(vt_out, halo_in);

    // ConvLayer — compute this tile
    snprintf(nm,sizeof nm,"%s_conv%d",prefix,t); int nCV=add_node(nm,"ConvLayer");
    add_in(nCV,vt_out);
    char cv_out[MAXNAME]; snprintf(cv_out,sizeof cv_out,"%s_c%d",prefix,t); add_out(nCV,cv_out);
    set_tensor(cv_out, tile_out);

    // Relu — real networks nearly always follow a conv with an activation.
    // This op is the in-place candidate: elementwise, same size as its input, and the only
    // consumer of the conv output.
    // (The real hardware instead fuses relu into the matrix engine's output stage. Keeping it
    //  separate and recovering the memory via in-place reuse is a different trade-off.)
    snprintf(nm,sizeof nm,"%s_relu%d",prefix,t); int nRL=add_node(nm,"Relu");
    add_in(nRL,cv_out);
    char rl_out[MAXNAME]; snprintf(rl_out,sizeof rl_out,"%s_r%d",prefix,t); add_out(nRL,rl_out);
    set_tensor(rl_out, tile_out);            // same size as the conv output -> in-place condition 2
    strncpy(tile_outs[t],rl_out,MAXNAME-1);  // Concat consumes the relu outputs
  }

  // Concat — reassemble the tiles
  snprintf(nm,sizeof nm,"%s_concat",prefix); int nCC=add_node(nm,"Concat");
  for(int t=0;t<tiles;t++) add_in(nCC,tile_outs[t]);
  snprintf(final_out,MAXNAME,"%s_out",prefix); add_out(nCC,final_out);
  set_tensor(final_out, align_size((long)H*W*OC*ebytes));
}

// ── DAG construction ──
static int producer_of(const char* t){
  for(int i=0;i<N;i++) for(int k=0;k<g[i].nout;k++) if(strcmp(g[i].outs[k],t)==0) return i;
  return -1;
}
// For every op, for every input tensor: whoever produces that tensor is a dependency.
// indeg counts unmet dependencies, so indeg == 0 means "ready to run".
static void build_dag(void){
  for(int i=0;i<N;i++) for(int k=0;k<g[i].nin;k++){int p=producer_of(g[i].ins[k]);
    if(p>=0&&p!=i){int dup=0;for(int d=0;d<g[i].ndep;d++)if(g[i].deps[d]==p)dup=1;
      if(!dup){g[i].deps[g[i].ndep++]=p;g[i].indeg++;}}}
}

// ── Cost-based scheduling (kept for comparison; see scheduler_cbs.c for the standalone study) ──
// cost = bytes newly made live - bytes freed, i.e. the change in memory pressure.
static long cost_delta(int i){
  long d=0; for(int k=0;k<g[i].nout;k++) d+=size_of(g[i].outs[k]);
  for(int k=0;k<g[i].nin;k++){int last=1;
    for(int j=0;j<N;j++){if(j==i||g[j].scheduled)continue;
      for(int m=0;m<g[j].nin;m++)if(strcmp(g[j].ins[m],g[i].ins[k])==0)last=0;}
    if(last)d-=size_of(g[i].ins[k]);}
  return d;
}
static void schedule_cbs(int* rl,int* rn){
  int indeg[MAXN]; for(int i=0;i<N;i++){indeg[i]=g[i].indeg;g[i].scheduled=0;} *rn=0;
  for(;;){int best=-1;long bc=0;
    for(int i=0;i<N;i++){if(g[i].scheduled||indeg[i]!=0)continue;long c=cost_delta(i);
      if(best<0||c<bc||(c==bc&&g[i].id<g[best].id)){best=i;bc=c;}}
    if(best<0)break; g[best].scheduled=1; rl[(*rn)++]=best;
    for(int i=0;i<N;i++)for(int d=0;d<g[i].ndep;d++)if(g[i].deps[d]==best)indeg[i]--;}
}

// ── Stage 2: parallel scheduling ──
// Each time slot admits up to CAP[r] ready ops per resource, so the schedule becomes
// two-dimensional: time x resource.
//   slot[t][r][k] = the op running at time t on resource r, worker k  (-1 = idle)
//   rl[]          = the same schedule flattened into a sequence, which the allocator consumes
static void schedule_par(int slot[][R_NUM][MAXCAP], int* nslots, int* rl, int* rn) {
  int indeg[MAXN];
  // Real data dependencies plus whatever ordering the allocator imposed. With n_anti == 0 this
  // is the ordinary parallel schedule; after allocation it is the constrained one.
  for (int i = 0; i < N; i++) { indeg[i] = g[i].indeg; g[i].scheduled = 0; }
  for (int k = 0; k < n_anti; k++) indeg[anti_v[k]]++;
  *rn = 0;
  int t = 0;
  for (;; t++) {
    int picked[R_NUM][MAXCAP], nfilled[R_NUM] = {0, 0, 0};
    for (int r = 0; r < R_NUM; r++) for (int k = 0; k < MAXCAP; k++) picked[r][k] = -1;

    // Assign ready ops (indeg == 0, not yet scheduled) to free workers on their own resource.
    for (int i = 0; i < N; i++) {
      if (g[i].scheduled || indeg[i] != 0) continue;
      Resource r = g[i].res;
      if (nfilled[r] >= CAP[r]) continue; // that engine is already full this slot
      picked[r][nfilled[r]++] = i;
    }

    // Nothing ready anywhere -> done.
    int any = 0;
    for (int r = 0; r < R_NUM; r++) if (nfilled[r] > 0) any = 1;
    if (!any) break;

    // Commit the picks: record the slot, mark scheduled, append to the flat runlist.
    for (int r = 0; r < R_NUM; r++) for (int k = 0; k < nfilled[r]; k++) {
      int b = picked[r][k];
      slot[t][r][k] = b;
      g[b].scheduled = 1;
      rl[(*rn)++] = b;
    }

    // Release successors of everything picked this slot, along both edge kinds.
    for (int r = 0; r < R_NUM; r++) for (int k = 0; k < nfilled[r]; k++) {
      int b = picked[r][k];
      for (int i = 0; i < N; i++) for (int d = 0; d < g[i].ndep; d++)
        if (g[i].deps[d] == b) indeg[i]--;
      for (int e = 0; e < n_anti; e++) if (anti_u[e] == b) indeg[anti_v[e]]--;
    }
  }
  *nslots = t;
}

// ── Stages 3 and 4: in-place linking, then VTCM allocation ──
typedef struct{char name[MAXNAME];long size;int birth,death;long offset;
               int alias_of;} Live;   // alias_of = tensor sharing this address (-1 = independent)
static Live L[MAXTEN]; static int nL=0;
// The entire safety condition for sharing an address: lifetimes must not overlap.
static int overlaps(int a,int b){ return !(L[a].death<L[b].birth||L[b].death<L[a].birth); }
static int find_live(const char* nm){                  // tensor name -> index into L[]
  for(int i=0;i<nL;i++) if(strcmp(L[i].name,nm)==0) return i;
  return -1;
}

// Which ops may write their result over their own input?
// Only ops of the form y[i] = f(x[i]) — read index equals write index.
//   Convolution:  no. One output reads several inputs, so overwriting destroys data a later
//                 output still needs.
//   Transpose:    no, and for a subtler reason. It is one-to-one, but read and write positions
//                 differ, so writing y[1] can clobber an x[1] that y[2] still needs.
static int is_destructive_op(const char* type) {
  if(strcmp(type, "Relu")==0) return 1;
  if(strcmp(type, "EltwiseAdd")==0) return 1;
  return 0;
}

// Link outputs onto inputs where destructive reuse is legal. Run BEFORE allocation, matching
// the vendor's order (its allocation stage calls link_source_destructive_operands first), so
// that sharing is known by the time offsets are chosen.
//   pos[node] = that node's step in the runlist
static int link_inplace(int* pos) {
  int linked=0;
  for (int j = 0; j < N; j++) {
    if (!is_destructive_op(g[j].type)) continue;
    if (g[j].nout != 1) continue;                 // single output only
    int oi = find_live(g[j].outs[0]);
    if (oi < 0) continue;
    for (int k = 0; k < g[j].nin; k++) {
      int ii = find_live(g[j].ins[k]);
      if (ii < 0) continue;
      if (L[ii].size != L[oi].size) continue;     // sizes must match exactly
      if (L[ii].death != pos[j]) continue;        // this op must be the input's LAST consumer
      if (L[ii].alias_of >= 0) continue;          // already aliased by someone else
      L[oi].alias_of = ii;
      linked++;
      break;
    }
  }
  return linked;
}

static void alloc_vtcm(int* rl,int rn,long vtcm,long* peak_o,long* spill_o){
  int pos[MAXN]; for(int s=0;s<rn;s++) pos[rl[s]]=s;
  nL=0;
  // Lifetimes come from the schedule: born when produced, dead after the last consumer runs.
  for(int j=0;j<N;j++) for(int k=0;k<g[j].nout;k++){
    const char* tn=g[j].outs[k]; long sz=size_of(tn); if(sz==0)continue;
    int birth=pos[j],death=birth;
    for(int c=0;c<N;c++)for(int m=0;m<g[c].nin;m++)if(strcmp(g[c].ins[m],tn)==0&&pos[c]>death)death=pos[c];
    strncpy(L[nL].name,tn,MAXNAME-1); L[nL].size=sz; L[nL].birth=birth; L[nL].death=death;
    L[nL].offset=-1; L[nL].alias_of=-1; nL++;
  }

  int nlinked = link_inplace(pos);
  if(nlinked) printf("   [in-place links: %d]\n", nlinked);

  // Place in birth order, so the already-placed tensors are exactly those that could conflict.
  int order[MAXTEN]; for(int i=0;i<nL;i++)order[i]=i;
  for(int i=0;i<nL;i++)for(int j=i+1;j<nL;j++)if(L[order[j]].birth<L[order[i]].birth){int t=order[i];order[i]=order[j];order[j]=t;}
  long peak=0,spill=0;
  for(int oi=0;oi<nL;oi++){int i=order[oi];
    // An in-place alias needs no search at all: it takes its source's offset.
    if(L[i].alias_of>=0 && L[L[i].alias_of].offset>=0){
      L[i].offset = L[L[i].alias_of].offset;
      long e=L[i].offset+L[i].size;
      if(e>vtcm) spill += (e-(vtcm>L[i].offset?vtcm:L[i].offset));
      if(e>peak) peak=e;
      continue;
    }
    // Candidate offsets: 0, plus the top of every overlapping tensor already placed.
    long cands[MAXTEN+1];int nc=0;cands[nc++]=0;
    for(int oj=0;oj<nL;oj++){int j=order[oj];if(j==i||L[j].offset<0)continue;if(overlaps(i,j))cands[nc++]=L[j].offset+L[j].size;}
    for(int a=0;a<nc;a++)for(int b=a+1;b<nc;b++)if(cands[b]<cands[a]){long t=cands[a];cands[a]=cands[b];cands[b]=t;}
    // First fit: lowest candidate that collides with nothing live at the same time.
    long chosen=-1;
    for(int ci=0;ci<nc;ci++){long off=cands[ci];int conf=0;
      for(int oj=0;oj<nL;oj++){int j=order[oj];if(j==i||L[j].offset<0||!overlaps(i,j))continue;
        if(!(off+L[i].size<=L[j].offset||L[j].offset+L[j].size<=off)){conf=1;break;}}
      if(!conf){chosen=off;break;}}
    if(chosen<0)chosen=0; L[i].offset=chosen;
    long end=chosen+L[i].size; if(end>vtcm)spill+=(end-(vtcm>chosen?vtcm:chosen)); if(end>peak)peak=end;
  }
  *peak_o=peak; *spill_o=spill;
}

// ── Stage 5: read the ordering constraints allocation just created ──
//
// Two tensors were given overlapping addresses because their lifetimes did not overlap under
// the schedule allocation saw. That is only safe while that ordering holds. If the later
// tensor were produced before the earlier one is finished with, both would need the same bytes
// at once.
//
// So for every overlapping pair, the op that last consumes the earlier tensor must complete
// before the op that produces the later one. rl[] maps a schedule position back to its node,
// which is how a lifetime endpoint becomes an op.
//
// In-place aliases are skipped: they share an address deliberately, and the real data
// dependency between them already enforces the order.
static int addr_overlap(int a, int b) {
  long ae = L[a].offset + L[a].size, be = L[b].offset + L[b].size;
  return !(ae <= L[b].offset || be <= L[a].offset);
}

static int record_anti_deps(int* rl, int rn) {
  n_anti = 0;
  for (int i = 0; i < nL; i++) {
    for (int j = i + 1; j < nL; j++) {
      if (L[i].offset < 0 || L[j].offset < 0) continue;
      if (L[i].alias_of == j || L[j].alias_of == i) continue;   // same buffer on purpose
      if (overlaps(i, j)) continue;                             // lifetimes clash: not sharing
      if (!addr_overlap(i, j)) continue;                        // disjoint addresses: no constraint
      int e = (L[i].death <= L[j].birth) ? i : j;               // earlier / later by lifetime
      int l = (e == i) ? j : i;
      if (L[e].death >= rn || L[l].birth >= rn) continue;
      add_anti(rl[L[e].death], rl[L[l].birth]);
    }
  }
  return n_anti;
}

int main(int argc,char**argv){
  long vtcm = (argc>1)? atol(argv[1]) : (2L*1024*1024);
  int H = (argc>2)? atoi(argv[2]) : 64;
  int W = (argc>3)? atoi(argv[3]) : 64;
  int OC= (argc>4)? atoi(argv[4]) : 32;

  printf("========= mini HTP compiler =========\n");
  printf("input high-level graph: Conv [1,%d,%d,%d] (Cin=%d), fp16\n\n", H,W,OC,OC);

  // Stage 1 — lowering
  char fout[MAXNAME];
  set_tensor("graph_input", align_size((long)H*W*OC*2));
  lower_conv("cv", "graph_input", H,W,OC,/*Cin*/OC,/*ebytes*/2, fout);
  printf("(1) Lowering: 1 high-level Conv -> %d low-level ops (tiles = ceil(%d/8) = %ld)\n",
         N, H, ceil_div(H,TILE_H));
  for(int i=0;i<N;i++) printf("    %-16s %s\n", g[i].name, g[i].type);

  // Stage 2 — pre-schedule. A sequential order is what allocation will see, and it is what
  // fixes every tensor's lifetime. The vendor calls this the pre-scheduler; its job is an order
  // that keeps on-chip pressure low, so the cost-based pass is the right one here.
  build_dag();
  int rl[MAXN], rn;
  schedule_cbs(rl, &rn);
  printf("\n(2) Pre-schedule (sequential, memory-cost order): %d ops\n", rn);

  // Stage 3 — allocation, on that order
  long peak, spill;
  alloc_vtcm(rl, rn, vtcm, &peak, &spill);
  printf("\n(3) Allocation (VTCM %ld KB, first-fit + lifetime reuse):\n", vtcm/1024);
  long total=0; for(int i=0;i<nL;i++) total+=L[i].size;
  printf("    %d tensors, %ld KB total -> %ld KB peak (%ld KB saved by reuse), %ld KB spill\n",
         nL, total/1024, peak/1024, (total-peak)/1024, spill/1024);

  // Stage 4 — read back the ordering constraints that allocation just created
  int na = record_anti_deps(rl, rn);
  printf("\n(4) Anti-dependencies created by allocation: %d\n", na);

  // Stage 5 — final parallel re-schedule, obeying them.
  // Run it twice to price the constraints: once ignoring them (which is what parallelising
  // BEFORE allocation would have produced) and once respecting them.
  static int slot[MAXN][R_NUM][MAXCAP];
  int rl2[MAXN], rn2, nfree;
  int keep = n_anti; n_anti = 0;
  for(int t=0;t<MAXN;t++) for(int r=0;r<R_NUM;r++) for(int k=0;k<MAXCAP;k++) slot[t][r][k]=-1;
  schedule_par(slot, &nfree, rl2, &rn2);
  n_anti = keep;

  int nslots;
  for(int t=0;t<MAXN;t++) for(int r=0;r<R_NUM;r++) for(int k=0;k<MAXCAP;k++) slot[t][r][k]=-1;
  schedule_par(slot, &nslots, rl2, &rn2);

  printf("\n(5) Final schedule (parallel; HMX=%d HVX=%d DMA=%d): time x resource\n",
         CAP[R_HMX],CAP[R_HVX],CAP[R_DMA]);
  printf("    %-5s | %-26s| %-26s| %s\n","slot",RNAME[0],RNAME[1],RNAME[2]);
  printf("    ------+---------------------------+---------------------------+---------------------------\n");
  for(int t=0;t<nslots;t++){
    printf("    t%-4d |",t);
    for(int r=0;r<R_NUM;r++){
      char buf[256]=""; int first=1;
      for(int k=0;k<CAP[r];k++){
        int b=slot[t][r][k]; if(b<0) continue;
        if(!first) strncat(buf,",",sizeof buf-strlen(buf)-1);
        strncat(buf,g[b].name,sizeof buf-strlen(buf)-1); first=0;
      }
      if(first) strcpy(buf,"-");
      printf(" %-26s%s", buf, r==R_NUM-1 ? "" : "|");
    }
    printf("\n");
  }
  printf("    sequential            : %d slots\n", N);
  printf("    parallel, unconstrained: %d slots  (%.1f%% fewer)\n",
         nfree, 100.0*(N-nfree)/N);
  printf("    parallel, as allocated : %d slots  (%.1f%% fewer)"
         "   <- %d slots given back to allocation\n",
         nslots, 100.0*(N-nslots)/N, nslots-nfree);

  return 0;
}
