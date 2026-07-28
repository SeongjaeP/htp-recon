// allocator.c — VTCM offset assignment
//
// Where the scheduler decides *when* an op runs, the allocator decides *where* each tensor
// lives. This is register allocation's problem in a different unit: tensors instead of values,
// byte offsets instead of registers, and the same question of which lifetimes may share storage.
//
// The order matters, and it is not arbitrary. A tensor's lifetime is [produced, last consumed]
// in schedule positions, so the schedule is what *creates* the lifetimes. Change the order and
// every lifetime changes with it — which is why scheduling runs first.
//
// One structural detail was taken from the vendor's own stage ordering: its allocation stage
// links destructive (in-place) operands BEFORE allocating, so that sharing is already known when
// offsets are chosen. This file does the same.
//
// Input:  graph.txt + sizes.txt + [vtcm_bytes]
// Output: each tensor's offset, the peak footprint, and the number of bytes that exceed VTCM
//         (spill)
//
// See docs/05-memory-allocation.md.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 512
#define MAXNAME 128
#define MAXTEN 4096

// ── Graph (same shape as scheduler_cbs.c) ──
typedef struct {
  int id; char name[MAXNAME]; char type[MAXNAME];
  char *ins[16]; int nin; char *outs[8]; int nout;
  int indeg; int deps[MAXN]; int ndep; int scheduled; int depth;
} Node;
static Node g[MAXN]; static int N=0;
static char tenName[MAXTEN][MAXNAME]; static long tenSize[MAXTEN]; static int NT=0;
static long size_of_tensor(const char* t){
  for(int i=0;i<NT;i++) if(strcmp(tenName[i],t)==0) return tenSize[i];
  return 0;
}
static int producer_of(const char* t){
  for(int i=0;i<N;i++) for(int k=0;k<g[i].nout;k++) if(strcmp(g[i].outs[k],t)==0) return i;
  return -1;
}
static char* xstrdup(const char* s){ char* p=malloc(strlen(s)+1); strcpy(p,s); return p; }
static void parse_line(char* line){
  Node* n=&g[N]; memset(n,0,sizeof(*n)); char* save; char* tok;
  tok=strtok_r(line,"\t",&save); if(!tok) return; n->id=atoi(tok);
  tok=strtok_r(NULL,"\t",&save); strncpy(n->name,tok,MAXNAME-1);
  tok=strtok_r(NULL,"\t",&save); strncpy(n->type,tok,MAXNAME-1);
  strtok_r(NULL,"\t",&save);
  char* insb=strtok_r(NULL,"\t",&save); strtok_r(NULL,"\t",&save);
  char* outb=strtok_r(NULL,"\t\n",&save);
  if(insb){char* s2;char* t2=strtok_r(insb," ",&s2);while(t2){n->ins[n->nin++]=xstrdup(t2);t2=strtok_r(NULL," ",&s2);}}
  if(outb){char* s2;char* t2=strtok_r(outb," \n",&s2);while(t2){n->outs[n->nout++]=xstrdup(t2);t2=strtok_r(NULL," \n",&s2);}}
  N++;
}
static void load_sizes(const char* path){
  FILE* f=fopen(path,"r"); if(!f) return; char line[512];
  while(fgets(line,sizeof line,f)){ char* save;char* nm=strtok_r(line,"\t",&save);char* sz=strtok_r(NULL,"\t\n",&save);
    if(nm&&sz){strncpy(tenName[NT],nm,MAXNAME-1);tenSize[NT]=atol(sz);NT++;} }
  fclose(f);
}
static void build_dag(void){
  for(int i=0;i<N;i++) for(int k=0;k<g[i].nin;k++){int p=producer_of(g[i].ins[k]);
    if(p>=0&&p!=i){int dup=0;for(int d=0;d<g[i].ndep;d++)if(g[i].deps[d]==p)dup=1;
      if(!dup){g[i].deps[g[i].ndep++]=p;g[i].indeg++;}}}
}
// Cost-based schedule, identical to scheduler_cbs.c — see that file for the derivation.
static long cost_delta(int i){
  long d=0; for(int k=0;k<g[i].nout;k++) d+=size_of_tensor(g[i].outs[k]);
  for(int k=0;k<g[i].nin;k++){ int last=1;
    for(int j=0;j<N;j++){ if(j==i||g[j].scheduled)continue;
      for(int m=0;m<g[j].nin;m++) if(strcmp(g[j].ins[m],g[i].ins[k])==0)last=0; }
    if(last) d-=size_of_tensor(g[i].ins[k]); }
  return d;
}
static void schedule_cbs(int* rl,int* rn){
  int indeg[MAXN]; for(int i=0;i<N;i++){indeg[i]=g[i].indeg;g[i].scheduled=0;} *rn=0;
  for(;;){ int best=-1; long bc=0;
    for(int i=0;i<N;i++){ if(g[i].scheduled||indeg[i]!=0)continue; long c=cost_delta(i);
      if(best<0||c<bc||(c==bc&&g[i].id<g[best].id)){best=i;bc=c;} }
    if(best<0)break; g[best].scheduled=1; rl[(*rn)++]=best;
    for(int i=0;i<N;i++) for(int d=0;d<g[i].ndep;d++) if(g[i].deps[d]==best) indeg[i]--; }
}

// ── Allocation ──
// Alignment matches the hardware's rule: blocks of 2 KB or more align to 2 KB, smaller ones to
// 128 B. (The vendor's own allocator uses a finer 256 B alignment; see docs/05.)
#define ALIGN_LARGE 2048
#define ALIGN_SMALL 128
static long align_size(long n){
  long a = (n>=ALIGN_LARGE)?ALIGN_LARGE:ALIGN_SMALL;
  return (n + a-1) & ~(a-1);
}

typedef struct {
    char name[MAXNAME];
    long size;
    int birth, death;
    long offset;
    int spilled;
    int alias_of;   // in-place: index of the tensor whose address this one shares (-1 = own)
  } Tensor;
static Tensor T[MAXTEN]; static int nT=0;

// Which ops may write their output over their own input?
// Only ops where the read index equals the write index: y[i] = f(x[i]).
//   Conv / MatMul: no — one output reads many inputs, so overwriting the first output destroys
//                  data a later output still needs.
//   Transpose:     no — one-to-one, but read and write positions differ, so writing y[1] can
//                  clobber an x[1] that y[2] still needs.
static int is_destructive_op(const char* type){
  return strstr(type,"Neuron")    ||  // activations
         strstr(type,"Relu")      ||
         strstr(type,"Eltwise")   ||  // elementwise binary ops
         strstr(type,"Cast");         // type conversion, when the sizes match
}
static int find_tensor(const char* nm){                // tensor name -> index into T[]
  for(int i=0;i<nT;i++) if(strcmp(T[i].name,nm)==0) return i;
  return -1;
}

// Lifetimes, derived from the runlist: born when produced, dead after the last consumer runs.
static void compute_liveness(int* rl,int rn){
  int pos[MAXN]; for(int s=0;s<rn;s++) pos[rl[s]]=s;
  nT=0;
  for(int j=0;j<N;j++) for(int k=0;k<g[j].nout;k++){
    const char* tn=g[j].outs[k]; long sz=size_of_tensor(tn); if(sz==0) continue;
    int birth=pos[j], death=birth;
    for(int c=0;c<N;c++) for(int m=0;m<g[c].nin;m++)
      if(strcmp(g[c].ins[m],tn)==0 && pos[c]>death) death=pos[c];
    strncpy(T[nT].name,tn,MAXNAME-1); T[nT].size=align_size(sz);
    T[nT].birth=birth; T[nT].death=death; T[nT].offset=-1; T[nT].spilled=0;
    T[nT].alias_of=-1; nT++;
  }
}

// Link outputs onto inputs where destructive reuse is legal, before any offset is chosen.
// Condition 3 is the load-bearing one: if any op reads this input later, overwriting it now
// corrupts that read. death == pos[j] says "the last use is right now", which is exactly
// permission to destroy it.
static int link_inplace(int* rl,int rn){
  int pos[MAXN]; for(int s=0;s<rn;s++) pos[rl[s]]=s;
  int linked=0;
  for(int j=0;j<N;j++){
    if(!is_destructive_op(g[j].type)) continue;        // (1) elementwise ops only
    if(g[j].nout!=1) continue;                         // single output, for simplicity
    int oi=find_tensor(g[j].outs[0]);
    if(oi<0 || T[oi].alias_of>=0) continue;
    for(int k=0;k<g[j].nin;k++){
      int ii=find_tensor(g[j].ins[k]);
      if(ii<0) continue;                               // input of unknown size (external)
      if(T[ii].size != T[oi].size) continue;           // (2) sizes must match exactly
      if(T[ii].death != pos[j]) continue;              // (3) this op is the input's last consumer
      if(T[ii].alias_of>=0) continue;                  // already shared with someone else
      T[oi].alias_of = ii;
      linked++;
      break;                                           // reuse at most one input
    }
  }
  return linked;
}
// The entire safety condition for sharing an address.
static int overlaps(int a,int b){
  return !(T[a].death < T[b].birth || T[b].death < T[a].birth);
}

// First-fit offset assignment.
// Tensors are placed in birth order — walking time forward, so the already-placed tensors are
// exactly those that could conflict. For each tensor, the candidate offsets are 0 plus the top
// of every overlapping tensor already placed; the lowest candidate that collides with nothing
// wins. Taking the lowest is what makes reuse happen: when a tensor dies, the low address it
// occupied becomes the first candidate for whatever comes next. Anything past vtcm spills.
static void allocate(long vtcm, long* peak_out, long* spill_out){
  int order[MAXTEN]; for(int i=0;i<nT;i++) order[i]=i;
  for(int i=0;i<nT;i++) for(int j=i+1;j<nT;j++)
    if(T[order[j]].birth < T[order[i]].birth){ int t=order[i];order[i]=order[j];order[j]=t; }

  long peak=0, spill=0;
  for(int oi=0; oi<nT; oi++){
    int i=order[oi];
    // An in-place alias needs no search: it takes its source's offset.
    if(T[i].alias_of>=0 && T[T[i].alias_of].offset>=0){
      T[i].offset = T[T[i].alias_of].offset;
      long e=T[i].offset+T[i].size;
      if(e>vtcm){ T[i].spilled=1; spill += (e-(vtcm>T[i].offset?vtcm:T[i].offset)); }
      if(e>peak) peak=e;
      continue;
    }
    long cands[MAXTEN+1]; int nc=0; cands[nc++]=0;
    for(int oj=0; oj<nT; oj++){ int j=order[oj]; if(j==i||T[j].offset<0) continue;
      if(overlaps(i,j)) cands[nc++]=T[j].offset + T[j].size; }
    for(int a=0;a<nc;a++) for(int b=a+1;b<nc;b++) if(cands[b]<cands[a]){long t=cands[a];cands[a]=cands[b];cands[b]=t;}
    long chosen=-1;
    for(int ci=0; ci<nc; ci++){ long off=cands[ci]; int conflict=0;
      for(int oj=0; oj<nT; oj++){ int j=order[oj]; if(j==i||T[j].offset<0||!overlaps(i,j))continue;
        // do [off, off+size) and [T[j].offset, +size) intersect?
        if(!(off+T[i].size<=T[j].offset || T[j].offset+T[j].size<=off)){conflict=1;break;} }
      if(!conflict){ chosen=off; break; }
    }
    if(chosen<0) chosen=0;
    T[i].offset=chosen;
    long end=chosen+T[i].size;
    if(end>vtcm){ T[i].spilled=1; spill += (end - (vtcm>chosen?vtcm:chosen)); }
    if(end>peak) peak=end;
  }
  *peak_out=peak; *spill_out=spill;
}

int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: %s <graph> <sizes> [vtcm_bytes]\n",argv[0]); return 2; }
  FILE* f=fopen(argv[1],"r"); if(!f){perror("open");return 1;}
  char line[4096]; while(fgets(line,sizeof line,f)){if(strlen(line)>3)parse_line(line);} fclose(f);
  load_sizes(argv[2]);
  long vtcm = (argc>3)? atol(argv[3]) : (2L*1024*1024);

  build_dag();
  int rl[MAXN],rn; schedule_cbs(rl,&rn);
  compute_liveness(rl,rn);

  int nlinked = link_inplace(rl,rn);   // before allocation, matching the vendor's stage order

  long peak,spill; allocate(vtcm,&peak,&spill);

  printf("=== VTCM allocation (vtcm=%ld KB, %d live tensors, %d in-place links) ===\n",
         vtcm/1024, nT, nlinked);
  printf(" %-18s %8s %6s %6s %10s %-6s %s\n","tensor","size","birth","death","offset","spill","in-place");
  for(int i=0;i<nT;i++)
    printf(" %-18s %8ld %6d %6d %10ld %-6s %s\n",
      T[i].name,T[i].size,T[i].birth,T[i].death,T[i].offset,T[i].spilled?"SPILL":"",
      T[i].alias_of>=0 ? T[T[i].alias_of].name : "");
  printf("\n peak VTCM used = %ld B (%ld KB)\n", peak, peak/1024);
  printf(" spill_bytes    = %ld B (%ld KB)\n", spill, spill/1024);
  return 0;
}
