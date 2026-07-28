// scheduler_par.c — resource-aware parallel scheduling
//
// A single sequential runlist understates what the hardware can do, because the NPU has separate
// engines that run at the same time. The device profiler shows it directly: one convolution
// reported using both the matrix engine and DMA, and spent 43 % of its cycles overlapped with
// vector work.
//
// So instead of picking one best op per round, pick one per resource. The schedule becomes
// two-dimensional: time x resource, with a per-resource worker limit.
//
// Capacities come from the device kernel's own diagnostics:
//   "Invalid number of HMX thread config: %d. Maximum of 1 is allowed."       -> matrix: 1/core
//   "Invalid number of HVX thread config: %d. Must be less or equal to %d"    -> vector: N
//   "Started %d vec workers, %d matrix workers, %d eltwise workers"           -> pools per engine
//
// Consequence: convolution and matmul are inherently serial, while elementwise and normalisation
// ops genuinely run in parallel. See docs/04-scheduling.md and docs/07-hardware-resources.md.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 512
#define MAXNAME 128

typedef enum { R_HMX=0, R_HVX=1, R_DMA=2, R_NUM=3 } Resource;
static const char* RNAME[R_NUM] = {"HMX","HVX","DMA"};

#define MAXCAP 8
static int CAP[R_NUM] = { 1, 4, 1 };   // HMX=1 is a hardware limit; HVX is configurable

typedef struct {
  int id; char name[MAXNAME]; char type[MAXNAME];
  char *ins[16]; int nin; char *outs[8]; int nout;
  int indeg; int deps[MAXN]; int ndep; int scheduled;
  Resource res;   // which engine this op runs on
} Node;
static Node g[MAXN]; static int N=0;

// op type -> engine.
// NOTE: this uses substring matching, which is convenient for arbitrary graph files but is not
// strictly correct — depthwise convolution actually runs on the vector engine, so "Conv" as a
// substring would misfile it. minicc.c uses exact matching for that reason.
static Resource resource_of(const char* type){
  if(strstr(type,"Conv")||strstr(type,"MatMul")) return R_HMX;   // matrix engine
  if(strstr(type,"Transpose")||strstr(type,"Neuron")||           // vector engine
     strstr(type,"Eltwise")||strstr(type,"Relu")||strstr(type,"SlicePad")) return R_HVX;
  return R_DMA;  // InputSlice / flat_from_vtcm / Concat / graph I/O — data movement
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
  n->res = resource_of(n->type);               // tag the resource at parse time
  strtok_r(NULL,"\t",&save);
  char* insb=strtok_r(NULL,"\t",&save); strtok_r(NULL,"\t",&save);
  char* outb=strtok_r(NULL,"\t\n",&save);
  if(insb){char* s2;char* t2=strtok_r(insb," ",&s2);while(t2){n->ins[n->nin++]=xstrdup(t2);t2=strtok_r(NULL," ",&s2);}}
  if(outb){char* s2;char* t2=strtok_r(outb," \n",&s2);while(t2){n->outs[n->nout++]=xstrdup(t2);t2=strtok_r(NULL," \n",&s2);}}
  N++;
}
static void build_dag(void){
  for(int i=0;i<N;i++) for(int k=0;k<g[i].nin;k++){int p=producer_of(g[i].ins[k]);
    if(p>=0&&p!=i){int dup=0;for(int d=0;d<g[i].ndep;d++)if(g[i].deps[d]==p)dup=1;
      if(!dup){g[i].deps[g[i].ndep++]=p;g[i].indeg++;}}}
}

// slot[t][r][k] = the op running at time t on resource r, worker k  (-1 = idle)
static void schedule_parallel(int slot[][R_NUM][MAXCAP], int* nslots){
  int indeg[MAXN]; for(int i=0;i<N;i++){indeg[i]=g[i].indeg; g[i].scheduled=0;}
  int t=0;
  for(;; t++){
    int picked[R_NUM][MAXCAP]; int nfilled[R_NUM]={0,0,0};
    for(int r=0;r<R_NUM;r++) for(int k=0;k<MAXCAP;k++) picked[r][k]=-1;

    // Walk ready ops in id order, assigning each to a free worker on its own resource.
    // Capped at CAP[r]: with HMX=1, only one convolution can occupy any single slot.
    for(int i=0;i<N;i++){
      if(g[i].scheduled||indeg[i]!=0) continue;      // ready ops only
      Resource r=g[i].res;
      if(nfilled[r] >= CAP[r]) continue;             // engine full this slot; try the next one
      picked[r][nfilled[r]++]=i;
    }
    // Nothing ready anywhere -> done.
    int any=0; for(int r=0;r<R_NUM;r++) if(nfilled[r]>0) any=1;
    if(!any) break;
    for(int r=0;r<R_NUM;r++) for(int k=0;k<nfilled[r];k++){
      int b=picked[r][k]; slot[t][r][k]=b; g[b].scheduled=1;
    }
    // Release successors only after the whole slot has been committed.
    for(int r=0;r<R_NUM;r++) for(int k=0;k<nfilled[r];k++){
      int b=picked[r][k];
      for(int i=0;i<N;i++) for(int d=0;d<g[i].ndep;d++) if(g[i].deps[d]==b) indeg[i]--;
    }
  }
  *nslots=t;
}

int main(int argc,char**argv){
  if(argc<2){ fprintf(stderr,"usage: %s <graph.txt> [num_hvx]\n",argv[0]);
    fprintf(stderr,"  num_hvx: vector worker count (default 4). HMX is always 1 in hardware.\n");
    return 2; }
  FILE* f=fopen(argv[1],"r"); if(!f){perror("open");return 1;}
  char line[4096]; while(fgets(line,sizeof line,f)){if(strlen(line)>3)parse_line(line);} fclose(f);
  if(argc>2){ int nh=atoi(argv[2]); if(nh>=1&&nh<=MAXCAP) CAP[R_HVX]=nh; }
  build_dag();

  printf("=== resource capacities ===\n");
  printf("  HMX=%d (hardware limit: 1 per core)  HVX=%d  DMA=%d\n",
         CAP[R_HMX],CAP[R_HVX],CAP[R_DMA]);
  printf("\n=== per-node resource tagging ===\n");
  for(int i=0;i<N;i++) printf("  %-14s %-16s -> %s\n", g[i].name, g[i].type, RNAME[g[i].res]);

  static int slot[MAXN][R_NUM][MAXCAP];
  for(int t=0;t<MAXN;t++) for(int r=0;r<R_NUM;r++) for(int k=0;k<MAXCAP;k++) slot[t][r][k]=-1;
  int nslots; schedule_parallel(slot, &nslots);

  printf("\n=== parallel schedule (time x resource) ===\n");
  printf("  %-5s | %-16s | %-34s | %-12s\n","slot","HMX(1)","HVX(N)","DMA(1)");
  printf("  ------+------------------+------------------------------------+-------------\n");
  for(int t=0;t<nslots;t++){
    printf("  t%-4d |", t);
    for(int r=0;r<R_NUM;r++){
      char buf[256]=""; int first=1;
      for(int k=0;k<CAP[r];k++){
        int b=slot[t][r][k]; if(b<0) continue;
        if(!first) strncat(buf,",",sizeof buf-strlen(buf)-1);
        strncat(buf,g[b].name,sizeof buf-strlen(buf)-1); first=0;
      }
      if(first) strcpy(buf,"-");
      printf(" %-16s |", buf);
    }
    printf("\n");
  }
  // A slot is a logical parallel step, not a duration: op cycle counts are not modelled, so
  // this is not a latency prediction.
  printf("\n  sequential: %d slots (one op each)\n", N);
  printf("  parallel:   %d slots -> %.1f%% fewer\n", nslots, 100.0*(N-nslots)/N);
  return 0;
}
