// scheduler_cbs.c — cost-based scheduling
//
// The vendor's scheduler is cost-based, and its diagnostic strings name the structure:
// dependency construction -> grouping by depth -> candidate selection by cost -> greedy with a
// dynamic-programming fallback, driven by a pluggable cost-function type.
//
// This file reproduces the greedy half with the most natural cost function for a machine whose
// binding constraint is on-chip memory: the change in live bytes.
//
//   cost(op) = sum of sizes of the op's outputs            (newly live)
//            - sum of sizes of inputs the op last consumes (freed)
//
// That is liveness-aware / Sethi-Ullman style scheduling. It is deliberately naive, and
// implementing it exposes why the real one needs the DP fallback — see docs/04-scheduling.md.
//
// Input: <graph.txt> (id\tname\ttype\tIN\t...\tOUT\t...) plus <sizes.txt> (tensor\tbytes)
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 512
#define MAXNAME 128
#define MAXTEN 4096

typedef struct {
  int id; char name[MAXNAME]; char type[MAXNAME];
  char *ins[16]; int nin;
  char *outs[8]; int nout;
  int indeg; int deps[MAXN]; int ndep;
  int scheduled; int depth;
} Node;
static Node g[MAXN]; static int N=0;

// tensor size table
static char tenName[MAXTEN][MAXNAME]; static long tenSize[MAXTEN]; static int NT=0;
static long size_of_tensor(const char* t){
  for(int i=0;i<NT;i++) if(strcmp(tenName[i],t)==0) return tenSize[i];
  return 0; // unknown size = external input or scalar; treated as free
}

static int producer_of(const char* t){
  for(int i=0;i<N;i++) for(int k=0;k<g[i].nout;k++) if(strcmp(g[i].outs[k],t)==0) return i;
  return -1;
}
static char* xstrdup(const char* s){ char* p=malloc(strlen(s)+1); strcpy(p,s); return p; }

static void parse_line(char* line){
  Node* n=&g[N]; memset(n,0,sizeof(*n));
  char* save; char* tok;
  tok=strtok_r(line,"\t",&save); if(!tok) return; n->id=atoi(tok);
  tok=strtok_r(NULL,"\t",&save); strncpy(n->name,tok,MAXNAME-1);
  tok=strtok_r(NULL,"\t",&save); strncpy(n->type,tok,MAXNAME-1);
  strtok_r(NULL,"\t",&save);
  char* insb=strtok_r(NULL,"\t",&save);
  strtok_r(NULL,"\t",&save); // "OUT"
  char* outb=strtok_r(NULL,"\t\n",&save);
  if(insb){ char* s2; char* t2=strtok_r(insb," ",&s2); while(t2){ n->ins[n->nin++]=xstrdup(t2); t2=strtok_r(NULL," ",&s2);} }
  if(outb){ char* s2; char* t2=strtok_r(outb," \n",&s2); while(t2){ n->outs[n->nout++]=xstrdup(t2); t2=strtok_r(NULL," \n",&s2);} }
  N++;
}
static void load_sizes(const char* path){
  FILE* f=fopen(path,"r"); if(!f) return;
  char line[512];
  while(fgets(line,sizeof line,f)){
    char* save; char* nm=strtok_r(line,"\t",&save); char* sz=strtok_r(NULL,"\t\n",&save);
    if(nm&&sz){ strncpy(tenName[NT],nm,MAXNAME-1); tenSize[NT]=atol(sz); NT++; }
  }
  fclose(f);
}
static void build_dag(void){
  for(int i=0;i<N;i++) for(int k=0;k<g[i].nin;k++){
    int p=producer_of(g[i].ins[k]);
    if(p>=0&&p!=i){ int dup=0; for(int d=0;d<g[i].ndep;d++) if(g[i].deps[d]==p)dup=1;
      if(!dup){ g[i].deps[g[i].ndep++]=p; g[i].indeg++; } }
  }
}
static int compute_depth(int i){
  if(g[i].depth>0) return g[i].depth;
  int mx=0; for(int d=0;d<g[i].ndep;d++){int dd=compute_depth(g[i].deps[d]); if(dd>mx)mx=dd;}
  return g[i].depth=mx+1;
}

// ── The cost function ──
// A tensor must stay in on-chip memory from the moment it is produced until its last consumer
// runs. Running op i makes i's outputs live, and kills any input for which i is the last
// consumer. So cost(i) is the net change in live bytes: lower is better.

// Is except_op the last consumer of tname? True when no other unscheduled node reads it.
static int last_consumer_scheduled_except(const char* tname, int except_op){
  for(int i=0;i<N;i++){
    if(i==except_op || g[i].scheduled) continue;
    for(int k=0;k<g[i].nin;k++) if(strcmp(g[i].ins[k],tname)==0) return 0;
  }
  return 1;
}
static long cost_delta(int i){
  long delta=0;
  for(int k=0;k<g[i].nout;k++) delta += size_of_tensor(g[i].outs[k]);       // becomes live
  for(int k=0;k<g[i].nin;k++)
    if(last_consumer_scheduled_except(g[i].ins[k], i))
      delta -= size_of_tensor(g[i].ins[k]);                                 // becomes dead
  return delta;
}

// Kahn's algorithm with the cost function as the tie-break: among ready ops, take the cheapest.
static void schedule_cbs(int* runlist,int* rn){
  int indeg[MAXN]; for(int i=0;i<N;i++){indeg[i]=g[i].indeg; g[i].scheduled=0;}
  *rn=0;
  for(;;){
    int best=-1; long bestCost=0;
    for(int i=0;i<N;i++){
      if(g[i].scheduled||indeg[i]!=0) continue;          // ready ops only
      long c=cost_delta(i);
      if(best<0 || c<bestCost || (c==bestCost && g[i].id<g[best].id)){
        best=i; bestCost=c;                              // greedy: minimum cost
      }
    }
    if(best<0) break;
    g[best].scheduled=1; runlist[(*rn)++]=best;
    for(int i=0;i<N;i++) for(int d=0;d<g[i].ndep;d++) if(g[i].deps[d]==best) indeg[i]--;
  }
}
// Baseline for comparison: critical-path-first, the depth policy from scheduler.c.
static void schedule_depth(int* runlist,int* rn){
  int indeg[MAXN]; for(int i=0;i<N;i++){indeg[i]=g[i].indeg; g[i].scheduled=0;}
  *rn=0;
  for(;;){
    int best=-1;
    for(int i=0;i<N;i++){
      if(g[i].scheduled||indeg[i]!=0) continue;
      if(best<0||g[i].depth>g[best].depth||(g[i].depth==g[best].depth&&g[i].id<g[best].id)) best=i;
    }
    if(best<0)break;
    g[best].scheduled=1; runlist[(*rn)++]=best;
    for(int i=0;i<N;i++) for(int d=0;d<g[i].ndep;d++) if(g[i].deps[d]==best) indeg[i]--;
  }
}

// Quality metric: the peak live footprint of a runlist.
// Given the order, a tensor is live over [produced, last consumed], so at each step sum the
// tensors whose interval covers it and take the maximum.
static long peak_live(int* rl,int rn){
  int pos[MAXN]; for(int s=0;s<rn;s++) pos[rl[s]]=s;   // node -> step
  long peak=0;
  for(int s=0;s<rn;s++){
    long live=0;
    for(int j=0;j<N;j++) for(int k=0;k<g[j].nout;k++){
      const char* tn=g[j].outs[k]; long sz=size_of_tensor(tn);
      if(sz==0) continue;
      int prod=pos[j];
      if(prod>s) continue;                              // not produced yet
      int lastUse=prod;
      for(int c=0;c<N;c++) for(int m=0;m<g[c].nin;m++)
        if(strcmp(g[c].ins[m],tn)==0 && pos[c]>lastUse) lastUse=pos[c];
      if(s<=lastUse) live+=sz;                          // live over [prod, lastUse]
    }
    if(live>peak) peak=live;
  }
  return peak;
}

int main(int argc,char**argv){
  if(argc<2){ fprintf(stderr,"usage: %s <graph.txt> [sizes.txt]\n",argv[0]); return 2; }
  FILE* f=fopen(argv[1],"r"); if(!f){perror("open");return 1;}
  char line[4096]; while(fgets(line,sizeof line,f)){ if(strlen(line)>3) parse_line(line);} fclose(f);
  if(argc>2) load_sizes(argv[2]);

  build_dag(); for(int i=0;i<N;i++) compute_depth(i);

  printf("=== DAG (%d nodes, %d tensor sizes) ===\n",N,NT);
  for(int i=0;i<N;i++){
    long outb=0; for(int k=0;k<g[i].nout;k++) outb+=size_of_tensor(g[i].outs[k]);
    printf(" [%d] %-12s %-16s depth=%d out=%ldB\n",g[i].id,g[i].name,g[i].type,g[i].depth,outb);
  }

  int rl1[MAXN],rn1, rl2[MAXN],rn2;
  schedule_depth(rl1,&rn1);
  schedule_cbs(rl2,&rn2);

  printf("\n=== depth-desc vs cost-based ===\n");
  printf(" %-3s %-14s %-14s\n","#","depth","mem-cost");
  for(int i=0;i<rn1;i++)
    printf(" %-3d %-14s %-14s\n",i,g[rl1[i]].name,g[rl2[i]].name);

  printf("\n peak live memory:  depth=%ldB   cost-based=%ldB\n",
         peak_live(rl1,rn1), peak_live(rl2,rn2));

  int pos[MAXN]; for(int i=0;i<rn2;i++) pos[rl2[i]]=i; int ok=1;
  for(int i=0;i<N;i++) for(int d=0;d<g[i].ndep;d++) if(pos[g[i].deps[d]]>pos[i]) ok=0;
  printf(" cost-based order valid: %s (%d/%d)\n", ok?"YES":"NO", rn2, N);
  return 0;
}
