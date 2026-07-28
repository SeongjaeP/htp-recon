// scheduler.c — topological (list) scheduling
//
// A lowered graph states dependencies, not an order. Flattening it into a runlist is the
// scheduler's job, and several orders are always legal — they differ in peak memory, because
// the order is what determines how long each tensor stays live.
//
// The vendor's scheduler is cost-based; its own diagnostic strings name the pieces
// (dependency construction, depth grouping, candidate selection, greedy with a DP fallback).
// This file implements the base algorithm — Kahn's topological sort — and makes the tie-break
// policy explicit, since that is where the scheduler's character lives.
//
// Input: a .graph text file, one op per line:
//     id \t name \t type \t IN \t <input tensors...> \t OUT \t <output tensors...>
// Output: a runlist, plus a check that it respects every dependency.
//
// See docs/04-scheduling.md.
#define _POSIX_C_SOURCE 200809L   // for strtok_r; without it, -std=c11 leaves it undeclared
                                  // and the implicit-declaration call segfaults
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 512
#define MAXT 2048
#define MAXNAME 128

typedef struct {
  int id;
  char name[MAXNAME];
  char type[MAXNAME];
  char *ins[16];  int nin;   // input tensor names
  char *outs[8];  int nout;  // output tensor names
  int indeg;                 // unmet dependencies remaining; counts down during scheduling
  int deps[MAXN]; int ndep;  // node ids this node depends on
  int scheduled;
  int depth;                 // longest path from an entry node
} Node;

static Node g[MAXN];
static int N = 0;

// tensor name -> the node producing it, or -1 for a graph input / weight
static int producer_of(const char* t) {
  for (int i=0;i<N;i++)
    for (int k=0;k<g[i].nout;k++)
      if (strcmp(g[i].outs[k],t)==0) return i;
  return -1;
}

static char* xstrdup(const char* s){ char* p=malloc(strlen(s)+1); strcpy(p,s); return p; }

static void parse_line(char* line) {
  Node* n = &g[N];
  memset(n,0,sizeof(*n));
  char* save; char* tok;
  tok=strtok_r(line,"\t",&save); if(!tok) return; n->id=atoi(tok);
  tok=strtok_r(NULL,"\t",&save); strncpy(n->name,tok,MAXNAME-1);
  tok=strtok_r(NULL,"\t",&save); strncpy(n->type,tok,MAXNAME-1);
  strtok_r(NULL,"\t",&save);                  // the literal "IN"
  char* insblock=strtok_r(NULL,"\t",&save);   // space-separated inputs, up to the next tab
  char* outmark=strtok_r(NULL,"\t",&save);    // the literal "OUT"
  char* outsblock=strtok_r(NULL,"\t\n",&save);
  (void)outmark;
  if (insblock) { char* s2; char* t2=strtok_r(insblock," ",&s2);
    while(t2){ n->ins[n->nin++]=xstrdup(t2); t2=strtok_r(NULL," ",&s2);} }
  if (outsblock){ char* s2; char* t2=strtok_r(outsblock," \n",&s2);
    while(t2){ n->outs[n->nout++]=xstrdup(t2); t2=strtok_r(NULL," \n",&s2);} }
  N++;
}

// Whoever produces a tensor I consume is a node I depend on. indeg counts those dependencies,
// so a node with indeg == 0 has no producer inside the graph and is an entry point.
static void build_dag(void) {
  for (int i=0;i<N;i++) {
    for (int k=0;k<g[i].nin;k++) {
      int p = producer_of(g[i].ins[k]);
      if (p>=0 && p!=i) {
        int dup=0; for(int d=0;d<g[i].ndep;d++) if(g[i].deps[d]==p) dup=1;
        if(!dup){ g[i].deps[g[i].ndep++]=p; g[i].indeg++; }
      }
    }
  }
}

// depth(n) = 1 + max(depth of dependencies). Memoised on g[i].depth — without memoisation a
// diamond-shaped graph re-walks its shared ancestors exponentially.
static int compute_depth(int i){
  if (g[i].depth>0) return g[i].depth;
  int mx=0;
  for(int d=0;d<g[i].ndep;d++){ int dd=compute_depth(g[i].deps[d]); if(dd>mx)mx=dd; }
  return g[i].depth = mx+1;
}

// Kahn's algorithm: repeatedly take a node whose dependencies are all satisfied, then release
// its successors. Everything interesting is in the tie-break, i.e. which of several ready nodes
// to take:
//   0 = lowest id           source order, i.e. keep the IR's own ordering
//   1 = greatest depth      critical path first — pulls long chains forward, freeing their
//                           inputs sooner
//   2 = least depth         breadth first
//   3 = highest id          reverse source order
static void schedule(int policy, int* runlist, int* rn) {
  int indeg[MAXN]; for(int i=0;i<N;i++){indeg[i]=g[i].indeg; g[i].scheduled=0;}
  *rn=0;
  for(;;){
    // ready = not yet scheduled AND indeg == 0
    int best=-1;
    for(int i=0;i<N;i++){
      if(g[i].scheduled||indeg[i]!=0) continue;
      if(best<0){best=i;continue;}
      int take=0;
      if(policy==0) take = (g[i].id < g[best].id);
      else if(policy==1) take = (g[i].depth > g[best].depth) ||
                                (g[i].depth == g[best].depth && g[i].id < g[best].id);
      else if(policy==2) take = (g[i].depth < g[best].depth) ||
                                (g[i].depth == g[best].depth && g[i].id < g[best].id);
      else take = (g[i].id > g[best].id);
      if(take) best=i;
    }
    if(best<0) break;
    g[best].scheduled=1; runlist[(*rn)++]=best;
    // indeg acts as a countdown: it hits zero exactly when a node becomes runnable
    for(int i=0;i<N;i++)
      for(int d=0;d<g[i].ndep;d++)
        if(g[i].deps[d]==best) indeg[i]--;
  }
}

int main(int argc,char**argv){
  if(argc<2){ fprintf(stderr,"usage: %s <graph.txt> [policy 0|1|2|3]\n",argv[0]); return 2; }
  int policy = argc>2?atoi(argv[2]):0;
  FILE* f=fopen(argv[1],"r"); if(!f){perror("open");return 1;}
  char line[4096];
  while(fgets(line,sizeof line,f)){ if(strlen(line)>3) parse_line(line); }
  fclose(f);

  build_dag();
  for(int i=0;i<N;i++) compute_depth(i);

  printf("=== DAG (%d nodes) ===\n",N);
  for(int i=0;i<N;i++){
    printf(" [%d] %-12s %-16s depth=%d  deps=[",g[i].id,g[i].name,g[i].type,g[i].depth);
    for(int d=0;d<g[i].ndep;d++) printf("%s%s",d?",":"",g[g[i].deps[d]].name);
    printf("]\n");
  }

  int rl[MAXN],rn;
  const char* pn[]={"id-order (source order)","depth-desc (critical path)",
                    "depth-asc (breadth first)","id-reverse"};
  schedule(policy,rl,&rn);
  printf("\n=== RUNLIST (policy=%d: %s) ===\n",policy,pn[policy]);
  for(int i=0;i<rn;i++) printf(" %d. %s\n",i,g[rl[i]].name);

  // Validity check: every dependency must appear earlier in the runlist.
  int pos[MAXN]; for(int i=0;i<rn;i++) pos[rl[i]]=i;
  int ok=1;
  for(int i=0;i<N;i++) for(int d=0;d<g[i].ndep;d++)
    if(pos[g[i].deps[d]]>pos[i]){ ok=0; printf(" !! VIOLATION: %s before its dep %s\n",g[i].name,g[g[i].deps[d]].name);}
  printf("\ntopological order valid: %s (%d/%d nodes scheduled)\n", ok?"YES":"NO", rn, N);
  return 0;
}
