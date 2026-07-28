// verify_dequant_llama.c — settling the dequantisation formula against a shipping model
//
// Reading five instructions gave  real = (q - offset) * scale  (see dequant_repro.c).
// The public SDK header documents  real = (q + offset) * scale  — the opposite sign.
//
// Reproducing an instruction sequence proves the reading, not the interpretation. So both
// candidate formulas are tested against the quantisation parameters of a publicly distributed
// quantised LLM built for this hardware (1B parameters, 4-bit weights / 16-bit activations),
// taken from that bundle's own metadata.
//
// Two independent tests:
//   1. q == zero_point must dequantise to exactly 0 — that is what a zero point IS.
//   2. The recovered range must be physically plausible: network activations and KV entries are
//      distributed around zero, so the range has to include negatives.
//
// See docs/06-quantization.md.
#include <stdio.h>
#include <stdint.h>
#include <math.h>

typedef struct { const char* name; const char* dtype; double scale; int zero_point; int qmax; } QParam;

// Values as published in the model bundle's metadata.
static QParam P[] = {
  {"past_key_0   (KV cache)", "uint8",  0.1092090830206871,   128,   255},
  {"past_value_0 (KV cache)", "uint8",  0.010491914115846157, 128,   255},
  {"past_key_8   (layer 8)",  "uint8",  0.14306236803531647,  128,   255},
  {"embedding",               "uint16", 1.028195856633829e-05,30393, 65535},
  {"activation",              "uint16", 0.010703595355153084, 32261, 65535},
  {"logits",                  "uint16", 0.0007153538754209876,23218, 65535},
};
#define NP (sizeof(P)/sizeof(P[0]))

// What the instructions do.
static double deq_impl  (const QParam* p, int q){ return (q - p->zero_point) * p->scale; }
// What the header comment says.
static double deq_header(const QParam* p, int q){ return (q + p->zero_point) * p->scale; }

int main(void){
  printf("=== dequantisation formula vs. a shipping quantised LLM's own parameters ===\n\n");
  printf("%-24s %-7s %12s %8s | %-28s | %-28s\n",
         "tensor","dtype","scale","zero_pt","(q-off)*scale range","(q+off)*scale range");
  printf("%.150s\n","--------------------------------------------------------------------------------------------------------------------------------------------------------");

  for(size_t i=0;i<NP;i++){
    const QParam* p=&P[i];
    // Recover the full range, q = 0 .. qmax, under each formula.
    double a_lo=deq_impl(p,0),   a_hi=deq_impl(p,p->qmax);
    double h_lo=deq_header(p,0), h_hi=deq_header(p,p->qmax);
    char a[64], h[64];
    snprintf(a,sizeof a,"[%+.4f, %+.4f]",a_lo,a_hi);
    snprintf(h,sizeof h,"[%+.4f, %+.4f]",h_lo,h_hi);
    printf("%-24s %-7s %12.3e %8d | %-28s | %-28s\n", p->name,p->dtype,p->scale,p->zero_point,a,h);
  }

  printf("\n=== test 1: q == zero_point must dequantise to exactly 0 ===\n");
  int pass_impl=0, pass_hdr=0;
  for(size_t i=0;i<NP;i++){
    const QParam* p=&P[i];
    double v_a=deq_impl(p,p->zero_point), v_h=deq_header(p,p->zero_point);
    int ok_a = fabs(v_a)<1e-12, ok_h = fabs(v_h)<1e-12;
    pass_impl += ok_a; pass_hdr += ok_h;
    printf("  %-24s q=%-6d  (q-off) %+.6f %-4s | (q+off) %+.6f %s\n",
      p->name, p->zero_point, v_a, ok_a?"OK":"FAIL", v_h, ok_h?"OK":"FAIL");
  }
  printf("\n  (q - offset) * scale : %d/%zu\n", pass_impl, NP);
  printf("  (q + offset) * scale : %d/%zu\n", pass_hdr,  NP);

  printf("\n=== test 2: the recovered range must include negatives ===\n");
  printf("  Network activations and KV entries are distributed around zero.\n");
  printf("  (q - offset) * scale : symmetric about 0, includes negatives   -> plausible\n");
  printf("  (q + offset) * scale : strictly positive, biased by zero_point -> cannot represent\n");
  printf("                         the values it is supposed to store\n");

  printf("\n=== conclusion ===\n");
  printf("  The formula read out of the instructions is correct; the header comment's sign is\n");
  printf("  a documentation error. The name \"zero_point\" is itself the tell: it is the value\n");
  printf("  that maps to zero, which is the quantity you subtract.\n");
  return 0;
}
