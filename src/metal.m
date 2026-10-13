#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { uint32_t smallStart,smallCount,largeStart,largeCount,steps,tileStart,tileCount,stepStart,schurStart,schurCount,narrowStart,narrowCount,narrowSchurStart,narrowSchurCount,maxRows,maxCols,maxChildren; } SpartaLevel;
// Live work at one panel step. Nodes and Schur tiles are prefixes of the level;
// strip tiles are a suffix, because they are ordered by tile column and a step
// only needs the columns it has not already factored.
typedef struct { uint32_t nodes,tileStart,tiles,schur; } SpartaStep;
// Host view of buffer 0, read only by sparta_v1_pivot_range. Must match
// struct Node in src/kernels/common.metal and analysis::Node field for field.
typedef struct { uint32_t offset,nr,nc,rows,children,count,entries,nentries,first,loffset,woffset,
    uoffset,soffset,sld; } SpartaNode;
_Static_assert(sizeof(SpartaNode)==14*sizeof(uint32_t),"SpartaNode must mirror struct Node");
// Buffers bound to every kernel. Buffer 20 is the LU U12 store and stays at the
// four-byte stub on the LDL^T path; buffer 14 is set per dispatch with setBytes.
#define SPARTA_BUFFERS 21
// Threadgroup width for factor_panel, the wide-front pivot kernel. Deliberately
// not PIVOT_THREADS: that constant still sizes factor_narrow, whose fronts are
// too thin to use a wide group. See the sweep in factor_panel. Override at run
// time with SPARTA_PANEL_THREADS.
#define SPARTA_PANEL_THREADS 512
// Threads per front in the two solve sweeps. 0 means match the width to each
// level's tallest front; SPARTA_SOLVE_THREADS pins one width for every level.
// Any pinned value must stay a multiple of the simdgroup width.
#define SPARTA_SOLVE_THREADS 0
// The two compiled solve widths. Must match SOLVE_THREADS and SOLVE_WIDE in
// src/kernels/common.metal, which are the strides those pipelines were built
// with; dispatching one at the other's width would stride past its own rows.
#define SPARTA_SOLVE_THREADS_NARROW 256
#define SPARTA_SOLVE_THREADS_WIDE 512
// Above this many fronts in a level, stay narrow: the level fills the device on
// front count alone, so idle lanes inside each group cost more than the extra
// width buys on its tallest front.
#define SPARTA_SOLVE_CROWD 8
// Pivot panels completed between two Schur drains. 1 reproduces the old
// cadence, where every panel drained its own rank-16 block; 0 means drain once
// per level, after the last panel. Supplied by the caller from
// Config::schur_batch, so the audit and the encoder cannot disagree.
//
// Pivot-panel width and Schur-drain width are two different choices that used
// to share one number. A pivot search reads the fully summed columns and their
// coupling rows, which update_strip still refreshes after every panel; it never
// reads the trailing-by-trailing block, so that block can absorb several panels
// before it is read back and written out. Draining it every panel re-reads and
// re-writes the whole Schur complement once per 16 columns; at batch B it is
// read and written once per 16*B columns, for the same arithmetic.
//
// Measured device factor time, median of four warm refactors, three processes
// per cadence round-robin so thermal drift hits each equally, on an M4 Pro:
//
//   batch          1      2      4      8     16      0
//   109.4k EMerge  0.156  0.137  0.132  0.130  0.129  0.129
//   curl-curl 24^3 0.698  0.604  0.574  0.571  0.565  0.562
//   curl-curl 30^3 2.531  2.133  1.994  1.961  1.955  1.951
//
// So 1.21x, 1.24x and 1.30x over the old one-drain-per-panel cadence, the gain
// growing with front size as expected: the traffic saved scales with the Schur
// block and the arithmetic does not change at all. The curve is flat past 8,
// and final-only is best or tied on all three, which is why it is the default
// -- it is also the fewest dispatches and the simplest schedule. A bounded
// cadence remains available for a case that needs shorter kernels.
//
// Accuracy improves slightly rather than degrading, which is the expected sign:
// update_schur accumulates the whole batch in registers and touches the
// destination once per drain instead of once per panel, so there are fewer
// intermediate roundings of a f32 Schur entry, not more. Error against the
// complex128 reference on the 109.4k EMerge system, stable to four digits over
// six runs: 6.424e-4 at final-only against 6.889e-4 at cadence 1.
#define SPARTA_SCHUR_BATCH 0
// Threadgroups zero_fronts aims to put in flight per level. A level near the
// root holds one or two fronts, so without slicing the clear runs on one or two
// cores; this is the target the host slices each front to reach.
#define SPARTA_ZERO_GROUPS 512
// The same target for assemble_children, and a ceiling on its slice count.
// Both matter: the target decides how wide a level of many small fronts goes,
// the ceiling how far a level of one enormous front is cut.
//
// Slicing finer than a front's work costs nothing -- the extra threadgroups
// find their stride empty and exit -- but it buys nothing either.
// Threadgroups factor_panel_tail aims for, and the ceiling on how finely one
// front's rows are cut. A level near the root holds one or two fronts, so
// without slicing the tail would inherit the head's occupancy.
#define SPARTA_TAIL_GROUPS 512
#define SPARTA_TAIL_SLICES 64
// Threads per factor_panel_tail group, and so the rows it stages at a time:
// this must equal TAIL_ROWS in the shader, which sizes the staging block by it.
// Narrower is not obviously worse here, because the block is threadgroup memory
// and a smaller one fits more groups on a core. 64 is two whole simdgroups.
#define SPARTA_TAIL_THREADS 64
#define SPARTA_CHILD_GROUPS 512
#define SPARTA_CHILD_SLICES 32
// Kernel classes for the opt-in profiler, in dispatch order within a level.
enum { P_FACTOR,P_ZERO,P_INIT,P_CHILD,P_NARROW,P_NARROW_SCHUR,P_NARROW_ADVANCE,P_PANEL,P_PANEL_TAIL,P_STRIP,P_SCHUR,
       P_ADVANCE,P_DRAIN,P_DRAIN_ADVANCE,P_STORE,P_FWD,P_DIAG,P_BWD,SPARTA_CLASSES };
static const char* const kSpartaClassNames[SPARTA_CLASSES]={
 "factor_nodes","zero_fronts","initialize_large","assemble_children","factor_narrow","update_schur/narrow","advance_panel/narrow",
 "factor_panel","factor_panel/tail","update_strip","update_schur/batch","advance_panel","update_schur/final",
 "advance_panel/final","store_factors","solve_forward","solve_diagonal","solve_backward"};
@interface SpartaContext : NSObject {
@public
 id<MTLDevice> device; id<MTLCommandQueue> queue;
 id<MTLComputePipelineState> factor, zeroFronts, assembleChildren, initializeLarge, panelTail, factorNarrow, factorPanel, updateStrip, updateSchur, advancePanel, storeFactors;
 id<MTLComputePipelineState> solveForward, solveDiagonal, solveBackward;
 id<MTLComputePipelineState> solveForwardWide, solveBackwardWide;
 // The LU counterparts. Same schedule, same buffers, same threadgroup shapes;
 // `run` picks one set at the top and the dispatch loop is shared.
 id<MTLComputePipelineState> factorLu, panelTailLu, initializeLargeLu, factorNarrowLu, factorPanelLu, updateStripLu, updateSchurLu, storeFactorsLu;
 id<MTLComputePipelineState> solveForwardLu, solveBackwardLu;
 id<MTLComputePipelineState> solveForwardLuWide, solveBackwardLuWide;
 id<MTLBuffer> buffers[SPARTA_BUFFERS];
 uint32_t nodes,n,schurBatch,panelThreads,solveThreads,pivotThreads,lu,rhsCapacity; size_t valuesBytes; NSData* levels; NSData* steps;
 // Opt-in profiling only; zero cost and untouched when SPARTA_PROFILE is unset.
 int profile; NSMutableArray<id<MTLCommandBuffer>>* profBuffers; NSMutableArray<NSNumber*>* profClass,* profSize;
 double profSeconds[SPARTA_CLASSES]; uint64_t profCalls[SPARTA_CLASSES],profGroups[SPARTA_CLASSES];
}
@end
@implementation SpartaContext
@end
static void message(char *out,size_t cap,NSString* text){if(cap)snprintf(out,cap,"%s",text.UTF8String);}
int sparta_v1_available(void){@autoreleasepool {return MTLCreateSystemDefaultDevice()!=nil;}}
void* sparta_v1_create(const void* const* data,const size_t* lengths,const uint32_t* level_data,uint32_t level_count,const uint32_t* step_data,uint32_t step_count,uint32_t nodes,uint32_t n,uint32_t lu,uint32_t schurBatch,uint32_t panelThreads,uint32_t solveThreads,double luDiagThreshold,const uint32_t* geometry,char* error,size_t cap){
 @autoreleasepool {@try {
  SpartaContext* c=[SpartaContext new]; c->device=MTLCreateSystemDefaultDevice();
  if(!c->device){message(error,cap,@"Metal device unavailable");return NULL;}
  c->queue=[c->device newCommandQueue];c->nodes=nodes;c->n=n;c->lu=lu;c->valuesBytes=lengths[6];
  // Runtime rather than compile-time, unlike PANEL and SCHUR_TILE: the cadence
  // changes no struct, no buffer size and no plan layout, only how often the
  // encoder emits update_schur, so one build can sweep it.
  // Supplied by the caller from the plan's Config, so dispatch_audit counts
  // exactly the schedule that runs. Reading an environment variable here as
  // well would let the two disagree silently.
  c->schurBatch=schurBatch;
  // Profiling puts every dispatch in its own command buffer so each one's
  // GPUStartTime..GPUEndTime is attributable. That costs submissions and
  // forfeits any overlap between adjacent dispatches, so the instrumented
  // total runs longer than the real one -- the report prints both, and the
  // per-class split is what it is for, not the absolute total.
  // 1 = per-class totals; 2 also dumps every dispatch, which is how you see
  // whether a class is slow because it is big or because it is starved of
  // threadgroups near the root of the tree.
  c->panelThreads=panelThreads?panelThreads:SPARTA_PANEL_THREADS;
  // factor_narrow is compiled with PIVOT_THREADS and must be dispatched at the
  // same width, or its threadgroup reduction reads lanes that never ran.
  c->pivotThreads=geometry[3];
  // Threads per front in the solve sweeps. A level is mostly small fronts, so
  // the useful direction is narrower, not wider: at 256 a 13-column front sums
  // its dot product across eight simdgroups, which is eight partials and two
  // threadgroup barriers per pivot to reduce thirteen values. One simdgroup
  // removes the cross-simd reduction entirely and puts eight times as many
  // fronts in flight per core. See the width sweep above solve_wide.
  c->rhsCapacity=1;
  c->solveThreads=solveThreads;
  {const char* v=getenv("SPARTA_PROFILE");c->profile=v&&*v&&*v!='0'?atoi(v):0;}
  NSString* source=
#include "shader.h"
  ;
  MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
  // Pin the language version. The default follows the *host process*'s
  // deployment target, so the same shader source compiles from a Rust binary
  // but not from an older Python interpreter, where atomic_float does not
  // exist yet. Metal 3.0 ships with macOS 13.
  options.languageVersion=MTLLanguageVersion3_0;
  // Experimental LU diagonal preference; zero preserves strict partial pivoting.
  double luThreshold=luDiagThreshold;
  if(!(luThreshold>=0&&luThreshold<=1)){
    message(error,cap,@"lu_diag_threshold must be between 0 and 1");return NULL;}
  // The frontal geometry the plan was built with. Passing it here rather than
  // repeating the numbers in the shader is what stops the two from drifting:
  // one value from sparta::Config reaches both sides.
  options.preprocessorMacros=@{
    @"SPARTA_LU_DIAG_THRESHOLD":@(luThreshold),
    @"PANEL":@(geometry[0]),
    @"SCHUR_BLOCK":@(geometry[1]),
    @"STRIP_BLOCK":@(geometry[2]),
    @"PIVOT_THREADS":@(geometry[3]),
    @"SCHUR_CHUNK":@(geometry[4]),
    @"TAIL_ROWS":@(SPARTA_TAIL_THREADS),
  };
  NSError* e=nil;id<MTLLibrary> library=[c->device newLibraryWithSource:source options:options error:&e];
  if(!library){message(error,cap,e.localizedDescription);return NULL;}
  c->factor=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"factor_nodes"] error:&e];
  c->zeroFronts=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"zero_fronts"] error:&e];
  c->assembleChildren=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"assemble_children"] error:&e];
  c->initializeLarge=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"initialize_large"] error:&e];
  c->factorNarrow=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"factor_narrow"] error:&e];
  c->factorPanel=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"factor_panel"] error:&e];
  c->panelTail=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"factor_panel_tail"] error:&e];
  c->updateStrip=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"update_strip"] error:&e];
  c->updateSchur=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"update_schur"] error:&e];
  c->advancePanel=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"advance_panel"] error:&e];
  c->storeFactors=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"store_factors"] error:&e];
  c->solveForward=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"solve_forward"] error:&e];
  c->solveDiagonal=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"solve_diagonal"] error:&e];
  c->solveBackward=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"solve_backward"] error:&e];
  c->solveForwardWide=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"solve_forward_wide"] error:&e];
  c->solveBackwardWide=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"solve_backward_wide"] error:&e];
  c->factorLu=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"factor_nodes_lu"] error:&e];
  c->initializeLargeLu=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"initialize_large_lu"] error:&e];
  c->factorNarrowLu=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"factor_narrow_lu"] error:&e];
  c->factorPanelLu=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"factor_panel_lu"] error:&e];
  c->panelTailLu=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"factor_panel_tail_lu"] error:&e];
  c->updateStripLu=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"update_strip_lu"] error:&e];
  c->updateSchurLu=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"update_schur_lu"] error:&e];
  c->storeFactorsLu=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"store_factors_lu"] error:&e];
  c->solveForwardLu=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"solve_forward_lu"] error:&e];
  c->solveBackwardLu=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"solve_backward_lu"] error:&e];
  c->solveForwardLuWide=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"solve_forward_lu_wide"] error:&e];
  c->solveBackwardLuWide=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"solve_backward_lu_wide"] error:&e];
  // One `e` is reused above, so by here it holds only the last error. Name the
  // pipeline that is missing instead: with the geometry configurable, a failure
  // here usually means one kernel's threadgroup memory or width is out of range,
  // and which kernel it is tells you which setting caused it.
  struct { const char* name; id<MTLComputePipelineState> pipeline; } built[]={
    {"factor_nodes",c->factor},{"zero_fronts",c->zeroFronts},
    {"assemble_children",c->assembleChildren},
    {"initialize_large",c->initializeLarge},
    {"factor_narrow",c->factorNarrow},{"factor_panel",c->factorPanel},
    {"factor_panel_tail",c->panelTail},
    {"update_strip",c->updateStrip},{"update_schur",c->updateSchur},
    {"advance_panel",c->advancePanel},{"store_factors",c->storeFactors},
    {"solve_forward",c->solveForward},{"solve_diagonal",c->solveDiagonal},
    {"solve_backward",c->solveBackward},{"factor_nodes_lu",c->factorLu},
    {"initialize_large_lu",c->initializeLargeLu},{"factor_narrow_lu",c->factorNarrowLu},
    {"factor_panel_lu",c->factorPanelLu},{"factor_panel_tail_lu",c->panelTailLu},{"update_strip_lu",c->updateStripLu},
    {"update_schur_lu",c->updateSchurLu},{"store_factors_lu",c->storeFactorsLu},
    {"solve_forward_lu",c->solveForwardLu},{"solve_backward_lu",c->solveBackwardLu},
    {"solve_forward_wide",c->solveForwardWide},{"solve_backward_wide",c->solveBackwardWide},
    {"solve_forward_lu_wide",c->solveForwardLuWide},{"solve_backward_lu_wide",c->solveBackwardLuWide},
  };
  if(!c->queue){message(error,cap,@"Metal command queue unavailable");return NULL;}
  // Under the profiler, report what each pipeline costs in occupancy terms.
  //
  // maxTotalThreadsPerThreadgroup is the register-pressure readout: the driver
  // lowers it when a kernel needs more registers per thread than a full group
  // would allow, so a change that trades registers for instructions shows up
  // here as a smaller number even when the kernel's own time improves. Reading
  // it alongside the per-kernel timings is what separates a real win from one
  // paid for out of occupancy somewhere else.
  if(c->profile){
   fprintf(stderr,"\nSPARTA_PROFILE  pipeline limits\n");
   fprintf(stderr,"  %-22s %10s %10s %12s\n","kernel","maxThreads","simdWidth","tgMemBytes");
   for(unsigned i=0;i<sizeof(built)/sizeof(*built);i++){
    if(!built[i].pipeline)continue;
    fprintf(stderr,"  %-22s %10lu %10lu %12lu\n",built[i].name,
            (unsigned long)built[i].pipeline.maxTotalThreadsPerThreadgroup,
            (unsigned long)built[i].pipeline.threadExecutionWidth,
            (unsigned long)built[i].pipeline.staticThreadgroupMemoryLength);
   }
  }
  for(unsigned i=0;i<sizeof(built)/sizeof(*built);i++)if(!built[i].pipeline){
    message(error,cap,[NSString stringWithFormat:@"Metal pipeline %s failed: %@",built[i].name,
      e.localizedDescription?:@"check the geometry settings against this device's limits"]);
    return NULL;}
  id<MTLComputePipelineState> wide[]={c->zeroFronts,c->assembleChildren,c->initializeLarge,c->factorNarrow,c->factorPanel,c->initializeLargeLu,c->factorNarrowLu,c->factorPanelLu};
  id<MTLComputePipelineState> full[]={c->panelTail,c->panelTailLu,c->updateStrip,c->updateSchur,c->storeFactors,c->solveForward,c->solveBackward,
                                      c->updateStripLu,c->updateSchurLu,c->storeFactorsLu,c->solveForwardLu,c->solveBackwardLu,
                                      c->solveForwardWide,c->solveBackwardWide,c->solveForwardLuWide,c->solveBackwardLuWide};
  for(unsigned i=0;i<sizeof(wide)/sizeof(*wide);i++)if(wide[i].maxTotalThreadsPerThreadgroup<128){message(error,cap,@"Required Metal threadgroup sizes unavailable");return NULL;}
  for(unsigned i=0;i<sizeof(full)/sizeof(*full);i++)if(full[i].maxTotalThreadsPerThreadgroup<256){message(error,cap,@"Required Metal threadgroup sizes unavailable");return NULL;}
  // Sizing is bounded by what this device actually reports, not a fixed budget.
  size_t total=0;
  for(unsigned i=0;i<SPARTA_BUFFERS;i++){
   size_t want=MAX(lengths[i],4);
   total+=want;
   if(want>c->device.maxBufferLength){
    message(error,cap,[NSString stringWithFormat:@"buffer %u needs %.2f GiB, device maximum is %.2f GiB",
     i,want/1073741824.0,c->device.maxBufferLength/1073741824.0]);return NULL;}
  }
  size_t budget=c->device.recommendedMaxWorkingSetSize;
  if(budget&&total>budget){
   message(error,cap,[NSString stringWithFormat:@"factorization needs %.2f GiB of GPU buffers, device working set is %.2f GiB",
    total/1073741824.0,budget/1073741824.0]);return NULL;}
  for(unsigned i=0;i<SPARTA_BUFFERS;i++){
   c->buffers[i]=[c->device newBufferWithLength:MAX(lengths[i],4) options:MTLResourceStorageModeShared];
   if(!c->buffers[i]){message(error,cap,[NSString stringWithFormat:@"allocating %.2f GiB for buffer %u failed",
    MAX(lengths[i],4)/1073741824.0,i]);return NULL;}
   if(data[i]&&lengths[i])memcpy(c->buffers[i].contents,data[i],lengths[i]);
  }
  c->levels=[NSData dataWithBytes:level_data length:level_count*sizeof(SpartaLevel)];
  c->steps=[NSData dataWithBytes:step_data length:step_count*sizeof(SpartaStep)];
  return (__bridge_retained void*)c;
 }@catch(NSException* e){message(error,cap,e.reason);return NULL;}}
}
void sparta_v1_destroy(void* context){if(context)CFBridgingRelease(context);}

// Smallest and largest pivot magnitude in the finished factors.
//
// Every buffer is StorageModeShared, so this is a host scan of n complex
// entries -- 8n bytes, microseconds -- with no kernel, no command buffer and
// no encode. The point is a quality signal the caller gets without paying for
// a matvec against the original matrix.
//
// A 2x2 pivot's own diagonal entries mean nothing on their own: pivoting to a
// 2x2 block is exactly what happens when both are too small to divide by, so
// scanning `diagonal` alone would report a false alarm on a healthy matrix.
// The block's scale is its determinant instead, taken relative to its largest
// entry so the result is comparable with a 1x1 pivot magnitude.
//
// On the LU path buffer 8 is solve scratch and the pivots are U's diagonal,
// which each front keeps inside its factor columns: pivot k of a node is
// factors[loffset + k*nr + k], found through the node descriptors in buffer 0.
// That is a scan of n entries spread over the factor store, still without a
// kernel.
//
// Returns 0 when there is nothing to report.
static int lu_pivot_range(SpartaContext* c,double* lo,double* hi){
  if(!c->buffers[0]||!c->buffers[17])return 0;
  if((size_t)c->nodes*sizeof(SpartaNode)>c->buffers[0].length)return 0;
  const SpartaNode* nodes=(const SpartaNode*)c->buffers[0].contents;
  const float* f=(const float*)c->buffers[17].contents;
  size_t elements=c->buffers[17].length/8;
  double small=INFINITY,large=0.0;
  for(uint32_t s=0;s<c->nodes;s++){
   SpartaNode node=nodes[s];
   for(uint32_t k=0;k<node.nc;k++){
    size_t at=(size_t)node.loffset+(size_t)k*node.nr+k;
    if(at>=elements)return 0;
    double m=hypot(f[2*at],f[2*at+1]);
    if(!isfinite(m))continue;
    if(m<small)small=m;
    if(m>large)large=m;
   }
  }
  if(!isfinite(small)||large<=0.0)return 0;
  *lo=small;*hi=large;
  return 1;
}
int sparta_v1_pivot_range(void* context,double* lo,double* hi){
 @autoreleasepool {@try {
  if(!context)return 0;
  SpartaContext* c=(__bridge SpartaContext*)context;
  if(c->lu)return lu_pivot_range(c,lo,hi);
  if(!c->buffers[8]||!c->buffers[9]||!c->buffers[11])return 0;
  uint32_t n=c->n;
  if((size_t)n*8>c->buffers[8].length||(size_t)n*8>c->buffers[9].length||(size_t)n*4>c->buffers[11].length)return 0;
  const float* d=(const float*)c->buffers[8].contents;
  const float* sub=(const float*)c->buffers[9].contents;
  const uint32_t* two=(const uint32_t*)c->buffers[11].contents;
  double small=INFINITY,large=0.0;
  for(uint32_t k=0;k<n;){
   double m;
   if(two[k]&&k+1<n){
    double ar=d[2*k],ai=d[2*k+1],br=sub[2*k],bi=sub[2*k+1],cr=d[2*k+2],ci=d[2*k+3];
    // det = a*c - b*b, complex.
    double dr=ar*cr-ai*ci-(br*br-bi*bi),di=ar*ci+ai*cr-2.0*br*bi;
    double top=fmax(hypot(ar,ai),fmax(hypot(br,bi),hypot(cr,ci)));
    m=top>0.0?hypot(dr,di)/top:0.0;
    k+=2;
   }else{
    m=hypot(d[2*k],d[2*k+1]);
    k++;
   }
   if(!isfinite(m))continue;
   if(m<small)small=m;
   if(m>large)large=m;
  }
  if(!isfinite(small)||large<=0.0)return 0;
  *lo=small;*hi=large;
  return 1;
 }@catch(NSException* e){return 0;}}
}
// Buffer bindings are identical for every kernel, so one encoder per command
// buffer binds them once. Compute encoders default to MTLDispatchTypeSerial,
// which already orders successive dispatches against each other.
static id<MTLComputeCommandEncoder> begin(SpartaContext* c,id<MTLCommandBuffer> command) {
 id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];if(!encoder)return nil;
 for(unsigned i=0;i<SPARTA_BUFFERS;i++)if(i!=14)[encoder setBuffer:c->buffers[i] offset:0 atIndex:i];
 return encoder;
}
// Threads per front for one solve level, from the rows of its largest front.
//
// Both sweeps stride over a front's rows, so lanes past the tallest front in the
// level have nothing to own, while a level of tall fronts is starved at a narrow
// width. The best fixed width differs between systems and follows the height
// of their fronts, which is what this reads.
//
// SPARTA_SOLVE_THREADS pins the width for every level.
// Whether one solve level should take the wide pipeline.
//
// Only two widths are compiled, so this is a choice between them rather than a
// fitted width. The wide one pays off when the fronts are tall enough to own its
// lanes and there are few enough of them that the level cannot fill the device on
// front count alone -- which is the top of the tree. A level of many short fronts
// takes the narrow one, where it was already fastest.
static bool solve_wide(SpartaContext* c,uint32_t rows,uint32_t fronts) {
 if(c->solveThreads)return c->solveThreads>SPARTA_SOLVE_THREADS_NARROW;
 return rows>SPARTA_SOLVE_THREADS_NARROW&&fronts<=SPARTA_SOLVE_CROWD;
}
static void encode(SpartaContext* c,id<MTLComputeCommandEncoder> encoder,id<MTLComputePipelineState> pipeline,
                   uint32_t start,uint32_t count,unsigned mode,uint32_t width,uint32_t nrhs,uint32_t extra) {
 if(!count)return;
 [encoder setComputePipelineState:pipeline];
 uint32_t range[2]={start,count};
 // The solve kernels take {start,count,n,nrhs}: they stride the solution vector
 // by n to reach their own right-hand side.
 uint32_t wide_range[4]={start,count,c->n,nrhs};
 // Mode 9 reuses the four-word range to carry its slice count in .z.
 uint32_t split_range[4]={start,count,width?width:1,extra};
 if(mode==3)[encoder setBytes:&c->nodes length:4 atIndex:14];
 else if(mode==7||mode==8)[encoder setBytes:wide_range length:16 atIndex:14];
 else if(mode==9||mode==10||mode==11)[encoder setBytes:split_range length:16 atIndex:14];
 else [encoder setBytes:range length:8 atIndex:14];
 if(mode==0)[encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(count,32),1,1)];
 else if(mode==1)[encoder dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(c->pivotThreads,1,1)];
 else if(mode==2)[encoder dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
 else if(mode==4)[encoder dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 // Mode 5: one threadgroup per item, as wide as this pipeline allows. For
 // kernels that are a flat strided sweep over a front, a wider group is
 // strictly better, and near the root of the tree there are too few fronts for
 // narrow groups to fill the device. The kernels read their own width from
 // [[threads_per_threadgroup]], so the strides follow whatever is dispatched.
 else if(mode==5)[encoder dispatchThreadgroups:MTLSizeMake(count,1,1)
   threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)1024,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
 // Mode 6: the wide pivot path. Wider than mode 1 because one front is often
 // alone in flight near the root, but capped well below mode 5 -- the pivot loop
 // is serial in k, so past a point extra lanes only cost occupancy.
 else if(mode==6)[encoder dispatchThreadgroups:MTLSizeMake(count,1,1)
   threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)(width?width:c->panelThreads),pipeline.maxTotalThreadsPerThreadgroup),1,1)];
 // Mode 7: one threadgroup per front in the solve sweeps. The width is the
 // caller's, already matched to the level, and the kernels read it back from
 // [[threads_per_threadgroup]] so their strides follow it.
 else if(mode==7)[encoder dispatchThreadgroups:MTLSizeMake(count,nrhs,1)
   threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)width,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
 // Mode 9: `width` threadgroups per item rather than one, each taking a strided
 // slice of the same front. For a pass that is a flat write over the whole
 // front with no ordering inside it, one threadgroup per front leaves a level
 // of few large fronts running on a handful of cores; this spreads it.
 // Mode 11: mode 10 at the fixed width factor_panel_tail stages for.
 else if(mode==11)[encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)count*(width?width:1),1,1)
   threadsPerThreadgroup:MTLSizeMake(SPARTA_TAIL_THREADS,1,1)];
 // Mode 10 is mode 9 with a child ordinal carried in .w.
 else if(mode==9||mode==10)[encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)count*(width?width:1),1,1)
   threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)1024,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
 // Mode 8: the block-diagonal solve, one thread per (front, right-hand side).
 else if(mode==8)[encoder dispatchThreads:MTLSizeMake(count,nrhs,1)
   threadsPerThreadgroup:MTLSizeMake(MIN(count,(uint32_t)32),1,1)];
 else [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
// One dispatch, tagged with its kernel class. Outside profiling this is exactly
// `encode`; under SPARTA_PROFILE it closes the encoder and command buffer around
// the dispatch so the class can be billed for its own GPU execution window.
#define STEP(pipe,start,count,mode,klass) STEPW(pipe,start,count,mode,klass,0)
#define STEPW(pipe,start,count,mode,klass,width) STEPX(pipe,start,count,mode,klass,width,0)
#define STEPX(pipe,start,count,mode,klass,width,extra) do{ \
 uint32_t _n=(count); \
 if(_n){ \
  encode(c,encoder,pipe,start,_n,mode,(width),nrhs,(extra)); \
  if(c->profile){ \
   /* Mode 9 dispatches `width` threadgroups per item, so bill it for what \
      it actually launched or the occupancy column understates it. */ \
   uint32_t _g=((mode)==9||(mode)==10||(mode)==11?_n*((width)?(width):1):_n); \
   c->profCalls[klass]++;c->profGroups[klass]+=_g; \
   [encoder endEncoding]; \
   [c->profBuffers addObject:command];[c->profClass addObject:@(klass)]; \
   [c->profSize addObject:@(_g)]; \
   [submitted addObject:command];[command commit]; \
   command=[c->queue commandBuffer]; \
   if(!command){message(error,cap,@"No profiling command buffer");return 0;} \
   encoder=begin(c,command); \
   if(!encoder){message(error,cap,@"No profiling compute encoder");return 0;}}}}while(0)

static void report_profile(SpartaContext* c,double total){
 if(c->profile>=2)fprintf(stderr,"\nSPARTA_PROFILE_DISPATCH  seq,kernel,groups,seconds\n");
 for(unsigned i=0;i<[c->profBuffers count];i++){
  id<MTLCommandBuffer> b=c->profBuffers[i];
  unsigned k=[c->profClass[i] unsignedIntValue];
  double dt=MAX(0,b.GPUEndTime-b.GPUStartTime);
  c->profSeconds[k]+=dt;
  if(c->profile>=2)fprintf(stderr,"D,%u,%s,%u,%.6f\n",i,kSpartaClassNames[k],
                           [c->profSize[i] unsignedIntValue],dt);
 }
 double sum=0;for(unsigned k=0;k<SPARTA_CLASSES;k++)sum+=c->profSeconds[k];
 fprintf(stderr,"\nSPARTA_PROFILE  instrumented device total %.4fs over %lu dispatches\n",
         total,(unsigned long)[c->profBuffers count]);
 fprintf(stderr,"  %-22s %10s %8s %12s %10s\n","kernel","seconds","share","threadgroups","dispatches");
 for(unsigned k=0;k<SPARTA_CLASSES;k++){
  if(!c->profCalls[k])continue;
  fprintf(stderr,"  %-22s %10.4f %7.1f%% %12llu %10llu\n",kSpartaClassNames[k],
          c->profSeconds[k],sum>0?100.0*c->profSeconds[k]/sum:0.0,
          (unsigned long long)c->profGroups[k],(unsigned long long)c->profCalls[k]);
 }
 fprintf(stderr,"  %-22s %10.4f\n","sum of classes",sum);
 [c->profBuffers removeAllObjects];[c->profClass removeAllObjects];[c->profSize removeAllObjects];
 for(unsigned k=0;k<SPARTA_CLASSES;k++){c->profSeconds[k]=0;c->profCalls[k]=0;c->profGroups[k]=0;}
}

int sparta_v1_run(void* context,int factoring,const void* input,void* output,uint32_t nrhs,uint32_t* status,uint64_t* perturbed,double* seconds,char* error,size_t cap){
 @autoreleasepool {
  // Retain every submitted command and join even on an encoding exception.
  // Rust may reuse/free buffers immediately after this function returns.
  NSMutableArray<id<MTLCommandBuffer>>* submitted=[NSMutableArray new];
  @try {
   SpartaContext* c=(__bridge SpartaContext*)context;
   // One schedule, two kernel sets. Everything about the dispatch below --
   // level order, panel steps, tile prefixes and suffixes, the drain cadence,
   // the 64-pass command split -- is a property of the plan, not of the
   // factorization, so the only thing the LU path changes is which pipelines
   // these names resolve to and that there is no block diagonal to solve.
   id<MTLComputePipelineState> kFactor=c->lu?c->factorLu:c->factor;
   id<MTLComputePipelineState> kInit=c->lu?c->initializeLargeLu:c->initializeLarge;
   id<MTLComputePipelineState> kNarrow=c->lu?c->factorNarrowLu:c->factorNarrow;
   id<MTLComputePipelineState> kPanel=c->lu?c->factorPanelLu:c->factorPanel;
   id<MTLComputePipelineState> kStrip=c->lu?c->updateStripLu:c->updateStrip;
   id<MTLComputePipelineState> kSchur=c->lu?c->updateSchurLu:c->updateSchur;
   id<MTLComputePipelineState> kStore=c->lu?c->storeFactorsLu:c->storeFactors;
   id<MTLComputePipelineState> kForward=c->lu?c->solveForwardLu:c->solveForward;
   id<MTLComputePipelineState> kBackward=c->lu?c->solveBackwardLu:c->solveBackward;
   id<MTLComputePipelineState> kForwardWide=c->lu?c->solveForwardLuWide:c->solveForwardWide;
   id<MTLComputePipelineState> kBackwardWide=c->lu?c->solveBackwardLuWide:c->solveBackwardWide;
   if(c->profile&&!c->profBuffers){c->profBuffers=[NSMutableArray new];c->profClass=[NSMutableArray new];c->profSize=[NSMutableArray new];}
   if(factoring)nrhs=1;
   else if(!nrhs){message(error,cap,@"at least one right-hand side required");return 0;}
   // The solution block, and on the LU path the pivot-order handover scratch,
   // hold one column each. Grow them the first time a wider block arrives and
   // keep the capacity: a sweep solves the same shape over and over.
   if(!factoring&&nrhs>c->rhsCapacity){
    size_t want=(size_t)c->n*nrhs*8;
    id<MTLBuffer> grown=[c->device newBufferWithLength:MAX(want,4) options:MTLResourceStorageModeShared];
    if(!grown){message(error,cap,[NSString stringWithFormat:@"allocating %.2f GiB for %u right-hand sides failed",want/1073741824.0,nrhs]);return 0;}
    c->buffers[13]=grown;
    if(c->lu){
     id<MTLBuffer> scratch=[c->device newBufferWithLength:MAX(want,4) options:MTLResourceStorageModeShared];
     if(!scratch){message(error,cap,@"allocating LU solve scratch failed");return 0;}
     c->buffers[8]=scratch;
    }
    c->rhsCapacity=nrhs;
   }
   memcpy(c->buffers[factoring?6:13].contents,input,factoring?c->valuesBytes:(size_t)c->n*nrhs*8);
   id<MTLCommandBuffer> command=[c->queue commandBuffer];if(!command){message(error,cap,@"No command buffer");return 0;}
   id<MTLComputeCommandEncoder> encoder=begin(c,command);
   if(!encoder){message(error,cap,@"No compute encoder");return 0;}
   if(factoring){
    const SpartaLevel* levels=c->levels.bytes;size_t count=c->levels.length/sizeof(SpartaLevel);
    const SpartaStep* steps=c->steps.bytes;
    unsigned passes=0;
    for(size_t l=0;l<count;l++){
     SpartaLevel level=levels[l];
     STEP(kFactor,level.smallStart,level.smallCount,0,P_FACTOR);
     // Narrow and wide cooperative fronts are adjacent in the id list, so one
     // assembly dispatch covers both.
     //
     // Clearing the fronts runs first and separately, spread over more
     // threadgroups than there are fronts. Near the root a level holds only a
     // few fronts and each is enormous, so at one threadgroup per front the
     // clear -- the largest write in assembly -- ran on a handful of cores.
     // Slices are chosen to reach roughly SPARTA_ZERO_GROUPS threadgroups
     // without exceeding the work a front actually has.
     {
      uint32_t fronts=level.narrowCount+level.largeCount;
      if(fronts){
       uint32_t splits=fronts>=SPARTA_ZERO_GROUPS?1:(SPARTA_ZERO_GROUPS+fronts-1)/fronts;
       uint32_t useful=(uint32_t)(((uint64_t)level.maxRows*level.maxRows+1023)/1024);
       if(!useful)useful=1;
       if(splits>useful)splits=useful;
       STEPW(c->zeroFronts,level.narrowStart,fronts,9,P_ZERO,splits);
      }
     }
     STEP(kInit,level.narrowStart,level.narrowCount+level.largeCount,5,P_INIT);
     // One dispatch per child ordinal: two children of one front accumulate
     // into overlapping regions, so they cannot run together, but every front's
     // c-th child can. A level of leaves has maxChildren 0 and costs nothing.
     {
      uint32_t fronts=level.narrowCount+level.largeCount;
      uint32_t splits=fronts>=SPARTA_CHILD_GROUPS?1:(SPARTA_CHILD_GROUPS+fronts-1)/fronts;
      if(splits>SPARTA_CHILD_SLICES)splits=SPARTA_CHILD_SLICES;
      for(uint32_t c2=0;c2<level.maxChildren;c2++)
       STEPX(c->assembleChildren,level.narrowStart,fronts,10,P_CHILD,splits,c2);
     }
     // Narrow fronts take every pivot in one dispatch, then drain once.
     STEP(kNarrow,level.narrowStart,level.narrowCount,1,P_NARROW);
     STEP(kSchur,level.narrowSchurStart,level.narrowSchurCount,2,P_NARROW_SCHUR);
     STEP(c->advancePanel,level.narrowStart,level.narrowCount,0,P_NARROW_ADVANCE);
     for(uint32_t q=0;q<level.steps;q++){
      // One step per panel. Fronts that have finished are no longer dispatched;
      // the live nodes, their fully summed tiles, and their Schur tiles are
      // all prefixes of this level.
      SpartaStep live=steps[level.stepStart+q];
      if(!live.nodes)break;
      // Factor a panel against the workspace, then bring the fully summed strip
      // current: that is what the next panel's pivot search reads. The Schur
      // complement is drained on its own, slower cadence below.
      // The LDL^T head sweeps only the fully summed block, so it is dispatched
      // to that width rather than the front height: rounded up to a simdgroup,
      // floored so a very narrow level still gets a usable group, and capped by
      // panel_threads. The LU panel still spans every row and keeps that cap.
      uint32_t headWidth=c->lu?c->panelThreads:MIN(c->panelThreads,
        MAX(64u,(level.maxCols+31u)/32u*32u));
      STEPW(kPanel,level.largeStart,live.nodes,6,P_PANEL,headWidth);
      // The panel's rows below the fully summed block, spread wide. They carry
      // no pivot dependency, so one threadgroup per front -- which is all the
      // serial head can use -- left most of the panel's arithmetic on a couple
      // of cores. Slices are chosen exactly as zero_fronts chooses its own.
      if(live.nodes){
       uint32_t splits=live.nodes>=SPARTA_TAIL_GROUPS?1:(SPARTA_TAIL_GROUPS+live.nodes-1)/live.nodes;
       if(splits>SPARTA_TAIL_SLICES)splits=SPARTA_TAIL_SLICES;
       STEPX(c->lu?c->panelTailLu:c->panelTail,level.largeStart,live.nodes,11,P_PANEL_TAIL,splits,0);
      }
      STEP(kStrip,level.tileStart+live.tileStart,live.tiles,2,P_STRIP);
      if(c->schurBatch&&(q+1)%c->schurBatch==0){
       // update_schur applies the whole pending range [state.panel, state.next)
       // as one product, so a batched drain is the same kernel over a wider
       // range, not a different algorithm. Only live fronts are dispatched here;
       // anything that finished earlier in this level is caught by the final
       // drain, which covers every tile.
       STEP(kSchur,level.schurStart,live.schur,2,P_SCHUR);
       STEP(c->advancePanel,level.largeStart,live.nodes,0,P_ADVANCE);
      }
      // Bound per-command work without CPU waits or pivot readback. The same
      // queue and tracked buffers order updates across command boundaries.
      if(++passes==64){
       [encoder endEncoding];[submitted addObject:command];[command commit];
       command=[c->queue commandBuffer];passes=0;
       if(!command){message(error,cap,@"No continuation command buffer");return 0;}
       encoder=begin(c,command);
       if(!encoder){message(error,cap,@"No continuation compute encoder");return 0;}}
     }
     // Mandatory final drain, whatever the cadence. The parent assembles this
     // front's Schur block in the next level and the workspace is recycled from
     // there on, so nothing may be left pending past this point. It runs over
     // every Schur tile of the level rather than a live prefix: a front that
     // took its last panel mid-batch is no longer in any later step's prefix.
     // Tiles with nothing owing cost one Node and one FrontState load, because
     // update_schur returns immediately on next<=panel.
     STEP(kSchur,level.schurStart,level.schurCount,2,P_DRAIN);
     STEP(c->advancePanel,level.largeStart,level.largeCount,0,P_DRAIN_ADVANCE);
     // Fronts are recycled from the next level on, so this level's factor
     // columns have to be copied out while they are still resident.
     // Sliced as the front clear is, and for the same reason: near the root a
     // level is one to three fronts, and one threadgroup each left the copy
     // on that many cores.
     {
      uint32_t fronts=level.smallCount+level.narrowCount+level.largeCount;
      if(fronts){
       uint32_t splits=fronts>=SPARTA_ZERO_GROUPS?1:(SPARTA_ZERO_GROUPS+fronts-1)/fronts;
       uint32_t useful=(uint32_t)(((uint64_t)level.maxRows*level.maxRows+1023)/1024);
       if(!useful)useful=1;
       if(splits>useful)splits=useful;
       STEPW(kStore,level.smallStart,fronts,9,P_STORE,splits);
      }
     }
    }
   }else{
    // Fronts within a level are independent, so each gets its own threadgroup.
    // Forward runs leaves first, backward runs roots first.
    const SpartaLevel* levels=c->levels.bytes;size_t count=c->levels.length/sizeof(SpartaLevel);
    // Each level takes the narrow or the wide compiled width, as solve_wide
    // decides; neither is the widest the pipeline allows. Unlike assembly, a
    // level here is mostly small fronts, and a wider group leaves most lanes
    // with no row to own. Measured solve time at a fixed 1024 against a fixed
    // 256, before the per-level choice: 0.016s vs 0.008s on a 125.0k EMerge
    // system, 0.021s vs 0.018s on an 88.8k one. Only the large-front curl-curl
    // grid preferred the wider group.
    for(size_t l=0;l<count;l++){
     uint32_t fronts=levels[l].smallCount+levels[l].narrowCount+levels[l].largeCount;
     bool wide=solve_wide(c,levels[l].maxRows,fronts);
     STEPW(wide?kForwardWide:kForward,levels[l].smallStart,fronts,7,P_FWD,
           wide?SPARTA_SOLVE_THREADS_WIDE:SPARTA_SOLVE_THREADS_NARROW);
    }
    // LU has no block diagonal: its pivots divide inside the backward sweep.
    if(!c->lu)STEP(c->solveDiagonal,0,c->nodes,8,P_DIAG);
    for(size_t l=count;l>0;l--){
     uint32_t fronts=levels[l-1].smallCount+levels[l-1].narrowCount+levels[l-1].largeCount;
     bool wide=solve_wide(c,levels[l-1].maxRows,fronts);
     STEPW(wide?kBackwardWide:kBackward,levels[l-1].smallStart,fronts,7,P_BWD,
           wide?SPARTA_SOLVE_THREADS_WIDE:SPARTA_SOLVE_THREADS_NARROW);
    }
   }
   [encoder endEncoding];
   [submitted addObject:command];[command commit];
   for(id<MTLCommandBuffer> job in submitted)[job waitUntilCompleted];
   *seconds=0;
   for(id<MTLCommandBuffer> job in submitted){
    if(job.status!=MTLCommandBufferStatusCompleted){message(error,cap,job.error.localizedDescription?:@"GPU execution failed");return 0;}
    *seconds+=MAX(0,job.GPUEndTime-job.GPUStartTime);
   }
   if(c->profile)report_profile(c,*seconds);
   if(factoring){
    memcpy(status,c->buffers[12].contents,c->nodes*4);
    // Total the per-front perturbation counts. FrontState is {next,start,width,
    // panel,perturbed} and must match analysis::FrontState.
    *perturbed=0;
    const uint32_t* st=c->buffers[15].contents;
    for(uint32_t i=0;i<c->nodes;i++)*perturbed+=st[i*5+4];
   }else memcpy(output,c->buffers[13].contents,(size_t)c->n*nrhs*8);
   return 1;
  }@catch(NSException* e){message(error,cap,e.reason);return 0;}
  @finally {for(id<MTLCommandBuffer> job in submitted)if(job.status>=MTLCommandBufferStatusCommitted)[job waitUntilCompleted];}
 }
}
