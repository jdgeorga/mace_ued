#include <stdint.h>

// Minimal stub for Intel ITT/JIT profiling API.
// Provides symbols that some PyTorch/oneDNN builds reference at runtime.

int iJIT_NotifyEvent(int event_type, void *EventSpecificData) {
  (void)event_type;
  (void)EventSpecificData;
  return 0;
}

int iJIT_IsProfilingActive(void) { return 0; }

uint32_t iJIT_GetNewMethodID(void) { return 0; }

