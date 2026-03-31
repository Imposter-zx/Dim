// dim_runtime_wasm.c — Dim Language WebAssembly Runtime
//
// This file provides the runtime support for Dim-compiled WASM programs.
// For WASM, we use JavaScript-provided linear memory and I/O functions.

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

// ============================================================================
// Memory - Uses WASM linear memory (imported from JS)
// ============================================================================

// External declarations for memory imported from JavaScript
// In WASM, these are typically imported from the host environment
extern void* dim_wasm_memory_base;
extern uint32_t dim_wasm_memory_size;

// Simple bump allocator for WASM linear memory
static uint32_t heap_ptr = 4096;  // Start after static data

void* dim_alloc(uint32_t size) {
    if (size == 0) return NULL;
    void* ptr = (void*)(uintptr_t)heap_ptr;
    heap_ptr += size;
    // Check for overflow (simple check)
    if (heap_ptr > dim_wasm_memory_size) {
        return NULL;
    }
    return ptr;
}

void* dim_alloc_array(uint32_t count, uint32_t elem_size) {
    if (count == 0 || elem_size == 0) return NULL;
    return dim_alloc(count * elem_size);
}

void dim_dealloc(void* ptr) {
    // In WASM with linear memory, we typically don't actually deallocate
    // The collector would handle this in a full implementation
    (void)ptr;
}

// ============================================================================
// I/O Operations (imported from JavaScript)
// ============================================================================

// These would be imported from the JavaScript environment
// __import_dim_print_i32, __import_dim_print_str, etc.

void dim_runtime_print_i32(int32_t value);
void dim_runtime_print_i64(int64_t value);
void dim_runtime_print_f32(float value);
void dim_runtime_print_f64(double value);
void dim_runtime_print_bool(int32_t value);
void dim_runtime_print_str(int32_t ptr);
void dim_runtime_println_str(int32_t ptr);
void dim_runtime_panic(int32_t ptr);

// ============================================================================
// String Operations (using WASM linear memory)
// ============================================================================

int32_t dim_runtime_str_len(int32_t ptr) {
    if (ptr <= 0) return 0;
    const char* str = (const char*)(uintptr_t)ptr;
    return (int32_t)strlen(str);
}

int32_t dim_runtime_str_concat(int32_t a, int32_t b) {
    if (a <= 0 && b <= 0) return 0;
    const char* str_a = (a > 0) ? (const char*)(uintptr_t)a : "";
    const char* str_b = (b > 0) ? (const char*)(uintptr_t)b : "";
    
    size_t len_a = strlen(str_a);
    size_t len_b = strlen(str_b);
    uint32_t total = (uint32_t)(len_a + len_b + 1);
    
    int32_t result_ptr = (int32_t)(uintptr_t)dim_alloc(total);
    if (result_ptr == 0) return 0;
    
    char* result = (char*)(uintptr_t)result_ptr;
    memcpy(result, str_a, len_a);
    memcpy(result + len_a, str_b, len_b);
    result[len_a + len_b] = '\0';
    return result_ptr;
}

int32_t dim_runtime_str_substring(int32_t ptr, int32_t start, int32_t len) {
    if (ptr <= 0) return 0;
    const char* str = (const char*)(uintptr_t)ptr;
    size_t str_len = strlen(str);
    
    if (start < 0) start = 0;
    if (start >= (int32_t)str_len) return 0;
    
    if (len < 0) len = (int32_t)(str_len - start);
    if (start + len > (int32_t)str_len) len = (int32_t)(str_len - start);
    
    int32_t result_ptr = (int32_t)(uintptr_t)dim_alloc(len + 1);
    if (result_ptr == 0) return 0;
    
    char* result = (char*)(uintptr_t)result_ptr;
    memcpy(result, str + start, len);
    result[len] = '\0';
    return result_ptr;
}

int32_t dim_runtime_str_to_upper(int32_t ptr) {
    if (ptr <= 0) return 0;
    const char* str = (const char*)(uintptr_t)ptr;
    size_t len = strlen(str);
    
    int32_t result_ptr = (int32_t)(uintptr_t)dim_alloc(len + 1);
    if (result_ptr == 0) return 0;
    
    char* result = (char*)(uintptr_t)result_ptr;
    for (size_t i = 0; i < len; i++) {
        char c = str[i];
        if (c >= 'a' && c <= 'z') {
            result[i] = c - 'a' + 'A';
        } else {
            result[i] = c;
        }
    }
    result[len] = '\0';
    return result_ptr;
}

int32_t dim_runtime_str_to_lower(int32_t ptr) {
    if (ptr <= 0) return 0;
    const char* str = (const char*)(uintptr_t)ptr;
    size_t len = strlen(str);
    
    int32_t result_ptr = (int32_t)(uintptr_t)dim_alloc(len + 1);
    if (result_ptr == 0) return 0;
    
    char* result = (char*)(uintptr_t)result_ptr;
    for (size_t i = 0; i < len; i++) {
        char c = str[i];
        if (c >= 'A' && c <= 'Z') {
            result[i] = c - 'A' + 'a';
        } else {
            result[i] = c;
        }
    }
    result[len] = '\0';
    return result_ptr;
}

int32_t dim_runtime_str_trim(int32_t ptr) {
    if (ptr <= 0) return 0;
    const char* str = (const char*)(uintptr_t)ptr;
    
    const char* start = str;
    while (*start == ' ' || *start == '\t' || *start == '\n' || *start == '\r') {
        start++;
    }
    if (*start == '\0') return 0;
    
    const char* end = str + strlen(str) - 1;
    while (end > start && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) {
        end--;
    }
    
    size_t len = end - start + 1;
    int32_t result_ptr = (int32_t)(uintptr_t)dim_alloc(len + 1);
    if (result_ptr == 0) return 0;
    
    char* result = (char*)(uintptr_t)result_ptr;
    memcpy(result, start, len);
    result[len] = '\0';
    return result_ptr;
}

// ============================================================================
// File I/O (would use JavaScript fetch API)
// ============================================================================

int32_t dim_runtime_read_file(int32_t path_ptr);
void dim_runtime_write_file(int32_t path_ptr, int32_t content_ptr);
int32_t dim_runtime_file_exists(int32_t path_ptr);

// ============================================================================
// Math Operations
// ============================================================================

int32_t dim_runtime_abs_i32(int32_t value) {
    return value < 0 ? -value : value;
}

int64_t dim_runtime_abs_i64(int64_t value) {
    return value < 0 ? -value : value;
}

float dim_runtime_abs_f32(float value) {
    return value < 0 ? -value : value;
}

double dim_runtime_abs_f64(double value) {
    return value < 0 ? -value : value;
}

int32_t dim_runtime_min_i32(int32_t a, int32_t b) {
    return a < b ? a : b;
}

int32_t dim_runtime_max_i32(int32_t a, int32_t b) {
    return a > b ? a : b;
}

float dim_runtime_min_f32(float a, float b) {
    return a < b ? a : b;
}

float dim_runtime_max_f32(float a, float b) {
    return a > b ? a : b;
}

// ============================================================================
// AI/ML Runtime (stub for WASM)
// ============================================================================

void* dim_runtime_prompt_call(void* prompt_ref, void* input, int32_t deterministic) {
    (void)prompt_ref;
    (void)input;
    (void)deterministic;
    return NULL;
}