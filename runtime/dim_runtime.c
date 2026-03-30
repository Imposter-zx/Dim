// dim_runtime.c — Dim Language Runtime Library
//
// This file provides the runtime support for Dim-compiled programs.
// It implements memory allocation, I/O operations, and other primitives.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// ============================================================================
// Memory Allocation
// ============================================================================

void* dim_alloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Dim: memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

void* dim_alloc_array(size_t count, size_t elem_size) {
    void* ptr = calloc(count, elem_size);
    if (!ptr && count > 0) {
        fprintf(stderr, "Dim: memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

void dim_dealloc(void* ptr) {
    free(ptr);
}

void* dim_realloc(void* ptr, size_t new_size) {
    void* new_ptr = realloc(ptr, new_size);
    if (!new_ptr && new_size > 0) {
        fprintf(stderr, "Dim: memory reallocation failed\n");
        exit(1);
    }
    return new_ptr;
}

// ============================================================================
// I/O Operations
// ============================================================================

void dim_runtime_print_i32(int32_t value) {
    printf("%d", value);
}

void dim_runtime_print_i64(int64_t value) {
    printf("%lld", (long long)value);
}

void dim_runtime_print_f32(float value) {
    printf("%f", value);
}

void dim_runtime_print_f64(double value) {
    printf("%lf", value);
}

void dim_runtime_print_bool(bool value) {
    printf("%s", value ? "true" : "false");
}

void dim_runtime_print_str(const char* str) {
    if (str) {
        printf("%s", str);
    }
}

void dim_runtime_println_str(const char* str) {
    if (str) {
        printf("%s\n", str);
    } else {
        printf("\n");
    }
}

void dim_runtime_panic(const char* message) {
    fprintf(stderr, "Dim: panic: %s\n", message ? message : "unknown error");
    exit(1);
}

// ============================================================================
// String Operations
// ============================================================================

int32_t dim_runtime_str_len(const char* str) {
    if (!str) return 0;
    return (int32_t)strlen(str);
}

char* dim_runtime_str_concat(const char* a, const char* b) {
    if (!a) a = "";
    if (!b) b = "";
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    char* result = dim_alloc(len_a + len_b + 1);
    memcpy(result, a, len_a);
    memcpy(result + len_a, b, len_b);
    result[len_a + len_b] = '\0';
    return result;
}

char* dim_runtime_str_substring(const char* str, int32_t start, int32_t len) {
    if (!str) return "";
    size_t str_len = strlen(str);
    if (start < 0) start = 0;
    if (start >= (int32_t)str_len) return "";
    if (len < 0) len = (int32_t)str_len - start;
    if (start + len > (int32_t)str_len) len = (int32_t)str_len - start;
    char* result = dim_alloc(len + 1);
    memcpy(result, str + start, len);
    result[len] = '\0';
    return result;
}

char* dim_runtime_str_to_upper(const char* str) {
    if (!str) return "";
    size_t len = strlen(str);
    char* result = dim_alloc(len + 1);
    for (size_t i = 0; i < len; i++) {
        char c = str[i];
        if (c >= 'a' && c <= 'z') {
            c = c - 'a' + 'A';
        }
        result[i] = c;
    }
    result[len] = '\0';
    return result;
}

char* dim_runtime_str_to_lower(const char* str) {
    if (!str) return "";
    size_t len = strlen(str);
    char* result = dim_alloc(len + 1);
    for (size_t i = 0; i < len; i++) {
        char c = str[i];
        if (c >= 'A' && c <= 'Z') {
            c = c - 'A' + 'a';
        }
        result[i] = c;
    }
    result[len] = '\0';
    return result;
}

char* dim_runtime_str_trim(const char* str) {
    if (!str) return "";
    const char* start = str;
    while (*start == ' ' || *start == '\t' || *start == '\n' || *start == '\r') {
        start++;
    }
    if (*start == '\0') return "";
    const char* end = str + strlen(str) - 1;
    while (end > start && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) {
        end--;
    }
    size_t len = end - start + 1;
    char* result = dim_alloc(len + 1);
    memcpy(result, start, len);
    result[len] = '\0';
    return result;
}

// ============================================================================
// Math Operations
// ============================================================================

int32_t dim_runtime_abs_i32(int32_t value) {
    return value < 0 ? -value : value;
}

int32_t dim_runtime_min_i32(int32_t a, int32_t b) {
    return a < b ? a : b;
}

int32_t dim_runtime_max_i32(int32_t a, int32_t b) {
    return a > b ? a : b;
}

// ============================================================================
// File I/O
// ============================================================================

char* dim_runtime_read_file(const char* path) {
    if (!path) {
        fprintf(stderr, "Dim: read_file: null path\n");
        return "";
    }
    FILE* f = fopen(path, "r");
    if (!f) {
        return "";
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = dim_alloc(len + 1);
    size_t read_len = fread(buffer, 1, len, f);
    buffer[read_len] = '\0';
    fclose(f);
    return buffer;
}

void dim_runtime_write_file(const char* path, const char* content) {
    if (!path) {
        fprintf(stderr, "Dim: write_file: null path\n");
        return;
    }
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Dim: write_file: could not open %s\n", path);
        return;
    }
    if (content) {
        fputs(content, f);
    }
    fclose(f);
}

bool dim_runtime_file_exists(const char* path) {
    if (!path) return false;
    FILE* f = fopen(path, "r");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

// ============================================================================
// Array Operations
// ============================================================================

int32_t dim_runtime_arr_len(void* arr) {
    if (!arr) return 0;
    return 0; // Would need length stored separately or different approach
}

void dim_runtime_arr_push(void** arr, int32_t* len, int32_t* cap, void* item, size_t elem_size) {
    if (!arr || !len || !cap) return;
    if (*len >= *cap) {
        int32_t new_cap = (*cap == 0) ? 4 : (*cap * 2);
        void* new_arr = dim_alloc_array(new_cap, elem_size);
        if (*arr && *len > 0) {
            memcpy(new_arr, *arr, (*len) * elem_size);
            dim_dealloc(*arr);
        }
        *arr = new_arr;
        *cap = new_cap;
    }
    memcpy((char*)(*arr) + (*len) * elem_size, item, elem_size);
    (*len)++;
}

void* dim_runtime_arr_pop(void** arr, int32_t* len, size_t elem_size) {
    if (!arr || !*arr || !len || *len <= 0) return NULL;
    (*len)--;
    void* item = dim_alloc(elem_size);
    memcpy(item, (char*)(*arr) + (*len) * elem_size, elem_size);
    return item;
}

// ============================================================================
// AI/ML Runtime
// ============================================================================

// Stub for AI prompt calls - would integrate with actual AI APIs
void* dim_runtime_prompt_call(void* prompt_ref, void* input, bool deterministic) {
    // In a real implementation, this would call an AI API
    // For now, return a stub result
    return NULL;
}

// ============================================================================
// Startup
// ============================================================================

// Main entry point wrapper
int dim_runtime_main(int (*user_main)(void), int argc, char** argv) {
    (void)argc;
    (void)argv;
    return user_main();
}
