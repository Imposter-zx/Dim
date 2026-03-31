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
// String Operations (Extended)
// ============================================================================

bool dim_runtime_str_contains(const char* str, const char* sub) {
    if (!str || !sub) return false;
    return strstr(str, sub) != NULL;
}

bool dim_runtime_str_starts_with(const char* str, const char* prefix) {
    if (!str || !prefix) return false;
    size_t prefix_len = strlen(prefix);
    if (strlen(str) < prefix_len) return false;
    return strncmp(str, prefix, prefix_len) == 0;
}

bool dim_runtime_str_ends_with(const char* str, const char* suffix) {
    if (!str || !suffix) return false;
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    if (str_len < suffix_len) return false;
    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

int32_t dim_runtime_str_index_of(const char* str, const char* sub) {
    if (!str || !sub) return -1;
    const char* result = strstr(str, sub);
    if (!result) return -1;
    return (int32_t)(result - str);
}

char* dim_runtime_str_replace(const char* str, const char* old, const char* new_str) {
    if (!str || !old || !new_str) return "";
    
    size_t old_len = strlen(old);
    size_t new_len = strlen(new_str);
    size_t str_len = strlen(str);
    
    // Count occurrences
    int count = 0;
    const char* p = str;
    while ((p = strstr(p, old)) != NULL) {
        count++;
        p += old_len;
    }
    
    if (count == 0) {
        char* result = dim_alloc(str_len + 1);
        memcpy(result, str, str_len + 1);
        return result;
    }
    
    size_t result_len = str_len + count * ((int)new_len - (int)old_len);
    char* result = dim_alloc(result_len + 1);
    
    char* dest = result;
    const char* src = str;
    while (*src) {
        const char* match = strstr(src, old);
        if (!match) {
            strcpy(dest, src);
            break;
        }
        size_t copy_len = match - src;
        memcpy(dest, src, copy_len);
        dest += copy_len;
        memcpy(dest, new_str, new_len);
        dest += new_len;
        src = match + old_len;
    }
    result[result_len] = '\0';
    return result;
}

int32_t dim_runtime_str_split(const char* str, const char* delim, char*** output) {
    if (!str || !delim || !output) return 0;
    
    char* str_copy = dim_alloc(strlen(str) + 1);
    strcpy(str_copy, str);
    
    int count = 0;
    char* token = strtok(str_copy, delim);
    while (token) {
        count++;
        token = strtok(NULL, delim);
    }
    
    free(str_copy);
    
    str_copy = dim_alloc(strlen(str) + 1);
    strcpy(str_copy, str);
    
    *output = (char**)dim_alloc(count * sizeof(char*));
    
    int i = 0;
    token = strtok(str_copy, delim);
    while (token) {
        (*output)[i] = dim_alloc(strlen(token) + 1);
        strcpy((*output)[i], token);
        i++;
        token = strtok(NULL, delim);
    }
    
    free(str_copy);
    return count;
}

char* dim_runtime_str_join(char** arr, int32_t count, const char* sep) {
    if (!arr || count <= 0) return "";
    if (!sep) sep = "";
    
    size_t sep_len = strlen(sep);
    size_t total_len = 0;
    
    for (int i = 0; i < count; i++) {
        if (arr[i]) total_len += strlen(arr[i]);
    }
    total_len += (count - 1) * sep_len;
    
    char* result = dim_alloc(total_len + 1);
    char* dest = result;
    
    for (int i = 0; i < count; i++) {
        if (i > 0 && sep_len > 0) {
            memcpy(dest, sep, sep_len);
            dest += sep_len;
        }
        if (arr[i]) {
            size_t len = strlen(arr[i]);
            memcpy(dest, arr[i], len);
            dest += len;
        }
    }
    *dest = '\0';
    return result;
}

char* dim_runtime_str_repeat(const char* str, int32_t count) {
    if (!str || count <= 0) return "";
    size_t len = strlen(str);
    size_t result_len = len * count;
    char* result = dim_alloc(result_len + 1);
    for (int i = 0; i < count; i++) {
        memcpy(result + i * len, str, len);
    }
    result[result_len] = '\0';
    return result;
}

// ============================================================================
// Memory Management (Reference Counting)
// ============================================================================

typedef struct {
    void* data;
    int32_t ref_count;
    size_t size;
} DimRefCounted;

static DimRefCounted* dim_gc_rootset[1024];
static int32_t dim_gc_rootset_size = 0;

void* dim_alloc_ref(size_t size) {
    DimRefCounted* rc = (DimRefCounted*)malloc(sizeof(DimRefCounted));
    if (!rc) return NULL;
    rc->data = malloc(size);
    if (!rc->data) {
        free(rc);
        return NULL;
    }
    rc->ref_count = 1;
    rc->size = size;
    return rc->data;
}

void dim_inc_ref(void* ptr) {
    if (!ptr) return;
    DimRefCounted* rc = (DimRefCounted*)((char*)ptr - sizeof(DimRefCounted));
    rc->ref_count++;
}

void dim_dec_ref(void* ptr) {
    if (!ptr) return;
    DimRefCounted* rc = (DimRefCounted*)((char*)ptr - sizeof(DimRefCounted));
    rc->ref_count--;
    if (rc->ref_count <= 0) {
        free(rc->data);
        free(rc);
    }
}

void dim_gc_register_root(void* ptr) {
    if (dim_gc_rootset_size < 1024) {
        dim_gc_rootset[dim_gc_rootset_size++] = (DimRefCounted*)((char*)ptr - sizeof(DimRefCounted));
    }
}

void dim_gc_collect(void) {
    // Simple mark-and-sweep: marks roots, sweeps unmarked
    for (int i = 0; i < dim_gc_rootset_size; i++) {
        if (dim_gc_rootset[i]) {
            dim_gc_rootset[i]->ref_count = 1; // Reset for GC
        }
    }
    dim_gc_rootset_size = 0;
}

// ============================================================================
// Concurrency (Thread Pool)
// ============================================================================

#include <pthread.h>

typedef struct {
    void* (*func)(void*);
    void* arg;
    int32_t ready;
} DimTask;

static DimTask dim_task_queue[256];
static int32_t dim_task_queue_head = 0;
static int32_t dim_task_queue_tail = 0;
static pthread_mutex_t dim_task_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t dim_task_cond = PTHREAD_COND_INITIALIZER;
static int32_t dim_shutdown = 0;

void* dim_thread_worker(void* arg) {
    (void)arg;
    while (!dim_shutdown) {
        pthread_mutex_lock(&dim_task_mutex);
        while (dim_task_queue_head == dim_task_queue_tail && !dim_shutdown) {
            pthread_cond_wait(&dim_task_cond, &dim_task_mutex);
        }
        if (dim_shutdown) {
            pthread_mutex_unlock(&dim_task_mutex);
            break;
        }
        DimTask task = dim_task_queue[dim_task_queue_head % 256];
        dim_task_queue_head++;
        pthread_mutex_unlock(&dim_task_mutex);
        if (task.func) {
            task.func(task.arg);
        }
    }
    return NULL;
}

int32_t dim_thread_pool_init(int32_t num_threads) {
    static pthread_t threads[16];
    for (int i = 0; i < num_threads && i < 16; i++) {
        pthread_create(&threads[i], NULL, dim_thread_worker, NULL);
    }
    return num_threads;
}

void dim_thread_pool_submit(void* (*func)(void*), void* arg) {
    pthread_mutex_lock(&dim_task_mutex);
    dim_task_queue[dim_task_queue_tail % 256].func = func;
    dim_task_queue[dim_task_queue_tail % 256].arg = arg;
    dim_task_queue_tail++;
    pthread_cond_signal(&dim_task_cond);
    pthread_mutex_unlock(&dim_task_mutex);
}

void dim_thread_pool_shutdown(void) {
    dim_shutdown = 1;
    pthread_cond_broadcast(&dim_task_cond);
}

// ============================================================================
// Async/Await Support
// ============================================================================

typedef struct {
    void* result;
    int32_t ready;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} DimFuture;

DimFuture* dim_future_new(void) {
    DimFuture* f = (DimFuture*)malloc(sizeof(DimFuture));
    f->result = NULL;
    f->ready = 0;
    pthread_mutex_init(&f->mutex, NULL);
    pthread_cond_init(&f->cond, NULL);
    return f;
}

void dim_future_await(DimFuture* f) {
    pthread_mutex_lock(&f->mutex);
    while (!f->ready) {
        pthread_cond_wait(&f->cond, &f->mutex);
    }
    pthread_mutex_unlock(&f->mutex);
}

void dim_future_resolve(DimFuture* f, void* result) {
    pthread_mutex_lock(&f->mutex);
    f->result = result;
    f->ready = 1;
    pthread_cond_signal(&f->cond);
    pthread_mutex_unlock(&f->mutex);
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
