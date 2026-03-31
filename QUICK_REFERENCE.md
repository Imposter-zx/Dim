# Dim Language Quick Reference

A quick reference guide for the Dim programming language.

---

## Comments

```dim
# Single line comment
```

---

## Variables

```dim
let x = 42          # Immutable binding
let mut y = 10      # Mutable binding
y = y + 1           # Reassign mutable
```

---

## Types

### Primitive Types

```dim
let a: i32 = 42           # 32-bit integer
let b: f64 = 3.14         # 64-bit float
let c: bool = true       # Boolean
let s: str = "hello"     # String
let u: unit = none       # Unit type (void)
let n: i64 = 123456789   # 64-bit integer
```

### Collections

```dim
let arr: [i32] = [1, 2, 3]    # Dynamic array
let mat: [[f64]] = [[1, 2], [3, 4]]  # 2D array
```

---

## Functions

```dim
fn add(x: i32, y: i32) -> i32:
    return x + y

fn greet(msg: str):
    println(msg)

fn main() -> unit:
    let result = add(1, 2)
    return none
```

### Async Functions

```dim
async fn fetch(url: str) -> str:
    let data = await http_get(url)
    return data
```

### Generic Functions

```dim
fn identity[T](x: T) -> T:
    return x

fn max[T](a: T, b: T) -> T:
    if a > b { return a }
    return b
```

---

## Control Flow

### If/Else

```dim
if x > 0:
    return x
elif x < 0:
    return -x
else:
    return 0
```

### While Loop

```dim
let mut i = 0
while i < 10:
    i = i + 1
```

### For Loop

```dim
for item in items:
    println(item)
```

### Match

```dim
match value:
    0: return "zero"
    n if n > 0: return "positive"
    _: return "negative"
```

---

## Structures

```dim
struct Point:
    x: i32
    y: i32

fn create_point(x: i32, y: i32) -> Point:
    return Point{x: x, y: y}

# Usage
let p = Point{x: 10, y: 20}
let px = p.x
```

---

## Enums

```dim
enum Color:
    Red
    Green
    Blue

enum Result[T, E]:
    Ok(value: T)
    Err(error: E)

# Usage
let ok_result = Result.Ok(42)
let err_result = Result.Err("error")
```

---

## Traits

```dim
trait Printable:
    fn format(self) -> str

trait Comparable[T]:
    fn compare(self, other: T) -> i32

struct Point:
    x: f64
    y: f64

impl Printable for Point:
    fn format(self) -> str:
        return "(" + self.x.to_string() + ", " + self.y.to_string() + ")"

impl Comparable[Point] for Point:
    fn compare(self, other: Point) -> i32:
        dx = self.x - other.x
        dy = self.y - other.y
        dist = (dx * dx + dy * dy).sqrt()
        if dist < 1.0 { return -1 }
        if dist > 1.0 { return 1 }
        return 0
```

---

## Borrowing

```dim
fn process(data: Buffer):
    let view = &data       # Immutable borrow
    let ptr = &mut data    # Mutable borrow

fn modify(s: &mut String):
    s = "modified"
```

---

## Closures

```dim
let add = |x: i32, y: i32| -> i32: x + y
let result = add(1, 2)

let nums = [1, 2, 3, 4]
let evens = nums.filter(|n| n % 2 == 0)
```

---

## Error Handling

```dim
fn divide(a: i32, b: i32) -> Result[i32, str]:
    if b == 0:
        return Result.Err("division by zero")
    return Result.Ok(a / b)

# Using try/catch
fn safe_div(a: i32, b: i32):
    try:
        result = divide(a, b)
        println(result)
    catch e:
        println("Error: " + e)
    finally:
        println("Done")
```

---

## Prompts (AI)

```dim
prompt Classify:
    role system: "You are a classifier"
    role user: "Classify: {input}"
    output: enum Label:
        Positive
        Negative
        Neutral
```

---

## AI Tools

```dim
@tool(permissions=[NetRead])
fn fetch(url: str) -> str:
    return "content"

@tool(permissions=[FileRead("/tmp"), FileWrite("/tmp")])
fn cache(data: str) -> str:
    return data
```

---

## FFI (Foreign Function Interface)

```dim
foreign "libc.so" [
    fn puts(msg: str) -> i32
    fn rand() -> i32
    fn srand(seed: u32) -> unit
]

foreign "libm.so" [
    fn sin(x: f64) -> f64
    fn cos(x: f64) -> f64
    fn sqrt(x: f64) -> f64
]

use libc.rand
use libm.sqrt
```

---

## Modules and Imports

```dim
import std.io
import std.vec
import std.math
import std.str as string

# Alias import
import std.io as io

# From import
from std.vec import push, pop
```

---

## Standard Library

### I/O

```dim
print("Hello")
println("World")
let input_str = input("Enter: ")
let content = read_file("file.txt")
write_file("file.txt", content)
let exists = file_exists("file.txt")
```

### Math

```dim
let x = abs(-5)
let m = min(1, 2)
let mx = max(1, 2)
let s = sin(3.14159)
let c = cos(3.14159)
let root = sqrt(144.0)
let p = pow(2.0, 10.0)
let pi = math.PI()
let e = math.E()
```

### String

```dim
let len = "hello".len()
let upper = "hello".upper()
let lower = "HELLO".lower()
let trimmed = "  hello  ".trim()
let parts = "a,b,c".split(",")
let joined = ["a", "b", "c"].join("-")
```

---

## Operators

### Arithmetic
```dim
+   -   *   /   %   -
```

### Comparison
```dim
==  !=  <   >   <=  >=
```

### Logical
```dim
and  or  not
```

### Bitwise
```dim
&   |   ^   ~   <<   >>
```

---

## Keywords

```
fn      let     mut     const
if      elif    else    match
while   for     in      return
break   continue
struct  enum    trait   impl
async   await   actor   receive
prompt  role    output  tool
import  from    as      use
pub     priv    foreign
self    Self    try     catch
throw   finally panic
```

---

## Concurrency

```dim
# Thread pool usage
let threads = dim_thread_pool_init(4)
dim_thread_pool_submit(|| work())
dim_thread_pool_shutdown()

# Futures
let future = dim_future_new()
# ... async work ...
dim_future_await(future)
```

---

## Memory Management

```dim
# Reference counting
let data = dim_alloc_ref(100)
dim_inc_ref(data)
dim_dec_ref(data)

# GC
dim_gc_collect()
dim_gc_register_root(ptr)
```

---

## Testing

```dim
fn add(a: i32, b: i32) -> i32:
    return a + b

fn test_add_basic():
    result = add(2, 3)
    if result != 5 {
        panic("add(2, 3) should be 5")
    }

fn test_add_negative():
    result = add(-1, 1)
    if result != 0 {
        panic("add(-1, 1) should be 0")
    }

# Run with: dim test
```

---

_Dim Language v0.5.0_