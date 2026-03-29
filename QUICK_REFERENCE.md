# Dim Language Quick Reference

## Comments

```dim
# Single line comment
```

## Variables

```dim
let x = 42          # Immutable binding
let mut y = 10      # Mutable binding
y = y + 1           # Reassign mutable
```

## Types

### Primitive Types

```dim
let a: i32 = 42
let b: f32 = 3.14
let c: bool = true
let s: string = "hello"
let u: () = ()       # Unit type
```

### Collections

```dim
let arr: [i32, 3] = [1, 2, 3]    # Fixed-size array
# Tensor coming soon
```

## Functions

```dim
fn add(x: i32, y: i32) -> i32:
    return x + y

fn greet(name: string):
    print(name)

fn main():
    let result = add(1, 2)
```

### Async Functions

```dim
async fn fetch(url: string) -> string:
    let data = await http_get(url)
    return data
```

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
    print(item)
```

### Match

```dim
match value:
    0: return "zero"
    n if n > 0: return "positive"
    _: return "negative"
```

## Structures

```dim
struct Point:
    x: i32
    y: i32

fn create_point(x: i32, y: i32) -> Point:
    return Point(x: x, y: y)
```

## Enums

```dim
enum Color:
    Red
    Green
    Blue

enum Result:
    Ok(value: i32)
    Err(msg: string)
```

## Traits

```dim
trait Printable:
    fn print(self)

struct Point:
    x: i32
    y: i32

impl Printable for Point:
    fn print(self):
        print(self.x)
```

## Generics

```dim
fn identity[T](x: T) -> T:
    return x

fn max[T: Comparable](a: T, b: T) -> T:
    if a > b:
        return a
    return b
```

## Borrowing

```dim
fn process(data: Buffer):
    let view = &data       # Immutable borrow
    let ptr = &mut data    # Mutable borrow
```

## Prompts (AI)

```dim
prompt Classify:
    role system: "You are a classifier"
    role user: "Classify: {input}"
    output: enum Result:
        Positive
        Negative
```

## AI Tools

```dim
@tool(permissions=[NetRead])
fn fetch(url: string) -> string:
    return "content"

@tool(permissions=[FileRead('/tmp'), FileWrite('/tmp')])
fn cache(data: string) -> string:
    return data
```

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

## Keywords

```
fn      let     mut     const
if      elif    else
while   for     in
match   case
return  break   continue
struct  enum    trait    impl
async   await   spawn
prompt  role    output
@tool   model
verify  unsafe
import  from    as
pub     priv
self    Self
extends where
```

## Error Handling

```dim
fn divide(a: i32, b: i32) -> Result[i32, string]:
    if b == 0:
        return Err("division by zero")
    return Ok(a / b)
```
