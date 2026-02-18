# Dim Language: Comprehensive Examples

These examples demonstrate the "working" ergonomics and technical depth of the Dim language.

## 1. Systems & Security: Secure Packet Parser

Demonstrates indentation syntax, `Option/Result`, ownership, and capability-based I/O.

```dim
import std.net
import std.security

struct Packet:
    header: u32
    payload: Buffer
    checksum: u64

fn parse_secure_packet(cap: &NetCapability, port: u16) -> Result[Packet, Error]:
    # Capability-checked network access
    let stream = cap.listen(port)?

    verify:
        # Symbolic execution ensures buffer safety
        let raw = stream.read_fixed(1024)
        if raw.len < 12:
            return Err(Error.Truncated)

    let p = Packet(
        header: raw.read_u32(0),
        payload: raw.slice(4, raw.len - 8),
        checksum: raw.read_u64(raw.len - 8)
    )

    if security.validate_hmac(&p.payload, p.checksum):
        return Ok(p)
    return Err(Error.InvalidSignature)
```

## 2. AI/ML: Intelligent Defensive Agent

Demonstrates native `prompt` types, structured outputs, and tensor-based inference.

```dim
prompt DefenderAction:
    role system: "You are an autonomous firewall agent."
    role user: "Analyze these logs: {logs}"
    output: enum Action:
        Block(ip: string)
        Alert(msg: string)
        Ignore

fn monitor_network(logs: Tensor[f32]):
    let model = load_model("dim-guardian-v1")

    # Run inference in a sandboxed context
    with model.sandbox:
        let decision = await model.execute(DefenderAction(logs.to_string()))

        match decision:
            Action.Block(ip):
                firewall.drop(ip)
            Action.Alert(msg):
                log.warn("Anomalous activity: {msg}")
            Action.Ignore:
                pass
```

## 3. Web & JS Interop: High-Perf Data Processor

Demonstrates async/await, JS FFI, and WASM-friendly constructs.

```dim
extern "js" fn update_ui(data: string)

@export
fn process_and_render(raw_json: string) async:
    # High-speed parsing in Dim (WASM)
    let processed = await Task.spawn(fn:
        let data = json.parse(raw_json)
        return data.transform_optimized()
    )

    # Call back to JS for DOM updates
    update_ui(processed.to_json())
```
