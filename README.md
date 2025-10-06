[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
![CI](https://github.com/notPhani/c2bf-in-RUST/workflows/Rust%20CI/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub Stars](https://img.shields.io/github/stars/notPhani/c2bf-in-RUST?style=social)

# 🧠 C → Brainfuck (Rust) 💀

A caffeine-fueled C subset compiler that eats tokens and screams Brainfuck, implemented in Rust with hand-rolled everything: lexer, Pratt parser, semantic analysis with labeled scopes, source-level inliner via regex sorcery, compact IR, BF codegen with peephole cleanup, and a fat stack of tests that actually pass in ~0.14s on a potato.

**TL;DR:** Your code goes in as something sensible. Something comes out. It might be Brainfuck. You might cry.

---

## What it does

- **Parses C-ish** into an AST with unary/binary/ternary ops, arrays, loops, and function calls, then folds constants early to keep the tree thin and the brain sane(ish).
- **Runs semantic analysis** with "breadcrumb scopes" like `global.main.while3.block2` for sane resolution without a full symbol table stack, plus type checks, loop context checks, and initialized-use errors that yell in human.
- **Inlines functions** at the source level using regex extraction, fresh var names, and return rewriting; then re-tokenizes, re-parses, and re-semchecks because AST surgery was a war crime waiting to happen.
- **Lowers to a minimal IR** that maps cleanly to BF: cell ops, loops, I/O, and array helpers, then emits BF with pointer tracking, comments, and peephole no-op removal.
- **Ships 25 tests:** literals, calls, arrays-by-string, loops, if/else, comparisons, and Hello World; compiles to non-empty BF and enforces basic correctness heuristics.

---

## Pipeline / Architecture

```
tokenize → parse(AST) → semcheck → inline(source) → recompile → IR → BF → tests
```

If you look at this and think "oh, that's neat," I hate to break it to you: **chaos inside**.

**Public entrypoint:** `compile_to_brainfuck(source: &str) -> Result<String, String>` performing the full pipeline and returning Brainfuck or a spicy error message.

---

## Language Subset

### Types
- `int`, `char`, `string`
- Arrays as constant-size dims
- Strings can initialize arrays elementwise
- `char` and `int` interop allowed in expressions

### Control Flow
- `if/else`, `while`, `for` (IR implements while/for with recomputed conditions)
- `break`/`continue` (`continue` is a TODO marker)
- `return` with type-checking against the current function

### Expressions
- Full binary set including bitwise and shifts
- Unary `-`, `!`, `~`, `++`/`--` in typing rules
- Ternary parsed and folded when fully literal
- Function calls validated and inlined later

### I/O
- `putchar(expr)` lowers to `IR::Output` on the evaluated cell
- `getchar` tokenized but not wired through IR yet

---

## Lexer

**Key enums:** `Keywords`, `Operations`, `Punctuators`, `Token`

**Function:** `tokenize(source: &str) -> Vec<Token>`

**Strategy:** Direct char-peeking instead of DFA; handles comments, multi-char ops, and literal nightmares.

```rust
#[derive(Debug, Clone, PartialEq)]
enum Keywords { 
    Int, Char, String, Arr, If, Else, For, While, 
    PutChar, GetChar, Return, Function, Break, Continue 
}

#[derive(Debug, Clone, PartialEq)]
enum Operations {
    Add, Subtract, Multiply, Divide, Modulus,
    Equal, NotEqual, GreaterThan, LessThan, 
    GreaterThanOrEqual, LessThanOrEqual,
    And, Or, Not, Assign, Increment, Decrement,
    BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, 
    LeftShift, RightShift, Ternary
}

#[derive(Debug, Clone, PartialEq)]
enum Punctuators { 
    Semicolon, Colon, Comma, Dot, 
    LeftParen, RightParen, LeftBrace, RightBrace, 
    LeftBracket, RightBracket 
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Keyword(Keywords),
    Identifier(String),
    Number(i64),
    StringLiteral(String),
    CharLiteral(char),
    Operator(Operations),
    Punctuator(Punctuators)
}

fn tokenize(source: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut iter = source.chars().peekable();

    while let Some(&c) = iter.peek() {
        if c.is_whitespace() { iter.next(); continue; }
        if c.is_ascii_digit() { /* accumulate number */ }
        if c.is_alphabetic() || c == '_' { /* ident or keyword */ }
        if c == '"' { /* string literal */ }
        if c == '\'' { /* char literal */ }

        // Punctuators & Operators (==, !=, <=, >=, >>, <<, &&, ||, ++, --)
        match c {
            '+' | '-' | '*' | '/' | '%' | '=' | '!' | '>' | '<' 
            | '&' | '|' | '^' | '~' | '?' => { /* ... */ },
            _ => { iter.next(); }
        }
    }
    tokens
}
```

**Tradeoff:** No DFA—just direct branches for readability and explicit control over weird multi-token cases; cheap and cheerful for this grammar size.

---

## Parser (Pratt + Statements)

### ASTNode

Flat enum for all constructs. Assignments are sacred. Arrays and ternary supported. **Constant folding is the closest thing to self-care here.**

```rust
#[derive(Debug, Clone, PartialEq)]
enum ASTNode {
    Program(Vec<ASTNode>),
    Function { 
        return_type: Keywords, 
        name: String, 
        params: Vec<(Keywords, String)>, 
        body: Box<ASTNode> 
    },
    VariableDeclaration { 
        var_type: Keywords, 
        name: String, 
        array_dims: Option<Vec<usize>>, 
        initial_value: Option<Box<ASTNode>> 
    },
    For { 
        init: Box<ASTNode>, 
        condition: Box<ASTNode>, 
        increment: Box<ASTNode>, 
        body: Box<ASTNode> 
    },
    While { condition: Box<ASTNode>, body: Box<ASTNode> },
    If { 
        condition: Box<ASTNode>, 
        then_branch: Box<ASTNode>, 
        else_branch: Option<Box<ASTNode>> 
    },
    Assignment { target: Box<ASTNode>, value: Box<ASTNode> },
    BinaryOp { op: Operations, left: Box<ASTNode>, right: Box<ASTNode> },
    UnaryOp { op: Operations, expr: Box<ASTNode> },
    TernaryOp { 
        condition: Box<ASTNode>, 
        true_expr: Box<ASTNode>, 
        false_expr: Box<ASTNode> 
    },
    FunctionCall { name: String, args: Vec<ASTNode> },
    ArrayAccess { name: String, index: Box<ASTNode> },
    PutChar { expr: Box<ASTNode> },
    Break, 
    Continue,
    Return(Option<Box<ASTNode>>),
    LiteralInt(i64), 
    LiteralChar(char), 
    LiteralString(String),
    Identifier(String),
    Block(Vec<ASTNode>),
    Empty,
}
```

### Precedence & Expression Parsing

Pratt precedence map. Ternary parsed after RHS build (because chaos). Constant folding: shrink literal trees immediately.

```rust
fn get_precedence(op: &Operations) -> u8 {
    match op {
        Operations::Assign => 1,
        Operations::Or => 2,
        Operations::And => 3,
        Operations::BitwiseOr => 4,
        Operations::BitwiseXor => 5,
        Operations::BitwiseAnd => 6,
        Operations::Equal | Operations::NotEqual => 7,
        Operations::GreaterThan | Operations::LessThan 
        | Operations::GreaterThanOrEqual | Operations::LessThanOrEqual => 8,
        Operations::Add | Operations::Subtract => 9,
        Operations::Multiply | Operations::Divide | Operations::Modulus => 10,
        Operations::LeftShift | Operations::RightShift => 11,
        _ => 0, // unary/ternary/postfix handled elsewhere
    }
}
```

Parse **primary → postfix expansions → binary ops → optional ternary → fold**.

Assignments only as statements; LHS must be `Identifier` or `ArrayAccess`.

---

## Semantic Analysis

### Scopes as strings: `global.main.if_1` → literally follow the breadcrumbs

**Variable lookup:** walk up by truncating at `.` boundaries.

**Types:** `char→int` allowed; arrays need `int` indices; initialization tracked.

**Loop context:** push `inside_loop=true` in `While/For`.

### Semantic Errors (with unhinged messages)

```rust
#[derive(Debug, Clone)]
enum SemanticError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    TypeMismatch { expected: Keywords, found: Keywords },
    DuplicateDeclaration(String),
    ArgumentCountMismatch { expected: usize, found: usize },
    BreakOutsideLoop,
    ContinueOutsideLoop,
    ReturnTypeMismatch { expected: Keywords, found: Keywords },
    ArrayIndexNotInt,
    InvalidArraySize,
    UninitializedVariable(String),
}

impl std::fmt::Display for SemanticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SemanticError::UndefinedVariable(name) => 
                write!(f, "This variable vanished into thin air—undefined as hell! ({name})"),
            SemanticError::UndefinedFunction(name) => 
                write!(f, "This function is a ghost! Try summoning it first. ({name})"),
            SemanticError::TypeMismatch { expected, found } => 
                write!(f, "This ain't the type party you were expecting. Expected {expected:?}, found {found:?}."),
            SemanticError::ReturnTypeMismatch { expected, found } => 
                write!(f, "Return flavor is off. Expected {expected:?}, got {found:?}."),
            SemanticError::DuplicateDeclaration(name) => 
                write!(f, "This is already taken. Clone wars forbidden here! ({name})"),
            SemanticError::ArgumentCountMismatch { expected, found } => 
                write!(f, "This call expected {expected} args, but got {found}. Count your damn fingers!"),
            SemanticError::BreakOutsideLoop => 
                write!(f, "This break is lost outside any loop. Where ya breakin' from?!"),
            SemanticError::ContinueOutsideLoop => 
                write!(f, "This continue has nowhere to go. Loops only, buddy."),
            SemanticError::ArrayIndexNotInt => 
                write!(f, "Array index isn't an integer. Arrays like whole numbers only."),
            SemanticError::InvalidArraySize => 
                write!(f, "Invalid array size. Must be > 0—zero is a no-go here."),
            SemanticError::UninitializedVariable(name) => 
                write!(f, "Using uninitialized garbage! Initialize it first, you savage! ({name})"),
        }
    }
}
```

---

## Inlining (Source-to-Source)

### Iterative passes: extract → inline → strip; cap at 50 iterations

**Main untouched:** respect.

**Hygienic temps:** `__inl_<func>_<id>_argX`

**Return replacement:** `return X;` → `int <ret> = X;`

```rust
use regex::Regex;
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct FunctionSignature { 
    name: String, 
    params: Vec<String>, 
    body: String 
}

static mut INLINE_COUNTER: usize = 0;

// Entry: extract -> iterative inline -> strip non-main
pub fn clean_source(source: &str) -> String {
    unsafe { INLINE_COUNTER = 0; }
    let funcs = extract_functions(source);
    if funcs.is_empty() { return source.to_string(); }

    let mut result = source.to_string();
    for _ in 0..50 {
        let before = result.clone();
        for func in funcs.values() {
            result = inline_function(&result, func.clone());
        }
        if result == before { break; }
    }
    remove_functions(&result, &funcs)
}

fn extract_functions(source: &str) -> HashMap<String, FunctionSignature> {
    let mut funcs = HashMap::new();
    let re = Regex::new(
        r"(?:int|char|void)\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*\{([^}]*)\}"
    ).unwrap();

    for cap in re.captures_iter(source) {
        let name = cap[1].to_string();
        if name == "main" { continue; }

        let params: Vec<String> = cap[2]
            .split(',')
            .filter_map(|p| p.trim().split_whitespace().last().map(|s| s.to_string()))
            .collect();
        let body = cap[3].trim().to_string();

        funcs.insert(name.clone(), FunctionSignature { name, params, body });
    }
    funcs
}

// Replace call with hygienic temps + ret var, splice statements, restart
fn inline_function(source: &str, func: FunctionSignature) -> String {
    // find calls to func.name(...), split args respecting nesting,
    // inject:
    //   int __inl_<name>_<id>_argK = <arg>;
    // and rewrite returns in body to:
    //   int __inl_<name>_<id>_ret = <expr>;
    // then replace call site with __inl_<name>_<id>_ret
    // ...
    source.to_string() // actual implementation performs the transformations
}
```

---

## IR (Minimal, BF-Friendly)

### Explicit op set for BF mapping

Arrays with base+index temp. Linear cell allocator + temp pool. Pointer tracked to minimize `>/< `.

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum IRInstruction {
    // Memory
    Goto(usize),              // move pointer to cell
    Set(usize, i64),          // clear cell then set to value
    Add(usize, i64),          // add/sub
    Copy(usize, usize),       // non-destructive copy src -> dest
    Move(usize, usize),       // destructive move src -> dest

    // Control
    LoopStart(usize),
    LoopEnd(usize),

    // I/O
    Output(usize),
    Input(usize),

    // Arrays (helpers)
    ArrayLoad(usize, usize, usize),    // base, index_cell, dest
    ArrayStore(usize, usize, usize),   // base, index_cell, src

    // Debug
    Comment(String),
    Label(String),
}
```

**Temps reused** to curb BF explosion.

---

## Brainfuck Codegen

### Emit IR → BF with comments

Pointer moves minimized.

```rust
pub struct BrainfuckGenerator {
    current_pointer: usize,
    output: String,
    optimize: bool,
}

impl BrainfuckGenerator {
    pub fn new() -> Self { 
        Self { 
            current_pointer: 0, 
            output: String::new(), 
            optimize: true 
        } 
    }

    pub fn generate(&mut self, instructions: Vec<IRInstruction>) -> String {
        for instr in instructions {
            self.generate_instruction(instr);
        }
        if self.optimize {
            self.output = self.peephole_optimize(self.output.clone());
        }
        self.output.clone()
    }

    fn gen_goto(&mut self, cell: usize) {
        let distance = cell as isize - self.current_pointer as isize;
        if distance > 0 { 
            self.emit(&">".repeat(distance as usize)); 
        } else if distance < 0 { 
            self.emit(&"<".repeat((-distance) as usize)); 
        }
        self.current_pointer = cell;
    }

    fn gen_set(&mut self, cell: usize, value: i64) {
        self.gen_goto(cell);
        self.emit("[-]");
        if value > 0 { 
            self.emit(&"+".repeat(value as usize)); 
        } else if value < 0 { 
            self.emit(&"-".repeat((-value) as usize)); 
        }
    }

    fn gen_copy(&mut self, src: usize, dest: usize) {
        if src == dest { return; }
        // canonical non-destructive copy using two temp cells
        let temp1 = src.max(dest) + 1;
        let temp2 = temp1 + 1;
        // clear temps + dest, then funnel src -> (temp1, dest), 
        // then restore src from temp1
        // ...
    }

    fn peephole_optimize(&self, mut s: String) -> String {
        // remove "+-", "-+", "<>", "><", "[]", basic merges
        s = s.replace("+-", "")
             .replace("-+", "")
             .replace("<>", "")
             .replace("><", "")
             .replace("[]", "");
        s
    }

    fn emit(&mut self, code: &str) { 
        self.output.push_str(code); 
    }
}
```

**Logical ops** via loops. **Comparisons** via subtraction. 0/1 booleans. Brutal but works.

---

## Tests

**24+ examples:** literals, assignments, arithmetic, nested calls, if/else, loops, break/continue, multi-params, char usage, complex arithmetic, comparisons, unary negation, Hello World.

```rust
#[test]
fn test_02_function_call_with_inlining() {
    let source = r#"
        int get_value() { return 72; }
        int main() { putchar(get_value()); return 0; }
    "#;
    let result = compile_to_brainfuck(source);
    assert!(result.is_ok(), "Function inlining should work: {:?}", result.err());
}
```

**Run:**
```bash
cargo test
```

**Output:**
```
24 passed; 0 failed; finished in ~0.14s
```

---

## Design Notes and Tradeoffs

### Scopes as strings
- **Pros:** Dead simple resolution via dotted ancestry and portable across passes; no AST adornments or intrusive symbol tables needed.
- **Cons:** String churn and O(depth) lookups; fine for this scale, but a real compiler would intern and map.

### Source-level inlining
- **Pros:** Avoids gnarly AST rewrites; enables param substitution, hygienic temporaries, and return flattening in one sweep; easy to reason about and debug.
- **Cons:** Regex brittle on pathological formatting; mitigated by word-boundary replacements and balanced paren scans; reparse/semcheck catches introduced inconsistencies.

### IR minimalism
Keeps the BF mapping legible and measurable; explicit cells, explicit loops, explicit copies; no illusions about stacks or calls because BF has none; inlining becomes the "call mechanism".

### Safety rails
Uninitialized variable errors at use sites; invalid array sizes; array index type checks; return type enforcement; break/continue context checks; and semantic error `Display` messages that roast nonsense with gusto.

---

## Usage

### Library API
```rust
compile_to_brainfuck(source: &str) -> Result<String, String>
```
Performs: `parse → semcheck → inline → reparse → semcheck → IR → optimize → BF`

Returns `String` BF code or an error with spicy message text.

### Example Input
```c
int main() {
    putchar(65);
    return 0;
}
```

Will produce non-empty Brainfuck that prints 'A'.

---

## Build & Run

### Build optimized binary
```bash
cargo build --release
```

### Run and dump BF to file
```bash
cargo run -- path/to/file.c > out.bf
```

**Requirements:**
- `main()` function required
- Non-`main` functions are inlined then stripped

---

## Deployment

### CLI Binary
Ship the release binary from `target/release/` as the CLI tool; it reads a C source file and writes Brainfuck to stdout, so it composes well in pipelines or CI steps.

### Containerization (Optional)
Build in a Rust image stage, then copy the single binary into a scratch/alpine final stage; the program is a single static-ish binary with no runtime assets by design.

**Example Dockerfile:**
```dockerfile
FROM rust:latest as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM alpine:latest
COPY --from=builder /app/target/release/c-to-bf /usr/local/bin/
ENTRYPOINT ["c-to-bf"]
```

### Web Playground (Future)
The IR-to-BF generator is pure-Rust and amenable to `wasm32-unknown-unknown` builds; pair with an in-browser BF runner for a delightful chaos sandbox.

---

## Limits and Next Moves

### Current Limitations
- **No pointers/structs:** Arrays are constant-size and currently lowered with base-cell only in IR array ops; pointer arithmetic for arrays is a prime target.
- **Continue is emitted as a comment:** Proper support needs loop-body structuring and/or flag choreography; doable within current IR with a prepass.
- **Conservative peephole:** RLE, strength reduction, and copy/move fusion would shrink code a lot; IR-level DCE exists as `optimize_ir` hook but can be expanded.
- **getchar limited support:** Exists in tokens and IR but needs ergonomic expression support in front-end typing to round-trip safely.

### Future Enhancements
- Full pointer arithmetic and dynamic array indexing
- Proper `continue` implementation with control flow restructuring
- Advanced peephole optimizations (RLE, pattern fusion)
- Source-mapped errors that trace back through inlining
- Web playground with WASM BF runner

---

## If this looks neat, that's a trap

**Under the hood:** A string-labeled scope graph instead of a symbol stack, a regex-powered inliner that rewrites returns into temps and punts to a second compilation, an IR that leans into BF's brutality, and BF codegen that prefers correctness patterns over code golf.

It shouldn't work this cleanly.

It does.

At 3 AM.

With questionable joy.

⚠️ **Warning:** Using this compiler may cause existential crises, sleepless nights, and irrational joy.
