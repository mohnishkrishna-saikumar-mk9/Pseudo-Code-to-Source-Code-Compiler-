# 🔄 Pseudo Code to Source Code Compiler

> A complete compiler that translates human-readable pseudo code into **C**, **C++**, and **Java** source code with a modern GUI interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Compiler Architecture](#compiler-architecture)
- [Supported Pseudocode Constructs](#supported-pseudocode-constructs)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Project Structure](#project-structure)
- [Testing](#testing)

---

## 🔍 Overview

This project implements a **multi-phase compiler** that takes pseudo code as input and generates syntactically correct source code in three target languages. It demonstrates core compiler design concepts:

- **Lexical Analysis** — Tokenizes pseudocode into classified tokens
- **Recursive Descent Parsing** — Builds an Abstract Syntax Tree (AST)
- **AST Construction** — Represents program structure as a tree
- **Multi-language Code Generation** — Translates AST into C, C++, and Java

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔤 **Lexer** | Tokenizes input into keywords, identifiers, operators, numbers, strings |
| 🌳 **Parser** | Recursive descent parser builds a complete AST |
| 💻 **3 Language Targets** | Generates production-ready C, C++, and Java code |
| 🖥️ **GUI Interface** | Modern Tkinter GUI with syntax highlighting and token display |
| 📂 **File I/O** | Load pseudocode from files and save generated code |
| 🔁 **Control Flow** | Supports if/else, while loops, for loops, nested structures |
| 📦 **Arrays** | Full array declaration, access, and manipulation |
| ⚡ **Functions** | User-defined functions with parameters |
| 🔤 **String Support** | String variables with proper type handling per language |
| 📊 **Token Table** | Displays complete token analysis in tabular format |

---

## 🏗️ Compiler Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Pseudo Code │────▶│    Lexer      │────▶│    Parser    │────▶│  Code Gen    │
│   (Input)    │     │  (Tokenizer) │     │    (AST)     │     │  (C/C++/Java)│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                          │                      │                      │
                     Token Stream          Abstract Syntax        Target Source
                     [KEYWORD:if]            Tree (AST)              Code
                     [OP:==]              ┌──Program──┐         #include<stdio.h>
                     [ID:x]               │IfStatement│         int main() { ... }
                     [NUMBER:5]           └───────────┘
```

---

## 📝 Supported Pseudocode Constructs

### Variables & Types
```
declare integer x
declare float pi
declare string name
declare integer arr[10]
```

### Assignment
```
set x to 10
set name to "hello"
set arr[0] to 42
```

### Input / Output
```
input x
print x
print "Hello World"
print arr[i]
```

### Conditional Statements
```
if x > 10 then
    print "big"
else
    print "small"
endif
```

### Loops
```
for i = 0 to 10
    print i
endfor

while x > 0 do
    set x to x - 1
endwhile
```

### Functions
```
function add(a, b)
    print a + b
endfunction
```

### Control Flow
```
break
continue
return x
```

---

## 🖥️ Screenshots

### GUI Interface
The compiler features a modern dark-themed GUI with:
- **Left Panel** — Pseudocode input editor
- **Right Panel** — Generated source code output
- **Bottom Panel** — Token analysis table
- **Toolbar** — Language selection (C / C++ / Java), Compile, Clear, Save, Load

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- Tkinter (included with Python on most systems)

### Steps
```bash
# Clone the repository
git clone https://github.com/mohnishkrishna-saikumar-mk9/Pseudo-Code-to-Source-Code-Compiler-.git

# Navigate to the project directory
cd Pseudo-Code-to-Source-Code-Compiler-

# Run the compiler
python "Compiler Model.py"
```

> **Note:** No external dependencies required — uses only Python standard library.

---

## 🚀 Usage

1. **Launch** the GUI by running `python "Compiler Model.py"`
2. **Write** or **paste** pseudocode in the left input panel
3. **Select** target language: `C`, `C++`, or `Java`
4. **Click** `🔄 Compile` to generate source code
5. **View** the generated code in the right panel and tokens in the bottom panel
6. **Save** the output using the `💾 Save` button
7. **Load** pseudocode files using the `📂 Load` button

---

## 📖 Example

### Input Pseudocode
```
start
declare integer n
declare integer i
declare integer sum

set sum to 0

print "Enter a number:"
input n

for i = 1 to n
    set sum to sum + i
endfor

print "Sum is:"
print sum
end
```

### Generated C Code
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int main() {
    int n;
    int i;
    int sum;
    sum = 0;
    printf("Enter a number:\n");
    scanf("%d", &n);
    for (int i = 1; i < n; i++) {
        sum = sum + i;
    }
    printf("Sum is:\n");
    printf("%d\n", sum);
    return 0;
}
```

### Generated C++ Code
```cpp
#include <iostream>
#include <string>
#include <cctype>
using namespace std;

int main() {
    int n;
    int i;
    int sum;
    sum = 0;
    cout << "Enter a number:" << endl;
    cin >> n;
    for (int i = 1; i < n; i++) {
        sum = sum + i;
    }
    cout << "Sum is:" << endl;
    cout << sum << endl;
    return 0;
}
```

### Generated Java Code
```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n;
        int i;
        int sum;
        sum = 0;
        System.out.println("Enter a number:");
        n = scanner.nextInt();
        for (int i = 1; i < n; i++) {
            sum = sum + i;
        }
        System.out.println("Sum is:");
        System.out.println(sum);
        scanner.close();
    }
}
```

---

## 📁 Project Structure

```
📦 Pseudo-Code-to-Source-Code-Compiler
├── 📄 Compiler Model.py              # Main compiler source code (Lexer, Parser, Generators, GUI)
├── 📄 Test Pseudo Codes.txt           # Collection of test pseudocodes (easy, medium, complex)
├── 📄 tokenizer_pseudocode.txt        # Sample tokenizer pseudocode
├── 📄 Compiler_Project_Presentation.pptx  # Project presentation
└── 📄 README.md                       # Project documentation
```

### Module Breakdown (Compiler Model.py)

| Module | Lines | Description |
|--------|-------|-------------|
| `Lexer` | ~180 | Tokenizes pseudocode into token stream |
| `Parser` | ~250 | Recursive descent parser, builds AST |
| `CCodeGenerator` | ~150 | Generates C source code from AST |
| `CPPCodeGenerator` | ~140 | Generates C++ source code from AST |
| `JavaCodeGenerator` | ~170 | Generates Java source code from AST |
| `Compiler` | ~45 | Orchestrates lexing → parsing → code generation |
| `CompilerGUI` | ~230 | Tkinter-based graphical user interface |

---

## 🧪 Testing

The project includes test pseudocodes covering various complexity levels:

### Easy
- Variable declaration and printing
- Simple if-else conditions
- Basic arithmetic operations

### Medium
- For loop with accumulator
- While loop countdown
- Nested if with modulo (FizzBuzz)
- Array operations

### Complex
- Nested loops (multiplication table)
- String manipulation with character analysis
- Functions with parameters
- Multi-array operations
- Combined loops, conditionals, and arrays

---

## 🛠️ Technologies Used

- **Language:** Python 3
- **GUI Framework:** Tkinter
- **Compiler Techniques:** Lexical Analysis, Recursive Descent Parsing, AST, Code Generation
- **Target Languages:** C, C++, Java

---

## 👥 Contributors

- **G Mohnish Krishna Saikumar** — [GitHub](https://github.com/mohnishkrishna-saikumar-mk9)
- **K Mokshagna** — [GitHub](https://github.com/moksha-hub)
- **A Surendra Naidu** - [GitHub](https://github.com/arigisurendranaidu2005-code)
---
