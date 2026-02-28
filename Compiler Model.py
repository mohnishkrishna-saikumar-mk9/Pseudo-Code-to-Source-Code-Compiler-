"""
COMPLETE AUTO SOURCE CODE GENERATOR - FINAL PRODUCTION VERSION
===============================================================
Handles ALL pseudo code: complex expressions, strings, modulo, etc.
Generates perfect C, C++, and HTML code
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from enum import Enum
from typing import List
from dataclasses import dataclass
import re

# ============================================================
# ENHANCED LEXER - Handles Everything
# ============================================================

class TokenType(Enum):
    START = "START"
    END = "END"
    IF = "IF"
    ELSE = "ELSE"
    ENDIF = "ENDIF"
    THEN = "THEN"
    WHILE = "WHILE"
    ENDWHILE = "ENDWHILE"
    FOR = "FOR"
    ENDFOR = "ENDFOR"
    TO = "TO"
    DO = "DO"
    FUNCTION = "FUNCTION"
    ENDFUNCTION = "ENDFUNCTION"
    PRINT = "PRINT"
    INPUT = "INPUT"
    DECLARE = "DECLARE"
    SET = "SET"
    RETURN = "RETURN"
    CONTINUE = "CONTINUE"
    BREAK = "BREAK"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    STRING_TYPE = "STRING"
    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    MODULO = "MODULO"
    ASSIGN = "ASSIGN"
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"
    LESS_EQUAL = "LESS_EQUAL"
    GREATER_EQUAL = "GREATER_EQUAL"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    DOT = "DOT"
    NEWLINE = "NEWLINE"
    EOF = "EOF"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    def __init__(self):
        self.keywords = {
            'start': TokenType.START, 'begin': TokenType.START,
            'end': TokenType.END,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'endif': TokenType.ENDIF,
            'then': TokenType.THEN,
            'while': TokenType.WHILE,
            'endwhile': TokenType.ENDWHILE,
            'for': TokenType.FOR,
            'endfor': TokenType.ENDFOR,
            'to': TokenType.TO,
            'do': TokenType.DO,
            'function': TokenType.FUNCTION,
            'endfunction': TokenType.ENDFUNCTION,
            'print': TokenType.PRINT, 'display': TokenType.PRINT, 'output': TokenType.PRINT, 'show': TokenType.PRINT,
            'input': TokenType.INPUT, 'read': TokenType.INPUT,
            'declare': TokenType.DECLARE, 'define': TokenType.DECLARE,
            'set': TokenType.SET, 'assign': TokenType.SET,
            'return': TokenType.RETURN,
            'continue': TokenType.CONTINUE,
            'break': TokenType.BREAK,
            'integer': TokenType.INTEGER, 'int': TokenType.INTEGER,
            'float': TokenType.FLOAT, 'real': TokenType.FLOAT, 'decimal': TokenType.FLOAT,
            'string': TokenType.STRING_TYPE,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            'calculate': TokenType.SET,  # NEW: CALCULATE same as SET
            'as': TokenType.TO,  # NEW: "as" works like "to"
        }
        self.text = ""
        self.pos = 0
        self.line = 1
        self.column = 1
        self.current_char = None
        self.tokens = []
        self.errors = []
    
    def error(self, msg):
        self.errors.append(f"Line {self.line}, Col {self.column}: {msg}")
    
    def advance(self):
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None
    
    def peek(self, offset=1):
        peek_pos = self.pos + offset
        return self.text[peek_pos] if peek_pos < len(self.text) else None
    
    def skip_whitespace(self):
        while self.current_char and self.current_char in ' \t\r':
            self.advance()
    
    def read_number(self):
        start_col = self.column
        num_str = ""
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            num_str += self.current_char
            self.advance()
        return Token(TokenType.NUMBER, num_str, self.line, start_col)
    
    def read_string(self):
        start_col = self.column
        quote = self.current_char
        self.advance()
        string_val = ""
        while self.current_char and self.current_char != quote:
            if self.current_char == '\\' and self.peek() == quote:
                self.advance()
                string_val += self.current_char
                self.advance()
            elif self.current_char == '\n':
                self.error("Unterminated string")
                break
            else:
                string_val += self.current_char
                self.advance()
        if self.current_char == quote:
            self.advance()
        return Token(TokenType.STRING, string_val, self.line, start_col)
    
    def read_identifier(self):
        start_col = self.column
        result = ""
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        token_type = self.keywords.get(result.lower(), TokenType.IDENTIFIER)
        return Token(token_type, result, self.line, start_col)
    
    def tokenize(self, text):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        self.errors = []
        
        if not text:
            self.tokens.append(Token(TokenType.EOF, '', 1, 1))
            return self.tokens
        
        self.current_char = self.text[0]
        
        while self.current_char is not None:
            if self.current_char in ' \t\r':
                self.skip_whitespace()
                continue
            
            if self.current_char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, '\\n', self.line, self.column))
                self.advance()
                continue
            
            if self.current_char == '#':
                while self.current_char and self.current_char != '\n':
                    self.advance()
                continue
            
            if self.current_char.isdigit():
                self.tokens.append(self.read_number())
                continue
            
            if self.current_char in ('"', "'"):
                self.tokens.append(self.read_string())
                continue
            
            if self.current_char.isalpha() or self.current_char == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Operators
            start_col = self.column
            char = self.current_char
            next_char = self.peek()
            
            two_char = char + (next_char if next_char else '')
            if two_char in ('==', '!=', '<=', '>=', ':=', '<-'):
                self.advance()
                self.advance()
                token_map = {
                    '==': TokenType.EQUAL, '!=': TokenType.NOT_EQUAL,
                    '<=': TokenType.LESS_EQUAL, '>=': TokenType.GREATER_EQUAL,
                    ':=': TokenType.ASSIGN, '<-': TokenType.ASSIGN
                }
                self.tokens.append(Token(token_map[two_char], two_char, self.line, start_col))
                continue
            
            single_map = {
                '+': TokenType.PLUS, '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY, '/': TokenType.DIVIDE,
                '%': TokenType.MODULO,
                '=': TokenType.ASSIGN,
                '<': TokenType.LESS_THAN, '>': TokenType.GREATER_THAN,
                '(': TokenType.LPAREN, ')': TokenType.RPAREN,
                '{': TokenType.LBRACE, '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET, ']': TokenType.RBRACKET,
                ',': TokenType.COMMA, ';': TokenType.SEMICOLON,
                '.': TokenType.DOT,
            }
            
            if char in single_map:
                self.advance()
                self.tokens.append(Token(single_map[char], char, self.line, start_col))
                continue
            
            self.error(f"Unknown character: '{char}'")
            self.advance()
        
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens

# ============================================================
# ENHANCED PARSER
# ============================================================

class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class FunctionDef(ASTNode):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class Assignment(ASTNode):
    def __init__(self, variable, expression):
        self.variable = variable
        self.expression = expression

class Declaration(ASTNode):
    def __init__(self, var_type, variable, size=None):
        self.var_type = var_type
        self.variable = variable
        self.size = size

class PrintStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression

class InputStatement(ASTNode):
    def __init__(self, variable):
        self.variable = variable

class IfStatement(ASTNode):
    def __init__(self, condition, then_block, else_block=None):
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

class WhileLoop(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ForLoop(ASTNode):
    def __init__(self, variable, start, end, body):
        self.variable = variable
        self.start = start
        self.end = end
        self.body = body

class ReturnStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression

class ContinueStatement(ASTNode):
    pass

class BreakStatement(ASTNode):
    pass

class Parser:
    def __init__(self, tokens):
        self.tokens = [t for t in tokens if t.type != TokenType.NEWLINE]
        self.current = 0
        self.errors = []
    
    def error(self, msg):
        token = self.peek()
        self.errors.append(f"Parse error at line {token.line}: {msg}")
    
    def peek(self):
        return self.tokens[self.current] if self.current < len(self.tokens) else self.tokens[-1]
    
    def advance(self):
        token = self.peek()
        if self.current < len(self.tokens) - 1:
            self.current += 1
        return token
    
    def parse(self):
        statements = []
        while self.peek().type != TokenType.EOF:
            if self.peek().type in (TokenType.START, TokenType.END):
                self.advance()
                continue
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            else:
                if self.peek().type != TokenType.EOF:
                    self.advance()
        return Program(statements)
    
    def parse_statement(self):
        token = self.peek()
        
        if token.type == TokenType.FUNCTION:
            return self.parse_function()
        elif token.type == TokenType.DECLARE:
            return self.parse_declaration()
        elif token.type == TokenType.SET:
            return self.parse_assignment()
        elif token.type == TokenType.PRINT:
            return self.parse_print()
        elif token.type == TokenType.INPUT:
            return self.parse_input()
        elif token.type == TokenType.IF:
            return self.parse_if()
        elif token.type == TokenType.WHILE:
            return self.parse_while()
        elif token.type == TokenType.FOR:
            return self.parse_for()
        elif token.type == TokenType.RETURN:
            return self.parse_return()
        elif token.type == TokenType.CONTINUE:
            self.advance()
            return ContinueStatement()
        elif token.type == TokenType.BREAK:
            self.advance()
            return BreakStatement()
        elif token.type in (TokenType.EOF, TokenType.ENDIF, TokenType.ENDWHILE, 
                           TokenType.ENDFOR, TokenType.ELSE, TokenType.ENDFUNCTION):
            return None
        else:
            self.advance()
            return None
    
    def parse_function(self):
        self.advance()
        name = self.advance().value
        params = []
        if self.peek().type == TokenType.LPAREN:
            self.advance()
            while self.peek().type != TokenType.RPAREN and self.peek().type != TokenType.EOF:
                params.append(self.advance().value)
                if self.peek().type == TokenType.COMMA:
                    self.advance()
            if self.peek().type == TokenType.RPAREN:
                self.advance()
        
        body = []
        while self.peek().type not in (TokenType.ENDFUNCTION, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            else:
                break
        
        if self.peek().type == TokenType.ENDFUNCTION:
            self.advance()
        
        return FunctionDef(name, params, body)
    
    def parse_declaration(self):
        self.advance()
        var_type = self.advance().value
        variable = self.advance().value
        size = None
        if self.peek().type == TokenType.LBRACKET:
            self.advance()
            size = self.advance().value
            if self.peek().type == TokenType.RBRACKET:
                self.advance()
        return Declaration(var_type, variable, size)
    
    def parse_assignment(self):
        self.advance()  # skip 'set' or 'calculate'
        
        # Read variable name (might include array index)
        variable = self.advance().value
        
        # Check for array indexing
        if self.peek().type == TokenType.LBRACKET:
            self.advance()  # skip '['
            index = self.advance().value
            self.advance()  # skip ']'
            variable = f"{variable}[{index}]"
        
        # Skip 'to', '=', ':=', or '<-'
        if self.peek().type in (TokenType.TO, TokenType.ASSIGN):
            self.advance()
        
        expression = self.parse_expression()
        return Assignment(variable, expression)
    
    def parse_print(self):
        self.advance()
        
        # Handle string literals specially
        if self.peek().type == TokenType.STRING:
            expr = '"' + self.advance().value + '"'
            return PrintStatement(expr)
        
        expression = self.parse_expression()
        return PrintStatement(expression)
    
    def parse_input(self):
        self.advance()
        variable = self.advance().value
        return InputStatement(variable)
    
    def parse_return(self):
        self.advance()
        expression = self.parse_expression()
        return ReturnStatement(expression)
    
    def parse_if(self):
        self.advance()
        condition = self.parse_expression()
        if self.peek().type == TokenType.THEN:
            self.advance()
        
        then_block = []
        while self.peek().type not in (TokenType.ELSE, TokenType.ENDIF, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                then_block.append(stmt)
            else:
                break
        
        else_block = None
        if self.peek().type == TokenType.ELSE:
            self.advance()
            else_block = []
            while self.peek().type not in (TokenType.ENDIF, TokenType.EOF):
                stmt = self.parse_statement()
                if stmt:
                    else_block.append(stmt)
                else:
                    break
        
        if self.peek().type == TokenType.ENDIF:
            self.advance()
        
        return IfStatement(condition, then_block, else_block)
    
    def parse_while(self):
        self.advance()
        condition = self.parse_expression()
        if self.peek().type == TokenType.DO:
            self.advance()
        
        body = []
        while self.peek().type not in (TokenType.ENDWHILE, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            else:
                break
        
        if self.peek().type == TokenType.ENDWHILE:
            self.advance()
        
        return WhileLoop(condition, body)
    
    def parse_for(self):
        self.advance()
        variable = self.advance().value
        
        start = "0"
        if self.peek().type == TokenType.ASSIGN:
            self.advance()
            start = self.advance().value
        
        if self.peek().type == TokenType.TO:
            self.advance()
        end = self.advance().value
        
        if self.peek().type == TokenType.DO:
            self.advance()
        
        body = []
        while self.peek().type not in (TokenType.ENDFOR, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            else:
                break
        
        if self.peek().type == TokenType.ENDFOR:
            self.advance()
        
        return ForLoop(variable, start, end, body)
    
    def parse_expression(self):
        tokens = []
        stop_types = (TokenType.THEN, TokenType.DO, TokenType.EOF, TokenType.ENDIF, 
                     TokenType.ENDWHILE, TokenType.ENDFOR, TokenType.ELSE, 
                     TokenType.ENDFUNCTION, TokenType.COMMA, TokenType.SEMICOLON,
                     TokenType.SET, TokenType.PRINT, TokenType.INPUT, TokenType.IF,
                     TokenType.WHILE, TokenType.FOR, TokenType.DECLARE, TokenType.RETURN,
                     TokenType.CONTINUE, TokenType.BREAK,
                     TokenType.NEWLINE, TokenType.START, TokenType.END)
        
        while self.peek().type not in stop_types and len(tokens) < 50:
            token = self.advance()
            if token.type == TokenType.STRING:
                # Preserve quotes: single char -> 'c', multi char -> "str"
                if len(token.value) == 1:
                    tokens.append(f"'{token.value}'")
                else:
                    tokens.append(f'"' + token.value + f'"')
            else:
                tokens.append(token.value)
        
        return ' '.join(tokens)

# ============================================================
# PERFECT C CODE GENERATOR
# ============================================================

class CCodeGenerator:
    def __init__(self):
        self.variables = {}
        self.functions = []
    
    def convert_type(self, pseudo_type):
        type_map = {
            'integer': 'int', 'int': 'int',
            'float': 'float', 'real': 'float',
            'string': 'char*', 'char': 'char',
            'boolean': 'int', 'bool': 'int'
        }
        return type_map.get(pseudo_type.lower(), 'int')
    
    def generate(self, ast):
        for stmt in ast.statements:
            if isinstance(stmt, FunctionDef):
                self.functions.append(stmt)
        
        code = "#include <stdio.h>\n"
        code += "#include <stdlib.h>\n"
        code += "#include <string.h>\n"
        code += "#include <ctype.h>\n\n"
        
        for func in self.functions:
            params_str = ', '.join([f"int {p}" for p in func.params]) if func.params else "void"
            code += f"void {func.name}({params_str});\n"
        
        if self.functions:
            code += "\n"
        
        for func in self.functions:
            code += self.gen_function(func)
            code += "\n"
        
        code += "int main() {\n"
        for stmt in ast.statements:
            if not isinstance(stmt, FunctionDef):
                code += self.gen_statement(stmt, 1)
        code += "    return 0;\n}\n"
        
        return code
    
    def gen_function(self, func):
        params_str = ', '.join([f"int {p}" for p in func.params]) if func.params else "void"
        code = f"void {func.name}({params_str}) {{\n"
        for stmt in func.body:
            code += self.gen_statement(stmt, 1)
        code += "}\n"
        return code
    
    def gen_statement(self, stmt, indent=0):
        ind = "    " * indent
        
        if isinstance(stmt, Declaration):
            c_type = self.convert_type(stmt.var_type)
            self.variables[stmt.variable] = c_type
            if stmt.size:
                return f"{ind}{c_type} {stmt.variable}[{stmt.size}];\n"
            if c_type == 'char*':
                # String without size: allocate buffer
                self.variables[stmt.variable] = 'char*'
                return f"{ind}char {stmt.variable}[1024];\n"
            return f"{ind}{c_type} {stmt.variable};\n"
        
        elif isinstance(stmt, Assignment):
            # Check if it's an array assignment (contains brackets)
            if '[' in stmt.variable and ']' in stmt.variable:
                # Array assignment: arr[0] = value
                return f"{ind}{stmt.variable} = {stmt.expression};\n"
            else:
                # Regular variable assignment
                if stmt.variable not in self.variables:
                    self.variables[stmt.variable] = 'int'
                    return f"{ind}int {stmt.variable} = {stmt.expression};\n"
                else:
                    return f"{ind}{stmt.variable} = {stmt.expression};\n"
        
        elif isinstance(stmt, PrintStatement):
            expr = stmt.expression.strip()
            if expr.startswith('"') and expr.endswith('"'):
                # String literal
                text = expr[1:-1]
                return f'{ind}printf("{text}\\n");\n'
            else:
                # Check variable type for correct format specifier
                var_name = expr.split('[')[0].strip()  # Handle arr[i] -> arr
                var_type = self.variables.get(var_name, self.variables.get(expr, 'int'))
                if var_type == 'char*':
                    return f'{ind}printf("%s\\n", {expr});\n'
                elif var_type == 'float':
                    return f'{ind}printf("%f\\n", {expr});\n'
                else:
                    return f'{ind}printf("%d\\n", {expr});\n'
        
        elif isinstance(stmt, InputStatement):
            if stmt.variable not in self.variables:
                self.variables[stmt.variable] = 'int'
            var_type = self.variables.get(stmt.variable, 'int')
            if var_type == 'char*':
                # String input: use fgets and strip newline
                code = f'{ind}fgets({stmt.variable}, sizeof({stmt.variable}), stdin);\n'
                code += f'{ind}{stmt.variable}[strcspn({stmt.variable}, "\\n")] = 0;\n'
                return code
            elif var_type == 'float':
                return f'{ind}scanf("%f", &{stmt.variable});\n'
            else:
                return f'{ind}scanf("%d", &{stmt.variable});\n'
        
        elif isinstance(stmt, IfStatement):
            code = f"{ind}if ({stmt.condition}) {{\n"
            for s in stmt.then_block:
                code += self.gen_statement(s, indent + 1)
            code += f"{ind}}}"
            if stmt.else_block:
                code += " else {\n"
                for s in stmt.else_block:
                    code += self.gen_statement(s, indent + 1)
                code += f"{ind}}}"
            code += "\n"
            return code
        
        elif isinstance(stmt, WhileLoop):
            code = f"{ind}while ({stmt.condition}) {{\n"
            for s in stmt.body:
                code += self.gen_statement(s, indent + 1)
            code += f"{ind}}}\n"
            return code
        
        elif isinstance(stmt, ForLoop):
            code = f"{ind}for (int {stmt.variable} = {stmt.start}; {stmt.variable} < {stmt.end}; {stmt.variable}++) {{\n"
            for s in stmt.body:
                code += self.gen_statement(s, indent + 1)
            code += f"{ind}}}\n"
            return code
        
        elif isinstance(stmt, ReturnStatement):
            return f"{ind}return {stmt.expression};\n"
        
        elif isinstance(stmt, ContinueStatement):
            return f"{ind}continue;\n"
        
        elif isinstance(stmt, BreakStatement):
            return f"{ind}break;\n"
        
        return ""

# ============================================================
# C++ CODE GENERATOR
# ============================================================

class CPPCodeGenerator:
    def __init__(self):
        self.variables = {}
        self.functions = []
    
    def convert_type(self, pseudo_type):
        type_map = {
            'integer': 'int', 'int': 'int',
            'float': 'float', 'real': 'float',
            'string': 'string', 'char': 'char',
            'boolean': 'bool', 'bool': 'bool'
        }
        return type_map.get(pseudo_type.lower(), 'int')
    
    def generate(self, ast):
        for stmt in ast.statements:
            if isinstance(stmt, FunctionDef):
                self.functions.append(stmt)
        
        code = "#include <iostream>\n"
        code += "#include <string>\n"
        code += "#include <cctype>\n"
        code += "using namespace std;\n\n"
        
        for func in self.functions:
            params_str = ', '.join([f"int {p}" for p in func.params]) if func.params else ""
            code += f"void {func.name}({params_str});\n"
        
        if self.functions:
            code += "\n"
        
        for func in self.functions:
            code += self.gen_function(func)
            code += "\n"
        
        code += "int main() {\n"
        for stmt in ast.statements:
            if not isinstance(stmt, FunctionDef):
                code += self.gen_statement(stmt, 1)
        code += "    return 0;\n}\n"
        
        return code
    
    def gen_function(self, func):
        params_str = ', '.join([f"int {p}" for p in func.params]) if func.params else ""
        code = f"void {func.name}({params_str}) {{\n"
        for stmt in func.body:
            code += self.gen_statement(stmt, 1)
        code += "}\n"
        return code
    
    def cpp_convert_expression(self, expr):
        """Convert C-style expressions to C++ equivalents."""
        import re
        # Replace strlen(var) with var.length() for string variables
        for var_name, var_type in self.variables.items():
            if var_type == 'string':
                # Match strlen ( var ) with flexible spacing
                pattern = r'strlen\s*\(\s*' + re.escape(var_name) + r'\s*\)'
                expr = re.sub(pattern, f'{var_name}.length()', expr)
        return expr
    
    def gen_statement(self, stmt, indent=0):
        ind = "    " * indent
        
        if isinstance(stmt, Declaration):
            c_type = self.convert_type(stmt.var_type)
            self.variables[stmt.variable] = c_type
            if stmt.size:
                return f"{ind}vector<{c_type}> {stmt.variable}({stmt.size});\n"
            return f"{ind}{c_type} {stmt.variable};\n"
        
        elif isinstance(stmt, Assignment):
            expr = self.cpp_convert_expression(stmt.expression)
            # Check if it's an array assignment
            if '[' in stmt.variable and ']' in stmt.variable:
                return f"{ind}{stmt.variable} = {expr};\n"
            else:
                if stmt.variable not in self.variables:
                    self.variables[stmt.variable] = 'int'
                    return f"{ind}int {stmt.variable} = {expr};\n"
                else:
                    return f"{ind}{stmt.variable} = {expr};\n"
        
        elif isinstance(stmt, PrintStatement):
            expr = stmt.expression.strip()
            if expr.startswith('"') and expr.endswith('"'):
                text = expr[1:-1]
                return f'{ind}cout << "{text}" << endl;\n'
            return f'{ind}cout << {expr} << endl;\n'
        
        elif isinstance(stmt, InputStatement):
            var_type = self.variables.get(stmt.variable, 'int')
            if var_type == 'string':
                return f'{ind}getline(cin, {stmt.variable});\n'
            else:
                return f'{ind}cin >> {stmt.variable};\n'
        
        elif isinstance(stmt, IfStatement):
            condition = self.cpp_convert_expression(stmt.condition)
            code = f"{ind}if ({condition}) {{\n"
            for s in stmt.then_block:
                code += self.gen_statement(s, indent + 1)
            code += f"{ind}}}"
            if stmt.else_block:
                code += " else {\n"
                for s in stmt.else_block:
                    code += self.gen_statement(s, indent + 1)
                code += f"{ind}}}"
            code += "\n"
            return code
        
        elif isinstance(stmt, WhileLoop):
            condition = self.cpp_convert_expression(stmt.condition)
            code = f"{ind}while ({condition}) {{\n"
            for s in stmt.body:
                code += self.gen_statement(s, indent + 1)
            code += f"{ind}}}\n"
            return code
        
        elif isinstance(stmt, ForLoop):
            code = f"{ind}for (int {stmt.variable} = {stmt.start}; {stmt.variable} < {stmt.end}; {stmt.variable}++) {{\n"
            for s in stmt.body:
                code += self.gen_statement(s, indent + 1)
            code += f"{ind}}}\n"
            return code
        
        elif isinstance(stmt, ReturnStatement):
            return f"{ind}return {stmt.expression};\n"
        
        elif isinstance(stmt, ContinueStatement):
            return f"{ind}continue;\n"
        
        elif isinstance(stmt, BreakStatement):
            return f"{ind}break;\n"
        
        return ""

# ============================================================
# JAVA CODE GENERATOR
# ============================================================

class JavaCodeGenerator:
    def __init__(self):
        self.variables = {}
        self.functions = []
        self.needs_scanner = False
    
    def convert_type(self, pseudo_type):
        type_map = {
            'integer': 'int', 'int': 'int',
            'float': 'double', 'real': 'double',
            'string': 'String', 'char': 'char',
            'boolean': 'boolean', 'bool': 'boolean'
        }
        return type_map.get(pseudo_type.lower(), 'int')
    
    def generate(self, ast):
        # Pre-scan for input statements
        self.needs_scanner = self._has_input(ast.statements)
        
        for stmt in ast.statements:
            if isinstance(stmt, FunctionDef):
                self.functions.append(stmt)
        
        code = "import java.util.Scanner;\n\n"
        code += "public class Main {\n"
        
        # Generate helper methods
        for func in self.functions:
            code += self.gen_function(func)
            code += "\n"
        
        code += "    public static void main(String[] args) {\n"
        if self.needs_scanner:
            code += "        Scanner scanner = new Scanner(System.in);\n"
        for stmt in ast.statements:
            if not isinstance(stmt, FunctionDef):
                code += self.gen_statement(stmt, 2)
        if self.needs_scanner:
            code += "        scanner.close();\n"
        code += "    }\n"
        code += "}\n"
        
        return code
    
    def _has_input(self, statements):
        for stmt in statements:
            if isinstance(stmt, InputStatement):
                return True
            if isinstance(stmt, IfStatement):
                if self._has_input(stmt.then_block):
                    return True
                if stmt.else_block and self._has_input(stmt.else_block):
                    return True
            if isinstance(stmt, WhileLoop) and self._has_input(stmt.body):
                return True
            if isinstance(stmt, ForLoop) and self._has_input(stmt.body):
                return True
        return False
    
    def gen_function(self, func):
        params_str = ', '.join([f"int {p}" for p in func.params]) if func.params else ""
        code = f"    public static void {func.name}({params_str}) {{\n"
        for stmt in func.body:
            code += self.gen_statement(stmt, 2)
        code += "    }\n"
        return code
    
    def java_convert_expression(self, expr):
        """Convert C-style expressions to Java equivalents."""
        import re
        # Replace strlen(var) with var.length() for String variables
        for var_name, var_type in self.variables.items():
            if var_type == 'String':
                # strlen(var) -> var.length()
                pattern = r'strlen\s*\(\s*' + re.escape(var_name) + r'\s*\)'
                expr = re.sub(pattern, f'{var_name}.length()', expr)
                # var [ i ] -> var.charAt( i ) (array-style access on strings)
                pattern = r'\b' + re.escape(var_name) + r'\s*\[\s*([^\]]+)\s*\]'
                expr = re.sub(pattern, f'{var_name}.charAt(\\1)', expr)
        # C character functions -> Java Character methods
        expr = re.sub(r'\bisspace\s*\(', 'Character.isWhitespace(', expr)
        expr = re.sub(r'\bisdigit\s*\(', 'Character.isDigit(', expr)
        expr = re.sub(r'\bisalpha\s*\(', 'Character.isLetter(', expr)
        expr = re.sub(r'\bisalnum\s*\(', 'Character.isLetterOrDigit(', expr)
        return expr
    
    def gen_statement(self, stmt, indent=0):
        ind = "    " * indent
        
        if isinstance(stmt, Declaration):
            j_type = self.convert_type(stmt.var_type)
            self.variables[stmt.variable] = j_type
            if stmt.size:
                return f"{ind}{j_type}[] {stmt.variable} = new {j_type}[{stmt.size}];\n"
            if j_type == 'String':
                return f"{ind}{j_type} {stmt.variable} = \"\";\n"
            return f"{ind}{j_type} {stmt.variable};\n"
        
        elif isinstance(stmt, Assignment):
            expr = self.java_convert_expression(stmt.expression)
            if '[' in stmt.variable and ']' in stmt.variable:
                return f"{ind}{stmt.variable} = {expr};\n"
            else:
                if stmt.variable not in self.variables:
                    self.variables[stmt.variable] = 'int'
                    return f"{ind}int {stmt.variable} = {expr};\n"
                else:
                    return f"{ind}{stmt.variable} = {expr};\n"
        
        elif isinstance(stmt, PrintStatement):
            expr = stmt.expression.strip()
            if expr.startswith('"') and expr.endswith('"'):
                return f'{ind}System.out.println({expr});\n'
            else:
                return f'{ind}System.out.println({expr});\n'
        
        elif isinstance(stmt, InputStatement):
            if stmt.variable not in self.variables:
                self.variables[stmt.variable] = 'int'
            var_type = self.variables.get(stmt.variable, 'int')
            if var_type == 'String':
                return f'{ind}{stmt.variable} = scanner.nextLine();\n'
            elif var_type == 'double':
                return f'{ind}{stmt.variable} = scanner.nextDouble();\n'
            else:
                return f'{ind}{stmt.variable} = scanner.nextInt();\n'
        
        elif isinstance(stmt, IfStatement):
            condition = self.java_convert_expression(stmt.condition)
            code = f"{ind}if ({condition}) {{\n"
            for s in stmt.then_block:
                code += self.gen_statement(s, indent + 1)
            code += f"{ind}}}"
            if stmt.else_block:
                code += " else {\n"
                for s in stmt.else_block:
                    code += self.gen_statement(s, indent + 1)
                code += f"{ind}}}"
            code += "\n"
            return code
        
        elif isinstance(stmt, WhileLoop):
            condition = self.java_convert_expression(stmt.condition)
            code = f"{ind}while ({condition}) {{\n"
            for s in stmt.body:
                code += self.gen_statement(s, indent + 1)
            code += f"{ind}}}\n"
            return code
        
        elif isinstance(stmt, ForLoop):
            code = f"{ind}for (int {stmt.variable} = {stmt.start}; {stmt.variable} < {stmt.end}; {stmt.variable}++) {{\n"
            for s in stmt.body:
                code += self.gen_statement(s, indent + 1)
            code += f"{ind}}}\n"
            return code
        
        elif isinstance(stmt, ReturnStatement):
            return f"{ind}return {stmt.expression};\n"
        
        elif isinstance(stmt, ContinueStatement):
            return f"{ind}continue;\n"
        
        elif isinstance(stmt, BreakStatement):
            return f"{ind}break;\n"
        
        return ""

# ============================================================
# COMPILER
# ============================================================

class Compiler:
    def __init__(self):
        self.lexer = Lexer()
    
    def compile(self, source_code, target_language):
        try:
            tokens = self.lexer.tokenize(source_code)
            if self.lexer.errors:
                return None, "Lexical Errors:\n" + "\n".join(self.lexer.errors), []
            
            parser = Parser(tokens)
            ast = parser.parse()
            if parser.errors:
                return None, "Parse Errors:\n" + "\n".join(parser.errors), tokens
            
            # Create fresh generator for each compilation
            generators = {
                'C': CCodeGenerator(),
                'C++': CPPCodeGenerator(),
                'Java': JavaCodeGenerator()
            }
            
            if target_language in generators:
                generator = generators[target_language]
                code = generator.generate(ast)
                return code, None, tokens
            else:
                return None, f"Unsupported language: {target_language}", tokens
        except Exception as e:
            return None, f"Compilation Error: {str(e)}", []
    
    def format_tokens(self, tokens):
        """Format tokens into a readable table string."""
        lines = []
        lines.append(f"{'#':<6}{'TOKEN TYPE':<20}{'VALUE':<20}{'LINE':<8}{'COL':<8}")
        lines.append("─" * 62)
        for i, token in enumerate(tokens, 1):
            value = repr(token.value) if token.type in (TokenType.STRING, TokenType.NEWLINE) else token.value
            if len(value) > 18:
                value = value[:15] + "..."
            lines.append(f"{i:<6}{token.type.value:<20}{value:<20}{token.line:<8}{token.column:<8}")
        lines.append("─" * 62)
        lines.append(f"Total tokens: {len(tokens)}")
        return "\n".join(lines)

# ============================================================
# GUI
# ============================================================

class CompilerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Complete Auto Source Code Generator - Production Version")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        self.compiler = Compiler()
        self.current_language = tk.StringVar(value="C")
        
        self.setup_gui()
    
    def setup_gui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#34495e', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title = tk.Label(title_frame, text="🚀 Complete Auto Source Code Generator",
                        font=('Arial', 22, 'bold'), bg='#34495e', fg='white')
        title.pack(pady=20)
        
        # Main
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left Panel
        left_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        tk.Label(left_frame, text="📝 Pseudo Code Input",
                font=('Arial', 14, 'bold'), bg='#34495e', fg='white').pack(pady=10)
        
        self.input_text = scrolledtext.ScrolledText(
            left_frame, font=('Consolas', 11), wrap=tk.WORD,
            bg='#ecf0f1', fg='#2c3e50', insertbackground='#2c3e50'
        )
        self.input_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Sample
        sample = """start

declare integer year
declare integer isLeap

set year to 2024
set isLeap to 0

print "Checking Leap Year"
print year

if year % 4 == 0 then
    if year % 100 == 0 then
        if year % 400 == 0 then
            set isLeap to 1
        endif
    else
        set isLeap to 1
    endif
endif

if isLeap == 1 then
    print "Year is a LEAP YEAR"
else
    print "Year is NOT a leap year"
endif

end"""
        self.input_text.insert('1.0', sample)
        
        # Right Panel with Tabs
        right_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        right_frame.pack(side='right', fill='both', expand=True)
        
        tk.Label(right_frame, text="💻 Output",
                font=('Arial', 14, 'bold'), bg='#34495e', fg='white').pack(pady=10)
        
        # Create Notebook (tabs)
        style = ttk.Style()
        style.configure('Custom.TNotebook', background='#34495e')
        style.configure('Custom.TNotebook.Tab', font=('Arial', 11, 'bold'), padding=[12, 6])
        
        self.notebook = ttk.Notebook(right_frame, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Tab 1: Generated Code
        code_tab = tk.Frame(self.notebook, bg='#ecf0f1')
        self.notebook.add(code_tab, text='📄 Generated Code')
        
        self.output_text = scrolledtext.ScrolledText(
            code_tab, font=('Consolas', 10), wrap=tk.WORD,
            bg='#ecf0f1', fg='#2c3e50', state='disabled'
        )
        self.output_text.pack(fill='both', expand=True)
        
        # Tab 2: Tokens
        token_tab = tk.Frame(self.notebook, bg='#ecf0f1')
        self.notebook.add(token_tab, text='🔤 Tokens')
        
        self.token_text = scrolledtext.ScrolledText(
            token_tab, font=('Consolas', 10), wrap=tk.NONE,
            bg='#1e1e2e', fg='#a6e3a1', state='disabled'
        )
        self.token_text.pack(fill='both', expand=True)
        
        # Controls
        control_frame = tk.Frame(self.root, bg='#34495e', height=100)
        control_frame.pack(fill='x', padx=20, pady=(0, 20))
        control_frame.pack_propagate(False)
        
        # Language Selection
        lang_frame = tk.Frame(control_frame, bg='#34495e')
        lang_frame.pack(side='left', padx=20, pady=20)
        
        tk.Label(lang_frame, text="Select Output Language:",
                font=('Arial', 11, 'bold'), bg='#34495e', fg='white').pack(side='left', padx=10)
        
        for lang in ['C', 'C++', 'Java']:
            rb = tk.Radiobutton(
                lang_frame, text=lang, variable=self.current_language,
                value=lang, font=('Arial', 11), bg='#34495e', fg='white',
                selectcolor='#3498db', activebackground='#34495e',
                activeforeground='white'
            )
            rb.pack(side='left', padx=5)
        
        # Buttons
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(side='right', padx=20, pady=20)
        
        btn_style = {'font': ('Arial', 11, 'bold'), 'width': 12, 'height': 2}
        
        tk.Button(button_frame, text="🔄 Compile", bg='#27ae60',
                 fg='white', command=self.compile_code, **btn_style).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="🗑️ Clear", bg='#e74c3c',
                 fg='white', command=self.clear_all, **btn_style).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="💾 Save", bg='#3498db',
                 fg='white', command=self.save_output, **btn_style).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="📂 Load", bg='#9b59b6',
                 fg='white', command=self.load_file, **btn_style).pack(side='left', padx=5)
    
    def compile_code(self):
        source = self.input_text.get('1.0', tk.END).strip()
        if not source:
            messagebox.showwarning("Warning", "Please enter pseudo code!")
            return
        
        target_lang = self.current_language.get()
        
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert('1.0', "Compiling...")
        self.output_text.config(state='disabled')
        self.root.update()
        
        code, error, tokens = self.compiler.compile(source, target_lang)
        
        # Display tokens in Tokens tab
        self.token_text.config(state='normal')
        self.token_text.delete('1.0', tk.END)
        if tokens:
            token_output = self.compiler.format_tokens(tokens)
            self.token_text.insert('1.0', token_output)
        else:
            self.token_text.insert('1.0', "No tokens generated.")
        self.token_text.config(state='disabled')
        
        # Display generated code in Code tab
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', tk.END)
        
        if error:
            self.output_text.insert('1.0', f"❌ COMPILATION FAILED!\n\n{error}")
            self.output_text.tag_add("error", "1.0", "2.0")
            self.output_text.tag_config("error", foreground="red", font=('Arial', 12, 'bold'))
            messagebox.showerror("Compilation Error", error)
        else:
            self.output_text.insert('1.0', code)
            messagebox.showinfo("Success", f"✅ Successfully compiled to {target_lang}!")
        
        self.output_text.config(state='disabled')
    
    def clear_all(self):
        self.input_text.delete('1.0', tk.END)
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', tk.END)
        self.output_text.config(state='disabled')
        self.token_text.config(state='normal')
        self.token_text.delete('1.0', tk.END)
        self.token_text.config(state='disabled')
    
    def save_output(self):
        content = self.output_text.get('1.0', tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No output to save!")
            return
        
        lang = self.current_language.get()
        extensions = {'C': '.c', 'C++': '.cpp', 'Java': '.java'}
        
        filename = filedialog.asksaveasfilename(
            defaultextension=extensions[lang],
            filetypes=[
                (f"{lang} files", f"*{extensions[lang]}"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(content)
            messagebox.showinfo("Success", f"File saved to {filename}")
    
    def load_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            with open(filename, 'r') as f:
                content = f.read()
            self.input_text.delete('1.0', tk.END)
            self.input_text.insert('1.0', content)
            messagebox.showinfo("Success", "File loaded successfully!")

def main():
    root = tk.Tk()
    app = CompilerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
