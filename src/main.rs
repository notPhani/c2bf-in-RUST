use std::vec;

#[derive(Debug, Clone, PartialEq)]
enum Keywords { Int, Char, String, Arr, If, Else, For, While, PutChar, GetChar, Return, Function, Break, Continue }

#[derive(Debug, Clone, PartialEq)]
enum Operations { Add, Subtract, Multiply, Divide, Modulus, Equal, NotEqual, GreaterThan, LessThan, GreaterThanOrEqual, LessThanOrEqual, And, Or, Not, Assign, Increment, Decrement, BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, LeftShift, RightShift, Ternary }

#[derive(Debug, Clone, PartialEq)]
enum Punctuators { Semicolon, Colon, Comma, Dot, LeftParen, RightParen, LeftBrace, RightBrace, LeftBracket, RightBracket }

#[derive(Debug, Clone, PartialEq)]
enum Token { Keyword(Keywords), Identifier(String), Number(i64), StringLiteral(String), CharLiteral(char), Operator(Operations), Punctuator(Punctuators) }

fn tokenize(source: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut iter = source.chars().peekable();

    while let Some(&c) = iter.peek() {
        // Skip whitespace
        if c.is_whitespace() { iter.next(); continue; }

        // Numbers
        if c.is_ascii_digit() {
            let mut num = 0i64;
            while let Some(&d) = iter.peek() {
                if d.is_ascii_digit() { num = num * 10 + (d as i64 - '0' as i64); iter.next(); } 
                else { break; }
            }
            tokens.push(Token::Number(num));
            continue;
        }

        // Identifiers or Keywords
        if c.is_alphabetic() || c == '_' {
            let mut ident = String::new();
            while let Some(&ch) = iter.peek() {
                if ch.is_alphanumeric() || ch == '_' { ident.push(ch); iter.next(); } 
                else { break; }
            }
            let keyword = match ident.as_str() {
                "int" => Some(Keywords::Int), "char" => Some(Keywords::Char), "string" => Some(Keywords::String), 
                "arr" => Some(Keywords::Arr), "if" => Some(Keywords::If), "else" => Some(Keywords::Else), 
                "for" => Some(Keywords::For), "while" => Some(Keywords::While), "putchar" => Some(Keywords::PutChar), 
                "getchar" => Some(Keywords::GetChar), "return" => Some(Keywords::Return), "function" => Some(Keywords::Function),
                "break" => Some(Keywords::Break), "continue" => Some(Keywords::Continue), _ => None,
            };
            tokens.push(if let Some(k) = keyword { Token::Keyword(k) } else { Token::Identifier(ident) });
            continue;
        }

        // String literal
        if c == '"' { iter.next(); let mut val = String::new(); while let Some(&ch) = iter.peek() { if ch == '"' { iter.next(); break; } val.push(ch); iter.next(); } tokens.push(Token::StringLiteral(val)); continue; }

        // Char literal
        if c == '\'' { iter.next(); let ch = iter.next().unwrap_or('\0'); if iter.peek() == Some(&'\'') { iter.next(); } tokens.push(Token::CharLiteral(ch)); continue; }

        // Punctuators
        let punct = match c {
            ';' => Some(Punctuators::Semicolon), ':' => Some(Punctuators::Colon), ',' => Some(Punctuators::Comma),
            '.' => Some(Punctuators::Dot), '(' => Some(Punctuators::LeftParen), ')' => Some(Punctuators::RightParen),
            '{' => Some(Punctuators::LeftBrace), '}' => Some(Punctuators::RightBrace),
            '[' => Some(Punctuators::LeftBracket), ']' => Some(Punctuators::RightBracket), _ => None,
        };
        if let Some(p) = punct { tokens.push(Token::Punctuator(p)); iter.next(); continue; }

        // Operators
        let mut op_token = None;
        match c {
            '+' => { iter.next(); op_token = Some(if iter.peek() == Some(&'+') { iter.next(); Operations::Increment } else { Operations::Add }); }
            '-' => { iter.next(); op_token = Some(if iter.peek() == Some(&'-') { iter.next(); Operations::Decrement } else { Operations::Subtract }); }
            '*' => { iter.next(); op_token = Some(Operations::Multiply); }
            '/' => {if let Some(&next) = iter.peek() {
                    if next == '/' { while let Some(&c) = iter.peek() { if c == '\n' { break; } iter.next(); } continue; }
                    else if next == '*' { iter.next(); while let Some(c) = iter.next() { if c == '*' { if let Some(&'/') = iter.peek() { iter.next(); break; } } } continue; }
                    else { op_token = Some(Operations::Divide); }
                    } else { op_token = Some(Operations::Divide); }
                    }
            '%' => { iter.next(); op_token = Some(Operations::Modulus); }
            '=' => { iter.next(); op_token = Some(if iter.peek() == Some(&'=') { iter.next(); Operations::Equal } else { Operations::Assign }); }
            '!' => { iter.next(); op_token = Some(if iter.peek() == Some(&'=') { iter.next(); Operations::NotEqual } else { Operations::Not }); }
            '>' => { iter.next(); op_token = Some(if iter.peek() == Some(&'=') { iter.next(); Operations::GreaterThanOrEqual } else if iter.peek() == Some(&'>') { iter.next(); Operations::RightShift } else { Operations::GreaterThan }); }
            '<' => { iter.next(); op_token = Some(if iter.peek() == Some(&'=') { iter.next(); Operations::LessThanOrEqual } else if iter.peek() == Some(&'<') { iter.next(); Operations::LeftShift } else { Operations::LessThan }); }
            '&' => { iter.next(); op_token = Some(if iter.peek() == Some(&'&') { iter.next(); Operations::And } else { Operations::BitwiseAnd }); }
            '|' => { iter.next(); op_token = Some(if iter.peek() == Some(&'|') { iter.next(); Operations::Or } else { Operations::BitwiseOr }); }
            '^' => { iter.next(); op_token = Some(Operations::BitwiseXor); }
            '~' => { iter.next(); op_token = Some(Operations::BitwiseNot); }
            '?' => { iter.next(); op_token = Some(Operations::Ternary); }
            _ => { iter.next(); }
        }
        if let Some(op) = op_token { tokens.push(Token::Operator(op)); }
    }

    tokens
}

// ------------------------------------------Parser------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum ASTNode {
    Program(Vec<ASTNode>),
    Function { return_type: Keywords, name: String, params: Vec<(Keywords, String)>, body: Box<ASTNode> },//Done
    VariableDeclaration { var_type: Keywords, name: String, array_dims: Option<Vec<usize>>, initial_value: Option<Box<ASTNode>> },//Done
    Assignment { target: Box<ASTNode>, value: Box<ASTNode> },
    ArrayElement { name: String, index: Box<ASTNode> },
    If { condition: Box<ASTNode>, then_branch: Box<ASTNode>, else_branch: Option<Box<ASTNode>> },
    While { condition: Box<ASTNode>, body: Box<ASTNode> },
    For { init: Box<ASTNode>, condition: Box<ASTNode>, increment: Box<ASTNode>, body: Box<ASTNode> },
    Return(Option<Box<ASTNode>>),
    Break,
    Continue,
    BinaryOp { op: Operations, left: Box<ASTNode>, right: Box<ASTNode> },
    UnaryOp { op: Operations, expr: Box<ASTNode> },
    TernaryOp { condition: Box<ASTNode>, true_expr: Box<ASTNode>, false_expr: Box<ASTNode> },
    FunctionCall { name: String, args: Vec<ASTNode> },
    LiteralInt(i64),//done
    LiteralChar(char),//done
    LiteralString(String),//done
    Identifier(String),//done
    ArrayAccess { name: String, index: Box<ASTNode> },
    Block(Vec<ASTNode>),
    Empty,
}

fn parse_literal(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    if tokens.is_empty() {
        return None;
    }

    let token = tokens[0].clone(); // take a clone first
    tokens.remove(0);              // now safe to mutate

    match token {
        Token::Number(n) => Some(ASTNode::LiteralInt(n)),
        Token::CharLiteral(c) => Some(ASTNode::LiteralChar(c)),
        Token::StringLiteral(s) => Some(ASTNode::LiteralString(s)),
        _ => None,
    }
}

fn parse_identifier(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    if let Some(Token::Identifier(name)) = tokens.first() {
        let name = name.clone();
        tokens.remove(0);
        Some(ASTNode::Identifier(name))
    } else {
        None
    }
}
