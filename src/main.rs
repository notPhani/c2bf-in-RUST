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
    Function { return_type: Keywords, name: String, params: Vec<(Keywords, String)>, body: Box<ASTNode> },
    VariableDeclaration { var_type: Keywords, name: String, array_dims: Option<Vec<usize>>, initial_value: Option<Box<ASTNode>> },
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
    LiteralInt(i64),
    LiteralChar(char),
    LiteralString(String),
    Identifier(String),
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

fn parse_dec_or_func(tokens: &mut Vec<Token>) -> Option<Vec<ASTNode>> {
    if tokens.len() < 2 {
        return None;
    }

    let var_type = match tokens.first() {
        Some(Token::Keyword(k)) => k.clone(),
        _ => return None,
    };
    tokens.remove(0);

    let name = match tokens.first() {
        Some(Token::Identifier(n)) => n.clone(),
        _ => return None,
    };
    tokens.remove(0);

    let nodes = match tokens.first() {
        Some(Token::Operator(Operations::Assign)) => {
            tokens.remove(0); // consume '='
            let init = parse_literal(tokens).map(Box::new);
            if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::Semicolon))) {
                tokens.remove(0);
            }
            vec![ASTNode::VariableDeclaration {
                var_type,
                name,
                array_dims: None,
                initial_value: init,
            }]
        },
        Some(Token::Punctuator(Punctuators::LeftBracket)) => {
            tokens.remove(0); // consume '['
            let arr_size = if let Some(Token::Number(n)) = tokens.first() {
                let size = *n as usize;
                tokens.remove(0);
                size
            } else { 0 };
            if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightBracket))) {
                tokens.remove(0);
            }
            let init = if matches!(tokens.first(), Some(Token::Operator(Operations::Assign))) {
                tokens.remove(0);
                parse_literal(tokens).map(Box::new)
            } else { None };
            if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::Semicolon))) {
                tokens.remove(0);
            }
            vec![ASTNode::VariableDeclaration {
                var_type,
                name,
                array_dims: Some(vec![arr_size]),
                initial_value: init,
            }]
        },
        Some(Token::Punctuator(Punctuators::LeftParen)) => {
            tokens.remove(0); // consume '('
            let params = parse_params(tokens);
            if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightParen))) {
                tokens.remove(0);
            }
            let body = if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::LeftBrace))) {
                parse_block(tokens)
            } else {
                Box::new(ASTNode::Empty)
            };
            vec![ASTNode::Function {
                return_type: var_type,
                name,
                params,
                body,
            }]
        },
        Some(Token::Punctuator(Punctuators::Semicolon)) => {
            tokens.remove(0);
            vec![ASTNode::VariableDeclaration {
                var_type,
                name,
                array_dims: None,
                initial_value: None,
            }]
        },
        _ => { return None; }
    };

    Some(nodes)
}

fn parse_params(tokens: &mut Vec<Token>) -> Vec<(Keywords, String)> {
    let mut params = Vec::new();

    while !tokens.is_empty() {
        // Check for closing paren
        if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightParen))) {
            break;
        }

        // Parse type
        let param_type = match tokens.first() {
            Some(Token::Keyword(k)) => k.clone(),
            _ => break,
        };
        tokens.remove(0);

        // Parse identifier
        let param_name = match tokens.first() {
            Some(Token::Identifier(n)) => n.clone(),
            _ => break,
        };
        tokens.remove(0);

        params.push((param_type, param_name));

        // Consume comma if there is one
        if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::Comma))) {
            tokens.remove(0);
        }
    }

    params
}

fn parse_block(tokens: &mut Vec<Token>) -> Box<ASTNode> {
    if !matches!(tokens.first(), Some(Token::Punctuator(Punctuators::LeftBrace))) {
        return Box::new(ASTNode::Empty);
    }
    tokens.remove(0); // consume '{'

    let mut stmts = Vec::new();

    while !tokens.is_empty() {
        if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightBrace))) {
            tokens.remove(0); // consume '}'
            break;
        }

        // Here we can call parse_dec_or_func, parse_statement, etc.
        // For now, let's just handle literals & identifiers as placeholders
        if let Some(node) = parse_literal(tokens).or_else(|| parse_identifier(tokens)) {
            stmts.push(node);
        } else {
            // Skip unrecognized token to avoid infinite loop
            tokens.remove(0);
        }
    }

    Box::new(ASTNode::Block(stmts))
}
fn parse_program(tokens: &mut Vec<Token>) -> ASTNode {
    let mut nodes = Vec::new();

    while !tokens.is_empty() {
        // try declaration/function
        if let Some(mut decls) = parse_dec_or_func(tokens) {
            nodes.append(&mut decls);
            continue;
        }

        // try literal or identifier (placeholder)
        if let Some(node) = parse_literal(tokens).or_else(|| parse_identifier(tokens)) {
            nodes.push(node);
            continue;
        }

        // fallback: skip unrecognized token
        tokens.remove(0);
    }

    ASTNode::Program(nodes)
}

fn main() {
    let source_code = r#"
    int main() {
        int a = 5;
        int b = 10;
        if (a < b) {
            putchar('A');
        } else {
            putchar('B');
        }
        return 0;
    }
    "#;

    let mut tokens = tokenize(source_code);

    println!("--- Tokens ---");
    for token in &tokens {
        println!("{:?}", token);
    }

    println!("\n--- AST ---");
    let ast = parse_program(&mut tokens);
    println!("{:#?}", ast);
}
