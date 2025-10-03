use std::{collections::HashMap};

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
                "ARRAY" => Some(Keywords::Arr), "if" => Some(Keywords::If), "else" => Some(Keywords::Else), 
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
    //ScopeLabel{label :String}, // This will serve no purpose in the AST but will help in code generation and semen anal, Scrapped idea
    Program(Vec<ASTNode>),//yet to implement
    Function { return_type: Keywords, name: String, params: Vec<(Keywords, String)>, body: Box<ASTNode> },
    VariableDeclaration { var_type: Keywords, name: String, array_dims: Option<Vec<usize>>, initial_value: Option<Box<ASTNode>> },
    For { init: Box<ASTNode>, condition: Box<ASTNode>, increment: Box<ASTNode>, body: Box<ASTNode> },
    Return(Option<Box<ASTNode>>),
    While { condition: Box<ASTNode>, body: Box<ASTNode> },
    If { condition: Box<ASTNode>, then_branch: Box<ASTNode>, else_branch: Option<Box<ASTNode>> },
    PutChar{ expr: Box<ASTNode> },
    Break,
    Continue,
    Assignment { target: Box<ASTNode>, value: Box<ASTNode> },
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
    let token = tokens.remove(0);
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


// I got few ideas for the parser for the expression : 
// We are gonna do the simple pratt parser but with some modifications
// We will fold the expressions if all the expressions are constant literals
// We will switch to the actual pratt parser if we encounter an identifier or a function call
// We will also handle the ternatry operator here
// We will write a seperate function for parsing unary operations because bruh x = ++x doesn't seem right atleast for C language
// The best thing to do here is to check the expression and run through all the literals and operators and then build the AST from that
// We will try to calculate the value of the expression if all the literals are constant,
// If we hit a error we can leave the expression as it is and return the AST as it is
// This will help in constant folding and optimization
// We will do seperate functions for parsing binary and unary operations and combine them in the parse_expression function

/* 
fn parse_unary_operation(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    if tokens.is_empty() { return None; }

    let op = match tokens.first()? {
        Token::Operator(Operations::Increment) => Operations::Increment,
        Token::Operator(Operations::Decrement) => Operations::Decrement,
        Token::Operator(Operations::Not) => Operations::Not,
        Token::Operator(Operations::Subtract) => Operations::Subtract,
        Token::Operator(Operations::BitwiseNot) => Operations::BitwiseNot,
        _ => return None,
    };

    tokens.remove(0); // consume the operator
    let expr = parse_expression(tokens)?; // recursive call
    Some(ASTNode::UnaryOp { op, expr: Box::new(expr[0].clone()) })
}
*/

fn parse_ternary_operation(tokens: &mut Vec<Token>, condition: ASTNode) -> Option<ASTNode> {
    if let Some(Token::Operator(Operations::Ternary)) = tokens.first() {
        tokens.remove(0); // consume '?'
        let true_expr = parse_expression(tokens)?;
        if let Some(Token::Punctuator(Punctuators::Colon)) = tokens.first() {
            tokens.remove(0); // consume ':'
            let false_expr = parse_expression(tokens)?;
            Some(ASTNode::TernaryOp {
                condition: Box::new(condition),
                true_expr: Box::new(true_expr),
                false_expr: Box::new(false_expr),
            })
        } else {
            None // error: expected ':'
        }
    } else {
        Some(condition) // not ternary, return condition untouched
    }
}


fn get_precedence(op: &Operations) -> u8 {
    match op {
        Operations::Assign => 1,
        Operations::Or => 2,
        Operations::And => 3,
        Operations::BitwiseOr => 4,
        Operations::BitwiseXor => 5,
        Operations::BitwiseAnd => 6,
        Operations::Equal | Operations::NotEqual => 7,
        Operations::GreaterThan | Operations::LessThan |
        Operations::GreaterThanOrEqual | Operations::LessThanOrEqual => 8,
        Operations::Add | Operations::Subtract => 9,
        Operations::Multiply | Operations::Divide | Operations::Modulus => 10,
        Operations::LeftShift | Operations::RightShift => 11,
        _ => 0, // Ternary, unary, postfix handled separately
    }
}

fn parse_primary(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    if tokens.is_empty() { return None; }

    let mut node = match tokens.first()? {
        Token::Punctuator(Punctuators::LeftParen) => {
            tokens.remove(0); // consume '('
            let expr = parse_expression(tokens)?;
            if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightParen))) {
                tokens.remove(0); // consume ')'
                expr
            } else {
                return None;
            }
        },
        Token::Operator(op) if matches!(op, Operations::Subtract | Operations::Not | Operations::BitwiseNot) => {
            let op = if let Token::Operator(o) = tokens.remove(0) { o } else { unreachable!() };
            let expr = parse_primary(tokens)?;
            ASTNode::UnaryOp { op, expr: Box::new(expr) }
        },
        Token::Number(_) | Token::CharLiteral(_) | Token::StringLiteral(_) => parse_literal(tokens)?,
        Token::Identifier(_) => parse_identifier(tokens)?,
        _ => return None,
    };

    // postfix: array access + function call
    loop {
        match tokens.first() {
            Some(Token::Punctuator(Punctuators::LeftBracket)) => {
                tokens.remove(0);
                let index = parse_expression(tokens)?;
                if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightBracket))) {
                    tokens.remove(0);
                    if let ASTNode::Identifier(name) = node {
                        node = ASTNode::ArrayAccess { name, index: Box::new(index) };
                    } else {
                        return None;
                    }
                } else { return None; }
            },
            Some(Token::Punctuator(Punctuators::LeftParen)) => {
                tokens.remove(0);
                let mut args = Vec::new();
                if !matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightParen))) {
                    loop {
                        let expr = parse_expression(tokens)?;
                        args.push(expr);
                        if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightParen))) {
                            break;
                        } else if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::Comma))) {
                            tokens.remove(0);
                        } else {
                            return None;
                        }
                    }
                }
                if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightParen))) {
                    tokens.remove(0);
                    if let ASTNode::Identifier(name) = node {
                        node = ASTNode::FunctionCall { name, args };
                    } else {
                        return None;
                    }
                } else { return None; }
            },
            _ => break,
        }
    }
    Some(node)
}



fn parse_binary_ops(tokens: &mut Vec<Token>, min_prec: u8) -> Option<ASTNode> {
    let mut left = parse_primary(tokens)?;

    while let Some(Token::Operator(op)) = tokens.first() {
        let prec = get_precedence(op);
        if prec < min_prec { break; }

        let op = if let Token::Operator(op) = tokens.remove(0) { op.clone() } else { unreachable!() };
        let mut right = parse_binary_ops(tokens, prec + 1)?;

        // ternary after binary
        if let Some(Token::Operator(Operations::Ternary)) = tokens.first() {
            let tern = parse_ternary_operation(tokens, right)?;
            right = tern.clone();
        }

        left = ASTNode::BinaryOp { op, left: Box::new(left), right: Box::new(right) };
    }

    Some(left)
}

fn is_foldable(node: &ASTNode) -> bool {
    match node {
        // Pure literals - always foldable
        ASTNode::LiteralInt(_) | ASTNode::LiteralChar(_) | ASTNode::LiteralString(_) => true,
        
        // Runtime values - never foldable
        ASTNode::Identifier(_) | ASTNode::FunctionCall { .. } | ASTNode::ArrayAccess { .. } => false,
        
        // Compound expressions - recursively check children
        ASTNode::UnaryOp { expr, .. } => is_foldable(expr),
        
        ASTNode::BinaryOp { left, right, .. } => {
            is_foldable(left) && is_foldable(right)
        },
        
        ASTNode::TernaryOp { condition, true_expr, false_expr, .. } => {
            is_foldable(condition) && is_foldable(true_expr) && is_foldable(false_expr)
        },
        _ => false, // Other nodes are not foldable
    }
}
fn eval_tree(node: ASTNode) -> Option<ASTNode> {
    match node {
        // Already literals - return as-is
        ASTNode::LiteralInt(_) | ASTNode::LiteralChar(_) | ASTNode::LiteralString(_) => Some(node),
        
        // Unary operations
        ASTNode::UnaryOp { op, expr } => {
            let eval_expr = eval_tree(*expr)?;
            match (op, eval_expr) {
                (Operations::Subtract, ASTNode::LiteralInt(n)) => Some(ASTNode::LiteralInt(-n)),
                (Operations::Not, ASTNode::LiteralInt(n)) => Some(ASTNode::LiteralInt(if n == 0 { 1 } else { 0 })),
                (Operations::BitwiseNot, ASTNode::LiteralInt(n)) => Some(ASTNode::LiteralInt(!n)),
                _ => None,
            }
        },
        
        // Binary operations
        ASTNode::BinaryOp { op, left, right } => {
            let eval_left = eval_tree(*left)?;
            let eval_right = eval_tree(*right)?;
            
            match (op, eval_left, eval_right) {
                (Operations::Add, ASTNode::LiteralInt(a), ASTNode::LiteralInt(b)) => Some(ASTNode::LiteralInt(a + b)),
                (Operations::Subtract, ASTNode::LiteralInt(a), ASTNode::LiteralInt(b)) => Some(ASTNode::LiteralInt(a - b)),
                (Operations::Multiply, ASTNode::LiteralInt(a), ASTNode::LiteralInt(b)) => Some(ASTNode::LiteralInt(a * b)),
                (Operations::Divide, ASTNode::LiteralInt(a), ASTNode::LiteralInt(b)) if b != 0 => Some(ASTNode::LiteralInt(a / b)),
                (Operations::Modulus, ASTNode::LiteralInt(a), ASTNode::LiteralInt(b)) if b != 0 => Some(ASTNode::LiteralInt(a % b)),
                (Operations::GreaterThan, ASTNode::LiteralInt(a), ASTNode::LiteralInt(b)) => Some(ASTNode::LiteralInt(if a > b { 1 } else { 0 })),
                (Operations::LessThan, ASTNode::LiteralInt(a), ASTNode::LiteralInt(b)) => Some(ASTNode::LiteralInt(if a < b { 1 } else { 0 })),
                _ => None,
            }
        },
        
        // Ternary operations
        ASTNode::TernaryOp { condition, true_expr, false_expr } => {
            let eval_condition = eval_tree(*condition)?;
            match eval_condition {
                ASTNode::LiteralInt(n) => {
                    if n != 0 { eval_tree(*true_expr) } else { eval_tree(*false_expr) }
                },
                _ => None,
            }
        },
        
        _ => None,
    }
}

fn parse_expression(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    let mut node = parse_binary_ops(tokens, 1)?;
    if let Some(Token::Operator(Operations::Ternary)) = tokens.first() {
        let tern = parse_ternary_operation(tokens, node)?;
        node = tern.clone();
    }
    if is_foldable(&node) {
        if let Some(folded_node) = eval_tree(node.clone()) {
            println!("Folded expression to: {:?}", folded_node);
            node = folded_node;
        }
    }
    Some(node)
}

//Phew mf that was a lot of work
//Now we write assignment parsing MF this is a lot of work too
// In parse_assignment, make sure assignments go through semantic analysis
fn parse_assignment(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    let left = parse_expression(tokens)?;
    
    if let Some(Token::Operator(Operations::Assign)) = tokens.first() {
        println!("DEBUG: Found assignment operator");
        tokens.remove(0);
        match left {
            ASTNode::Identifier(_) | ASTNode::ArrayAccess { .. } => {
                let right = parse_assignment(tokens)?;
                println!("DEBUG: Creating Assignment node");
                Some(ASTNode::Assignment { 
                    target: Box::new(left), 
                    value: Box::new(right)
                })
            },
            _ => {
                println!("DEBUG: Invalid assignment target");
                None
            }
        }
    } else {
        Some(left)
    }
}




fn parse_block(tokens: &mut Vec<Token>)->Option<ASTNode>{
    if tokens.first() != Some(&Token::Punctuator(Punctuators::LeftBrace)){return None;}
    tokens.remove(0);
    let mut statements = Vec::new();
    while tokens.first() != Some(&Token::Punctuator(Punctuators::RightBrace)) && !tokens.is_empty(){
        if let Some(stmt) = parse_statement(tokens){
            statements.push(stmt);
        } else {
            return None; // Error parsing statement
        }
    }
    if tokens.first() != Some(&Token::Punctuator(Punctuators::RightBrace)) { return None; }
    tokens.remove(0);
    
    Some(ASTNode::Block(statements))
}

fn parse_if(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    if tokens.first()? != &Token::Keyword(Keywords::If) { return None; }
    tokens.remove(0); // consume "if"

    if tokens.first()? != &Token::Punctuator(Punctuators::LeftParen) { return None; }
    tokens.remove(0); // consume "("
    let condition = parse_expression(tokens)?;
    if tokens.first()? != &Token::Punctuator(Punctuators::RightParen) { return None; }
    tokens.remove(0); // consume ")"

    let then_branch = parse_statement(tokens)?;
    let else_branch = if tokens.first() == Some(&Token::Keyword(Keywords::Else)) {
        tokens.remove(0); // consume "else"
        Some(Box::new(parse_statement(tokens)?))
    } else {
        None
    };

    Some(ASTNode::If {
        condition: Box::new(condition),
        then_branch: Box::new(then_branch),
        else_branch,
    })
}

fn parse_while(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    if tokens.first()? != &Token::Keyword(Keywords::While) { return None; }
    tokens.remove(0); // consume "while"

    if tokens.first()? != &Token::Punctuator(Punctuators::LeftParen) { return None; }
    tokens.remove(0); // consume "("
    let condition = parse_expression(tokens)?;
    if tokens.first()? != &Token::Punctuator(Punctuators::RightParen) { return None; }
    tokens.remove(0); // consume ")"

    let body = parse_statement(tokens)?;
    Some(ASTNode::While { condition: Box::new(condition), body: Box::new(body) })
}

fn parse_return(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    if tokens.first()? != &Token::Keyword(Keywords::Return) { return None; }
    tokens.remove(0); // consume "return"

    let expr = if tokens.first()? != &Token::Punctuator(Punctuators::Semicolon) {
        Some(Box::new(parse_expression(tokens)?))
    } else {
        None
    };

    if tokens.first()? == &Token::Punctuator(Punctuators::Semicolon) {
        tokens.remove(0); // consume ";"
    }
    Some(ASTNode::Return(expr))
}


fn parse_for(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    if tokens.first()? != &Token::Keyword(Keywords::For) { return None; }
    tokens.remove(0); // consume "for"

    if tokens.first()? != &Token::Punctuator(Punctuators::LeftParen) { return None; }
    tokens.remove(0); // consume "("

    let init = parse_statement(tokens)?;
    let condition = parse_expression(tokens)?;
    if tokens.first()? != &Token::Punctuator(Punctuators::Semicolon) { return None; }
    tokens.remove(0); // consume ";"
    let increment = parse_expression(tokens)?;
    if tokens.first()? != &Token::Punctuator(Punctuators::RightParen) { return None; }
    tokens.remove(0); // consume ")"

    let body = parse_statement(tokens)?;
    Some(ASTNode::For {
        init: Box::new(init),
        condition: Box::new(condition),
        increment: Box::new(increment),
        body: Box::new(body),
    })
}


fn parse_statement(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    match tokens.first()? {
        Token::Keyword(Keywords::If) => parse_if(tokens),
        Token::Keyword(Keywords::While) => parse_while(tokens),
        Token::Keyword(Keywords::For) => parse_for(tokens),
        Token::Keyword(Keywords::Return) => parse_return(tokens),
        Token::Keyword(Keywords::Break) => { 
            tokens.remove(0);
            if tokens.first() == Some(&Token::Punctuator(Punctuators::Semicolon)) {
                tokens.remove(0);
            }
            Some(ASTNode::Break) 
        },
        Token::Keyword(Keywords::Continue) => { 
            tokens.remove(0);
            if tokens.first() == Some(&Token::Punctuator(Punctuators::Semicolon)) {
                tokens.remove(0);
            }
            Some(ASTNode::Continue) 
        },
        Token::Punctuator(Punctuators::LeftBrace) => parse_block(tokens),
        // Handle variable and function declarations
        Token::Keyword(Keywords::Int) | Token::Keyword(Keywords::Char) | 
        Token::Keyword(Keywords::String) | Token::Keyword(Keywords::Arr) => {
            parse_dec_or_func(tokens)
        },
        Token::Keyword(Keywords::PutChar) => {
            tokens.remove(0); // consume 'putchar'
            if tokens.first() != Some(&Token::Punctuator(Punctuators::LeftParen)) {
                return None;
            }
            tokens.remove(0); // consume '('
            let expr = parse_expression(tokens)?;
            if tokens.first() != Some(&Token::Punctuator(Punctuators::RightParen)) {
                return None;
            }
            tokens.remove(0); // consume ')'
            if tokens.first() == Some(&Token::Punctuator(Punctuators::Semicolon)) {
                tokens.remove(0); // consume ';'
            }
            Some(ASTNode::PutChar { expr: Box::new(expr) })
        },
        _ => {
            // Expression/assignment statements
            let stmt = parse_assignment(tokens)?;
            if tokens.first() == Some(&Token::Punctuator(Punctuators::Semicolon)) {
                tokens.remove(0);
            }
            Some(stmt)
        }
    }
}


// since we are C to brainfuck compiler we will have to do something clever 
// Since C -> variable declaration and function declaration start the same
// Hence first consume the type as Keyword and then check if the next token is Identifier
// Then branch if "[" to array declaration or "(" to function declaration or "=" or ";" to variable declaration
// We will do this in the parse_statement function itself because it is easier to manage the tokens
// We will also have to handle the function parameters and return types

fn parse_variable(tokens: &mut Vec<Token>, var_type: Keywords, name: String) -> Option<ASTNode> {
    let mut array_dims = Vec::new();
    
    // Handle array dimensions: int arr[10][20];
    while tokens.first() == Some(&Token::Punctuator(Punctuators::LeftBracket)) {
        tokens.remove(0); // consume '['
        
        let size_expr = parse_expression(tokens)?;
        
        // For now, only allow constant integer array sizes
        match size_expr {
            ASTNode::LiteralInt(size) if size > 0 => {
                array_dims.push(size as usize);
            },
            _ => return None, // Non-constant or invalid array size
        }
        
        if tokens.first() != Some(&Token::Punctuator(Punctuators::RightBracket)) {
            return None; // Missing closing ']'
        }
        tokens.remove(0); // consume ']'
    }
    
    // Handle initialization
    let initial_value = if tokens.first() == Some(&Token::Operator(Operations::Assign)) {
        tokens.remove(0); // consume '='
        
        // Check for array initializer list: = {1, 2, 3}
        if tokens.first() == Some(&Token::Punctuator(Punctuators::LeftBrace)) {
            Some(Box::new(parse_array_initializer(tokens)?))
        } else {
            // Single expression initialization
            Some(Box::new(parse_expression(tokens)?))
        }
    } else {
        None // No initialization
    };

    // ALWAYS consume semicolon at the end
    if tokens.first() != Some(&Token::Punctuator(Punctuators::Semicolon)) {
        return None; // Missing semicolon
    }
    tokens.remove(0); // consume ';'

    Some(ASTNode::VariableDeclaration {
        var_type,
        name,
        array_dims: if array_dims.is_empty() { None } else { Some(array_dims) },
        initial_value,
    })

}

// Helper function for array initializers: {1, 2, 3, 4}
fn parse_array_initializer(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    if tokens.first() != Some(&Token::Punctuator(Punctuators::LeftBrace)) {
        return None;
    }
    tokens.remove(0); // consume '{'
    
    let mut elements = Vec::new();
    
    // Handle empty initializer: {}
    if tokens.first() == Some(&Token::Punctuator(Punctuators::RightBrace)) {
        tokens.remove(0); // consume '}'
        return Some(ASTNode::Block(elements)); // Reuse Block for initializer list
    }
    
    // Parse comma-separated expressions
    loop {
        let expr = parse_expression(tokens)?;
        elements.push(expr);
        
        match tokens.first() {
            Some(Token::Punctuator(Punctuators::Comma)) => {
                tokens.remove(0); // consume ','
            },
            Some(Token::Punctuator(Punctuators::RightBrace)) => {
                tokens.remove(0); // consume '}'
                break;
            },
            _ => return None, // Expected ',' or '}'
        }
    }
    Some(ASTNode::Block(elements)) // Reuse Block node for initializer
}


fn parse_function(tokens: &mut Vec<Token>, return_type: Keywords, name: String) -> Option<ASTNode> {
    // We expect '(' at this point
    if tokens.first() != Some(&Token::Punctuator(Punctuators::LeftParen)) {
        return None;
    }
    tokens.remove(0); // consume '('
    
    // Parse parameter list
    let params = parse_parameter_list(tokens)?;
    
    // Check if it's declaration (;) or definition ({...})
    match tokens.first() {
        Some(Token::Punctuator(Punctuators::Semicolon)) => {
            // Function declaration: int foo(int x, char y);
            tokens.remove(0); // consume ';'
            Some(ASTNode::Function {
                return_type,
                name,
                params,
                body: Box::new(ASTNode::Empty), // No body for declarations
            })
        },
        Some(Token::Punctuator(Punctuators::LeftBrace)) => {
            // Function definition: int foo(int x, char y) { ... }
            let body = parse_block(tokens)?;
            Some(ASTNode::Function {
                return_type,
                name,
                params,
                body: Box::new(body),
            })
        },
        _ => None, // Expected ';' or '{'
    }
}

// Helper function to parse parameter lists: (int a, char b, int c)
fn parse_parameter_list(tokens: &mut Vec<Token>) -> Option<Vec<(Keywords, String)>> {
    let mut params = Vec::new();
    
    // Handle empty parameter list: ()
    if tokens.first() == Some(&Token::Punctuator(Punctuators::RightParen)) {
        tokens.remove(0); // consume ')'
        return Some(params);
    }
    
    // Parse parameters
    loop {
        // Parse parameter type
        let param_type = match tokens.first() {
            Some(Token::Keyword(k)) => {
                let keyword = k.clone();
                tokens.remove(0);
                keyword
            },
            _ => return None, // Expected type keyword
        };
        
        // Parse parameter name
        let param_name = match tokens.first() {
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                tokens.remove(0);
                name
            },
            _ => return None, // Expected parameter name
        };
        
        params.push((param_type, param_name));
        
        // Check for continuation
        match tokens.first() {
            Some(Token::Punctuator(Punctuators::Comma)) => {
                tokens.remove(0); // consume ','
                // Continue to next parameter
            },
            Some(Token::Punctuator(Punctuators::RightParen)) => {
                tokens.remove(0); // consume ')'
                break; // End of parameter list
            },
            _ => return None, // Expected ',' or ')'
        }
    }
    
    Some(params)
}


fn parse_dec_or_func(tokens: &mut Vec<Token>)-> Option<ASTNode>{
    if tokens.first().is_none(){return None;}
    //Skip the first and the second tokens to a stack and then check the third token to branch
    let return_type = if let Token::Keyword(k) = tokens.remove(0) { k } else { return None; };
    let identifier = parse_identifier(tokens)?;
    let name  = if let ASTNode::Identifier(n) = identifier { n } else { return None; }; 
    match tokens.first()? {
        Token::Punctuator(Punctuators::LeftParen) => {
            parse_function(tokens, return_type, name)
        },
        Token::Punctuator(Punctuators::LeftBracket) | 
        Token::Operator(Operations::Assign) | 
        Token::Punctuator(Punctuators::Semicolon) => {
            parse_variable(tokens, return_type, name)
        },
        _ => None,
    }
}

fn parse_program(source: &str) -> Option<ASTNode> {
    let mut tokens = tokenize(source);
    let mut declarations = Vec::new();
    
    while !tokens.is_empty() {
        if let Some(decl) = parse_statement(&mut tokens) {
            declarations.push(decl);
        } else {
            return None; // Parse error
        }
    }
    
    Some(ASTNode::Program(declarations))
}

//------------------------------------------Semantic Analysis & IR-------------------------------------------
/*
create scopes
push scopes while doing syntax checks and semantic analysis 
our scope checks should be the last priority
then comes the in lining and dead code elimination


This should be easy right, we have struct like Scope that will contain two HashMap ( Var Info and Fun Info ) and one member Label (this will be something interesting ) 
for Var Info and FuncInfo : it's also really simple like it has it's current scope(opt we already have this info in Scope), indentifier, and the value stored typical stuff from the AST : Function { return_type: Keywords, name: String, params: Vec<(Keywords, String)>, body: Box<ASTNode> },
    VariableDeclaration { var_type: Keywords, name: String, array_dims: Option<Vec<usize>>, initial_value: Option<Box<ASTNode>> },

And then come the interesting part: the scope check "My ingenious brain said ; fuck tree walking lets name the scope like a tree and walk it " U get me?? No here is an example for ur mere mammal brain
Eg:
	int main(){
		if (cond) {
			int x;
		}
	} 
Here x is in the scope of global.main.if1 --> that is has access to all info of global and main and if condition. Boom. 
And we have these errors to take care of when pushing symbols to the scope

UndefinedVariable(String), -> included in the Scope check
UndefinedFunction(String), -> included in the Scope check
TypeMismatch { expected: Keywords, found: Keywords }, -> get the type from the symbol from Var Info and check the assignment
DuplicateDeclaration(String), -> included in the Scope check, (Shadowing is allowed)
ArgumentCountMismatch { expected: usize, found: usize }, -> get the number of params in the Func Info and check the number of args from the AST
BreakOutsideLoop, -> Push and remove "blocks" with symbols
ContinueOutsideLoop, -> Same
ReturnTypeMismatch { expected: Keywords, found: Keywords }, -> Same as the variable declarations
ArrayIndexNotInt, -> from Var Info do some instance of for easy shit
InvalidArraySize, -> Check for values greater than 1. 0 ain't allowed I guess

Then we have yeah in line function substitution : one proper simulation we can do inspired from the c2bf repo is -> declare the params as variables and proceed with the function boom
Then we have dead code elimination : out of scope values and not used variables are considered dead code and control flows with no effects are also dead code right?? yeah we delete them

best and simplest way is to re write tehe AST as whol and so sub_inline in place of function call and then substitute the values in conditions if possible or just print out non foldable if statements aren't supported here mada faka , Most of them will be foldable since there is no std input facility hahaha, I am devil

That is my idea, what do u think??

 */

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct VarInfo {
    var_type: Keywords,
    foldable_value: Option<ASTNode>,
    is_array: bool,
    array_dims: Option<Vec<usize>>,
    initial_value: Option<Box<ASTNode>>,
    is_initialized: bool, // NEW: Track initialization status
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FuncInfo {
    return_type: Keywords,
    params: Vec<(Keywords, String)>,
    body: Box<ASTNode>,
}

#[derive(Debug, Clone)]
struct Scope {
    label: String,
    variables: HashMap<String, VarInfo>,
    functions: HashMap<String, FuncInfo>,
    inside_loop: bool,
}

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
    UninitializedVariable(String), // NEW: For uninitialized variable usage
}

impl std::fmt::Display for SemanticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SemanticError::UndefinedVariable(name) => {
                write!(f, "This variable '{}' vanished into thin air—undefined as hell!", name)
            },
            SemanticError::UndefinedFunction(name) => {
                write!(f, "This function '{}' is a ghost! Try summoning it first.", name)
            },
            SemanticError::TypeMismatch { expected, found } => {
                write!(f, "This ain't the type party you were expecting. Expected '{:?}', found '{:?}'.", expected, found)
            },
            SemanticError::ReturnTypeMismatch { expected, found } => {
                write!(f, "This return's flavor is off. Expected '{:?}', got '{:?}'. Get your types right!", expected, found)
            },
            SemanticError::DuplicateDeclaration(name) => {
                write!(f, "This '{}' is already taken. Clone wars forbidden here!", name)
            },
            SemanticError::ArgumentCountMismatch { expected, found } => {
                write!(f, "This call expected {} args, but got {}. Count your damn fingers!", expected, found)
            },
            SemanticError::BreakOutsideLoop => {
                write!(f, "This 'break' is lost outside any loop. Where ya breakin' from?!")
            },
            SemanticError::ContinueOutsideLoop => {
                write!(f, "This 'continue' has nowhere to go. Loops only, buddy.")
            },
            SemanticError::ArrayIndexNotInt => {
                write!(f, "This array index isn't an integer. Arrays like whole numbers only.")
            },
            SemanticError::InvalidArraySize => {
                write!(f, "This array size is invalid. Must be > 0; zero is a no-go here.")
            },
            SemanticError::UninitializedVariable(name) => {
                write!(f, "This variable '{}' is using uninitialized garbage! Initialize it first, you savage!", name)
            },
        }
    }
}

fn get_scopes(current_scope: &str) -> Vec<String> {
    let parts: Vec<&str> = current_scope.split('.').collect(); 
    (1..=parts.len())
        .map(|i| parts[..i].join("."))
        .collect()
}

fn run_ast(ast: &mut Vec<ASTNode>) -> Result<Vec<ASTNode>, SemanticError> {
    let mut all_scopes = Vec::new();
    let mut scope_stack = Vec::new();
    let mut block_counter = 0;
    
    let global_scope = Scope {
        label: "global".to_string(),
        variables: HashMap::new(),
        functions: HashMap::new(),
        inside_loop: false,
    };
    all_scopes.push(global_scope.clone());
    scope_stack.push("global".to_string());
    
    let mut rebuilt_ast = Vec::new();
    
    for node in ast.iter() {
        let processed_node = process_node(
            node.clone(),
            &mut all_scopes,
            &mut scope_stack,
            &mut block_counter,
            None, // No current function at global scope
        )?;
        rebuilt_ast.push(processed_node);
    }
    
    Ok(rebuilt_ast)
}

// Helper function to check literal assignments with range validation
fn check_assignment_types(target_type: &Keywords, value_type: &Keywords) -> Result<(), SemanticError> {
    println!("DEBUG: Checking assignment types: {:?} = {:?}", target_type, value_type);
    
    match (target_type, value_type) {
        // Exact matches
        (Keywords::Int, Keywords::Int) => Ok(()),
        (Keywords::Char, Keywords::Char) => Ok(()),
        (Keywords::String, Keywords::String) => Ok(()),
        (Keywords::Arr, Keywords::Arr) => Ok(()),
        
        // Safe promotions
        (Keywords::Int, Keywords::Char) => Ok(()),
        (Keywords::Int, Keywords::String) => {
            println!("DEBUG: Rejecting string->int assignment");
            Err(SemanticError::TypeMismatch { 
                expected: Keywords::Int, 
                found: Keywords::String 
            })
        },
        (Keywords::Char, Keywords::String) => {
            Err(SemanticError::TypeMismatch { 
                expected: Keywords::Char, 
                found: Keywords::String 
            })
        },
        
        // All other cross-type assignments
        (expected, found) => {
            Err(SemanticError::TypeMismatch {
                expected: expected.clone(),
                found: found.clone(),
            })
        }
    }
}


// UPDATED: Added current_function_return_type parameter
fn process_node(
    node: ASTNode,
    all_scopes: &mut Vec<Scope>,
    scope_stack: &mut Vec<String>,
    block_counter: &mut usize,
    current_function_return_type: Option<&Keywords>, // NEW PARAMETER!
) -> Result<ASTNode, SemanticError> {
    match node {
        ASTNode::VariableDeclaration { var_type, name, array_dims, initial_value } => {
            let current_scope_label = scope_stack.last().unwrap().clone();
            
            if let Some(current_scope) = all_scopes.iter().find(|s| s.label == current_scope_label) {
                if current_scope.variables.contains_key(&name) {
                    return Err(SemanticError::DuplicateDeclaration(name));
                }
            }
            
            let is_array = array_dims.is_some();
            if let Some(ref dims) = array_dims {
                for &size in dims {
                    if size == 0 {
                        return Err(SemanticError::InvalidArraySize);
                    }
                }
            }
            
            let (processed_initial, is_initialized) = if let Some(init_val) = initial_value {
                let processed = process_node(*init_val, all_scopes, scope_stack, block_counter, current_function_return_type)?;
                let init_type = get_expr_type(&processed, all_scopes, scope_stack)?;
                if !types_compatible(&var_type, &init_type) {
                    return Err(SemanticError::TypeMismatch { 
                        expected: var_type, 
                        found: init_type 
                    });
                }
                (Some(Box::new(processed)), true)
            } else {
                (None, false)
            };
            
            let var_info = VarInfo {
                var_type: var_type.clone(),
                foldable_value: None,
                is_array,
                array_dims: array_dims.clone(),
                initial_value: processed_initial.clone(),
                is_initialized, // Track initialization
            };
            
            if let Some(current_scope) = all_scopes.iter_mut().find(|s| s.label == current_scope_label) {
                current_scope.variables.insert(name.clone(), var_info);
            }
            
            Ok(ASTNode::VariableDeclaration { var_type, name, array_dims, initial_value: processed_initial })
        },
        
        ASTNode::Function { return_type, name, params, body } => {
    let current_scope_label = scope_stack.last().unwrap().clone();
    
    // Check for duplicates first
    if let Some(current_scope) = all_scopes.iter().find(|s| s.label == current_scope_label) {
        if current_scope.functions.contains_key(&name) {
            return Err(SemanticError::DuplicateDeclaration(name));
        }
    }
    
    // ⭐ ADD FUNCTION TO SYMBOL TABLE FIRST (before processing body)
    let func_info = FuncInfo {
        return_type: return_type.clone(),
        params: params.clone(),
        body: body.clone(), // Use original body for now
    };
    
    if let Some(current_scope) = all_scopes.iter_mut().find(|s| s.label == current_scope_label) {
        current_scope.functions.insert(name.clone(), func_info);
    }
    
    // Now create function scope and process body
    let func_scope_label = format!("{}.{}", current_scope_label, name);
    let mut func_scope = Scope {
        label: func_scope_label.clone(),
        variables: HashMap::new(),
        functions: HashMap::new(),
        inside_loop: false,
    };
    
    // Add parameters to function scope
    for (param_type, param_name) in &params {
        let param_info = VarInfo {
            var_type: param_type.clone(),
            foldable_value: None,
            is_array: false,
            array_dims: None,
            initial_value: None,
            is_initialized: true,
        };
        func_scope.variables.insert(param_name.clone(), param_info);
    }
    
    all_scopes.push(func_scope);
    scope_stack.push(func_scope_label);
    
    // Process body AFTER function is in symbol table
    let processed_body = if matches!(*body, ASTNode::Empty) {
        body
    } else {
        Box::new(process_node(*body, all_scopes, scope_stack, block_counter, Some(&return_type))?)
    };
    
    scope_stack.pop();
    
    // Update the function info with processed body
    if let Some(current_scope) = all_scopes.iter_mut().find(|s| s.label == current_scope_label) {
        if let Some(func_info) = current_scope.functions.get_mut(&name) {
            func_info.body = processed_body.clone();
        }
    }
    
    Ok(ASTNode::Function { return_type, name, params, body: processed_body })
},

        
        ASTNode::Identifier(name) => {
            if let Some(var_info) = lookup_variable(&name, all_scopes, scope_stack) {
                // NEW: Check if variable is initialized
                if !var_info.is_initialized {
                    return Err(SemanticError::UninitializedVariable(name));
                }
                Ok(ASTNode::Identifier(name))
            } else {
                Err(SemanticError::UndefinedVariable(name))
            }
        },
        
        ASTNode::FunctionCall { name, args } => {
            let func_info = lookup_function(&name, all_scopes, scope_stack)
                .ok_or_else(|| SemanticError::UndefinedFunction(name.clone()))?;
            
            if args.len() != func_info.params.len() {
                return Err(SemanticError::ArgumentCountMismatch { 
                    expected: func_info.params.len(), 
                    found: args.len() 
                });
            }
            
            let mut processed_args = Vec::new();
            for (i, arg) in args.into_iter().enumerate() {
                let processed_arg = process_node(arg, all_scopes, scope_stack, block_counter, current_function_return_type)?;
                let arg_type = get_expr_type(&processed_arg, all_scopes, scope_stack)?;
                let expected_type = &func_info.params[i].0;
                
                if !types_compatible(expected_type, &arg_type) {
                    return Err(SemanticError::TypeMismatch { 
                        expected: expected_type.clone(), 
                        found: arg_type 
                    });
                }
                
                processed_args.push(processed_arg);
            }
            
            Ok(ASTNode::FunctionCall { name, args: processed_args })
        },
        
        ASTNode::ArrayAccess { name, index } => {
            let var_info = lookup_variable(&name, all_scopes, scope_stack)
                .ok_or_else(|| SemanticError::UndefinedVariable(name.clone()))?;
            
            if !var_info.is_array {
                return Err(SemanticError::TypeMismatch { 
                    expected: Keywords::Arr, 
                    found: var_info.var_type 
                });
            }
            
            let processed_index = process_node(*index, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            let index_type = get_expr_type(&processed_index, all_scopes, scope_stack)?;
            
            if !matches!(index_type, Keywords::Int | Keywords::Char) {
                return Err(SemanticError::ArrayIndexNotInt);
            }
            
            Ok(ASTNode::ArrayAccess { name, index: Box::new(processed_index) })
        },
        
        // In your assignment handling within `process_node`
ASTNode::Assignment { target, value } => {
    println!("DEBUG: Processing assignment"); // Add this for debugging
    
    let processed_target = process_node(*target, all_scopes, scope_stack, block_counter, current_function_return_type)?;
    let processed_value = process_node(*value, all_scopes, scope_stack, block_counter, current_function_return_type)?;

    let target_type = get_expr_type(&processed_target, all_scopes, scope_stack)?;
    let value_type = get_expr_type(&processed_value, all_scopes, scope_stack)?;

    println!("DEBUG: Assignment types - target: {:?}, value: {:?}", target_type, value_type);

    // Enhanced type checking function
    check_assignment_types(&target_type, &value_type)?;

    // Mark variable as initialized
    if let ASTNode::Identifier(var_name) = &processed_target {
        mark_variable_initialized(var_name, all_scopes, scope_stack);
    }

    Ok(ASTNode::Assignment {
        target: Box::new(processed_target),
        value: Box::new(processed_value),
    })
},

        ASTNode::Break => {
            let mut in_loop = false;
            for scope_label in scope_stack.iter().rev() {
                if let Some(scope) = all_scopes.iter().find(|s| &s.label == scope_label) {
                    if scope.inside_loop {
                        in_loop = true;
                        break;
                    }
                }
            }
            
            if !in_loop {
                return Err(SemanticError::BreakOutsideLoop);
            }
            Ok(ASTNode::Break)
        },
        
        ASTNode::Continue => {
            let mut in_loop = false;
            for scope_label in scope_stack.iter().rev() {
                if let Some(scope) = all_scopes.iter().find(|s| &s.label == scope_label) {
                    if scope.inside_loop {
                        in_loop = true;
                        break;
                    }
                }
            }
            
            if !in_loop {
                return Err(SemanticError::ContinueOutsideLoop);
            }
            Ok(ASTNode::Continue)
        },
        
        ASTNode::If { condition, then_branch, else_branch } => {
            let processed_condition = process_node(*condition, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            let processed_then = process_node(*then_branch, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            let processed_else = if let Some(else_node) = else_branch {
                Some(Box::new(process_node(*else_node, all_scopes, scope_stack, block_counter, current_function_return_type)?))
            } else {
                None
            };
            
            Ok(ASTNode::If {
                condition: Box::new(processed_condition),
                then_branch: Box::new(processed_then),
                else_branch: processed_else,
            })
        },
        
        ASTNode::While { condition, body } => {
            *block_counter += 1;
            let current_scope_label = scope_stack.last().unwrap();
            let while_scope_label = format!("{}.while{}", current_scope_label, block_counter);
            
            let while_scope = Scope {
                label: while_scope_label.clone(),
                variables: HashMap::new(),
                functions: HashMap::new(),
                inside_loop: true,
            };
            
            all_scopes.push(while_scope);
            scope_stack.push(while_scope_label);
            
            let processed_condition = process_node(*condition, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            let processed_body = process_node(*body, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            
            scope_stack.pop();
            
            Ok(ASTNode::While { 
                condition: Box::new(processed_condition), 
                body: Box::new(processed_body) 
            })
        },
        
        ASTNode::For { init, condition, increment, body } => {
            *block_counter += 1;
            let current_scope_label = scope_stack.last().unwrap();
            let for_scope_label = format!("{}.for{}", current_scope_label, block_counter);
            
            let for_scope = Scope {
                label: for_scope_label.clone(),
                variables: HashMap::new(),
                functions: HashMap::new(),
                inside_loop: true,
            };
            
            all_scopes.push(for_scope);
            scope_stack.push(for_scope_label);
            
            let processed_init = process_node(*init, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            let processed_condition = process_node(*condition, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            let processed_increment = process_node(*increment, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            let processed_body = process_node(*body, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            
            scope_stack.pop();
            
            Ok(ASTNode::For { 
                init: Box::new(processed_init),
                condition: Box::new(processed_condition),
                increment: Box::new(processed_increment),
                body: Box::new(processed_body) 
            })
        },
        
        ASTNode::Block(statements) => {
            *block_counter += 1;
            let current_scope_label = scope_stack.last().unwrap();
            let block_scope_label = format!("{}.block{}", current_scope_label, block_counter);
            
            let parent_in_loop = all_scopes.iter()
                .find(|s| &s.label == current_scope_label)
                .map(|s| s.inside_loop)
                .unwrap_or(false);
            
            let block_scope = Scope {
                label: block_scope_label.clone(),
                variables: HashMap::new(),
                functions: HashMap::new(),
                inside_loop: parent_in_loop,
            };
            
            all_scopes.push(block_scope);
            scope_stack.push(block_scope_label);
            
            let mut processed_statements = Vec::new();
            for stmt in statements {
                let processed = process_node(stmt, all_scopes, scope_stack, block_counter, current_function_return_type)?;
                processed_statements.push(processed);
            }
            
            scope_stack.pop();
            
            Ok(ASTNode::Block(processed_statements))
        },
        
        ASTNode::BinaryOp { op, left, right } => {
            let processed_left = process_node(*left, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            let processed_right = process_node(*right, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            
            Ok(ASTNode::BinaryOp { 
                op, 
                left: Box::new(processed_left), 
                right: Box::new(processed_right) 
            })
        },
        
        ASTNode::UnaryOp { op, expr } => {
            let processed_expr = process_node(*expr, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            
            Ok(ASTNode::UnaryOp {
                op,
                expr: Box::new(processed_expr)
            })
        },
        
        ASTNode::TernaryOp { condition, true_expr, false_expr } => {
            let processed_condition = process_node(*condition, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            let processed_true = process_node(*true_expr, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            let processed_false = process_node(*false_expr, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            
            Ok(ASTNode::TernaryOp {
                condition: Box::new(processed_condition),
                true_expr: Box::new(processed_true),
                false_expr: Box::new(processed_false),
            })
        },
        
        ASTNode::PutChar { expr } => {
            let processed_expr = process_node(*expr, all_scopes, scope_stack, block_counter, current_function_return_type)?;
            Ok(ASTNode::PutChar { expr: Box::new(processed_expr) })
        },
        
        // FIXED: Proper return type checking!
        ASTNode::Return(expr) => {
    match (current_function_return_type, expr) {
        // Function expects a return value and one is provided
        (Some(expected_type), Some(return_expr)) => {
            let processed_expr = process_node(
                *return_expr, 
                all_scopes, 
                scope_stack, 
                block_counter, 
                current_function_return_type
            )?;
            let expr_type = get_expr_type(&processed_expr, all_scopes, scope_stack)?;
            
            if !types_compatible(expected_type, &expr_type) {
                return Err(SemanticError::ReturnTypeMismatch { 
                    expected: expected_type.clone(), 
                    found: expr_type 
                });
            }
            
            Ok(ASTNode::Return(Some(Box::new(processed_expr))))
        },
        
        // Function expects a return value but none provided
        (Some(expected_type), None) => {
            Err(SemanticError::ReturnTypeMismatch { 
                expected: expected_type.clone(), 
                found: Keywords::String // Use different type to indicate "void"
            })
        },
        
        // No function context but return expression provided
        (None, Some(return_expr)) => {
            let processed_expr = process_node(
                *return_expr, 
                all_scopes, 
                scope_stack, 
                block_counter, 
                current_function_return_type
            )?;
            Ok(ASTNode::Return(Some(Box::new(processed_expr))))
        },
        
        // No function context and no return expression
        (None, None) => {
            Ok(ASTNode::Return(None))
        },
    }
},

        ASTNode::Program(declarations) => {
            let mut processed_declarations = Vec::new();
            for decl in declarations {
                let processed = process_node(decl, all_scopes, scope_stack, block_counter, current_function_return_type)?;
                processed_declarations.push(processed);
            }
            Ok(ASTNode::Program(processed_declarations))
        },
        
        node @ (ASTNode::LiteralInt(_) | ASTNode::LiteralChar(_) | ASTNode::LiteralString(_) | ASTNode::Empty) => {
            Ok(node)
        },
    }
}

// Helper function to mark variables as initialized
fn mark_variable_initialized(var_name: &str, all_scopes: &mut [Scope], scope_stack: &[String]) {
    let current_scope_label = scope_stack.last().unwrap();
    let scope_labels = get_scopes(current_scope_label);
    
    for label in scope_labels.iter().rev() {
        if let Some(scope) = all_scopes.iter_mut().find(|s| &s.label == label) {
            if let Some(var_info) = scope.variables.get_mut(var_name) {
                var_info.is_initialized = true;
                return;
            }
        }
    }
}

fn lookup_variable(name: &str, all_scopes: &[Scope], scope_stack: &[String]) -> Option<VarInfo> {
    let current_scope_label = scope_stack.last()?;
    let scope_labels = get_scopes(current_scope_label);
    
    for label in scope_labels.iter().rev() {
        if let Some(scope) = all_scopes.iter().find(|s| &s.label == label) {
            if let Some(var_info) = scope.variables.get(name) {
                return Some(var_info.clone());
            }
        }
    }
    None
}

fn lookup_function(name: &str, all_scopes: &[Scope], scope_stack: &[String]) -> Option<FuncInfo> {
    let current_scope_label = scope_stack.last()?;
    let scope_labels = get_scopes(current_scope_label);
    
    for label in scope_labels.iter().rev() {
        if let Some(scope) = all_scopes.iter().find(|s| &s.label == label) {
            if let Some(func_info) = scope.functions.get(name) {
                return Some(func_info.clone());
            }
        }
    }
    None
}

fn get_expr_type(node: &ASTNode, all_scopes: &[Scope], scope_stack: &[String]) -> Result<Keywords, SemanticError> {
    match node {
        ASTNode::LiteralInt(_) => Ok(Keywords::Int),
        ASTNode::LiteralChar(_) => Ok(Keywords::Char),
        ASTNode::LiteralString(_) => Ok(Keywords::String),
        
        ASTNode::Identifier(name) => {
            let var_info = lookup_variable(name, all_scopes, scope_stack)
                .ok_or_else(|| SemanticError::UndefinedVariable(name.clone()))?;
            Ok(var_info.var_type.clone())
        },
        
        ASTNode::ArrayAccess { name, .. } => {
            let var_info = lookup_variable(name, all_scopes, scope_stack)
                .ok_or_else(|| SemanticError::UndefinedVariable(name.clone()))?;
            
            if !var_info.is_array {
                return Err(SemanticError::TypeMismatch { 
                    expected: Keywords::Arr, 
                    found: var_info.var_type.clone() 
                });
            }
            
            Ok(var_info.var_type.clone())
        },
        
        ASTNode::FunctionCall { name, .. } => {
            let func_info = lookup_function(name, all_scopes, scope_stack)
                .ok_or_else(|| SemanticError::UndefinedFunction(name.clone()))?;
            Ok(func_info.return_type.clone())
        },
        
        ASTNode::BinaryOp { op, left, right } => {
            let left_type = get_expr_type(left, all_scopes, scope_stack)?;
            let right_type = get_expr_type(right, all_scopes, scope_stack)?;
            
            match op {
                Operations::Add | Operations::Subtract | Operations::Multiply | 
                Operations::Divide | Operations::Modulus => {
                    if types_compatible(&left_type, &right_type) {
                        match (&left_type, &right_type) {
                            (Keywords::Int, _) | (_, Keywords::Int) => Ok(Keywords::Int),
                            (Keywords::Char, Keywords::Char) => Ok(Keywords::Char),
                            _ => Err(SemanticError::TypeMismatch { expected: left_type, found: right_type })
                        }
                    } else {
                        Err(SemanticError::TypeMismatch { expected: left_type, found: right_type })
                    }
                },
                
                Operations::Equal | Operations::NotEqual | Operations::GreaterThan | 
                Operations::LessThan | Operations::GreaterThanOrEqual | Operations::LessThanOrEqual => {
                    if types_compatible(&left_type, &right_type) {
                        Ok(Keywords::Int)
                    } else {
                        Err(SemanticError::TypeMismatch { expected: left_type, found: right_type })
                    }
                },
                
                Operations::And | Operations::Or => {
                    if left_type == Keywords::Int && right_type == Keywords::Int {
                        Ok(Keywords::Int)
                    } else {
                        Err(SemanticError::TypeMismatch { expected: Keywords::Int, found: right_type })
                    }
                },
                
                Operations::BitwiseAnd | Operations::BitwiseOr | Operations::BitwiseXor |
                Operations::LeftShift | Operations::RightShift => {
                    if left_type == Keywords::Int && right_type == Keywords::Int {
                        Ok(Keywords::Int)
                    } else {
                        Err(SemanticError::TypeMismatch { expected: Keywords::Int, found: right_type })
                    }
                },
                
                Operations::Assign => Ok(left_type),
                
                _ => Ok(Keywords::Int),
            }
        },
        
        ASTNode::UnaryOp { op, expr } => {
            let expr_type = get_expr_type(expr, all_scopes, scope_stack)?;
            
            match op {
                Operations::Not => {
                    if expr_type == Keywords::Int {
                        Ok(Keywords::Int)
                    } else {
                        Err(SemanticError::TypeMismatch { expected: Keywords::Int, found: expr_type })
                    }
                },
                Operations::Subtract => {
                    if matches!(expr_type, Keywords::Int | Keywords::Char) {
                        Ok(expr_type)
                    } else {
                        Err(SemanticError::TypeMismatch { expected: Keywords::Int, found: expr_type })
                    }
                },
                Operations::BitwiseNot => {
                    if expr_type == Keywords::Int {
                        Ok(Keywords::Int)
                    } else {
                        Err(SemanticError::TypeMismatch { expected: Keywords::Int, found: expr_type })
                    }
                },
                Operations::Increment | Operations::Decrement => {
                    if matches!(expr_type, Keywords::Int | Keywords::Char) {
                        Ok(expr_type)
                    } else {
                        Err(SemanticError::TypeMismatch { expected: Keywords::Int, found: expr_type })
                    }
                },
                _ => Ok(expr_type),
            }
        },
        
        ASTNode::TernaryOp { condition, true_expr, false_expr } => {
            let cond_type = get_expr_type(condition, all_scopes, scope_stack)?;
            
            if cond_type != Keywords::Int {
                return Err(SemanticError::TypeMismatch {
                    expected: Keywords::Int,
                    found: cond_type,
                });
            }
            
            let true_type = get_expr_type(true_expr, all_scopes, scope_stack)?;
            let false_type = get_expr_type(false_expr, all_scopes, scope_stack)?;
            
            if types_compatible(&true_type, &false_type) {
                match (true_type.clone(), false_type) {
                    (Keywords::Int, _) | (_, Keywords::Int) => Ok(Keywords::Int),
                    (Keywords::Char, Keywords::Char) => Ok(Keywords::Char),
                    _ => Ok(true_type),
                }
            } else {
                Err(SemanticError::TypeMismatch {
                    expected: true_type,
                    found: false_type,
                })
            }
        },
        
        ASTNode::Assignment { target, .. } => {
            get_expr_type(target, all_scopes, scope_stack)
        },
        
        ASTNode::PutChar { .. } => Ok(Keywords::Int),
        ASTNode::Block(_) => Ok(Keywords::Int),
        
        _ => {
            println!("Warning: Unhandled expression type in get_expr_type: {:?}", node);
            Ok(Keywords::Int)
        }
    }
}

fn types_compatible(expected: &Keywords, found: &Keywords) -> bool {
    match (expected, found) {
        (Keywords::Int, Keywords::Int) => true,
        (Keywords::Char, Keywords::Char) => true,
        (Keywords::String, Keywords::String) => true,
        (Keywords::Char, Keywords::Int) => true,
        (Keywords::Int, Keywords::Char) => true,
        (Keywords::Int, Keywords::String) => false,
        _ => false,
    }
}

/*
Big design choice change:
    Since adding a new ASTNode variant requires updating multiple functions,
    I've decided to generate a pre IR variant of the AST tree to accomodate the renamed and inlined function
    Or just generate the IR straight up from this semantic analysis done AST
    Or Add it such that it doesn't change the code structure and it's just vec<ASTNode> rather than a seperate ASTnode
    The latest approach is feasible than generate an entire Pre IR layer and updating all functions but requires some serious 
    Changes to the AST structure if we were to snuck some vec<ASTNode> so here are the steps
        Write a helper to rearrange the AST with sneaking in the vec<ASTNode>
        Write a helper to generate the inline_function(&ast, counter, Scope) -> (vec<ASTNode)
        Write the filter pass inline_program(&mut ast) -> vec<ASTNode>, tape offsets as input to the next pass that is the actual IR

    That is today's challenge! Lets fuck this bitch in the ass
 */

 /*
 Turns out insert_inline function that should recursively insert ASTNodes inplace of a single ASTNode is extremely fidely
 So MY master brain came up with what I call the most shittiest plan in the fucking century i.e:
    ***Modify the source code itself***
    Hear me out... What if we generate the inline function as a string and just do some simple string manip and re run tokenize, parser and the semantic analysis
    I know it's just really waste out resources but, hear me out... it's just some simple string manip than tree traversal
    And most importantnly we can remove func declarations, rename variables, inline functions and dead code elimination in single pass
    So yeah That's the master plan
  */
/*
    And another master plan with greatest danger and trade offs :
    And here is why a pre tokenize inlining will be bad and good: bad if u are to print out the error, 
    the user doesn't know ur renamed variable and they be like what the fuck ?? and the good if it's just a compiler form C to BF,
     cuz u don't want to print out if the C is right or no? 
     and if I really want to print out the thing too, i would have to maintain another hash map from user_var to renamed and yeah dead code elimination is a hazard after wards, 
     SOO I guess inline_pass after the first semantic analysis would be good and have similar problems 
     but hey I am not spending another week procrastinating an insert_inline function
     So chat gpt said no body has done it before since I guess nobody is on meth writing a compiler
*/
use regex::Regex;

#[derive(Clone, Debug)]
struct FunctionSignature {
    name: String,
    params: Vec<String>,
    body: String,
}

static mut INLINE_COUNTER: usize = 0;

/// Main entry: extract functions, inline them, remove declarations
pub fn clean_source(source: &str) -> String {
    unsafe { INLINE_COUNTER = 0; }
    
    let funcs = extract_functions(source);
    
    if funcs.is_empty() {
        println!("[INLINE] No functions to inline");
        return source.to_string();
    }
    
    println!("[INLINE] Found {} functions", funcs.len());
    
    let mut result = source.to_string();
    
    // Iteratively inline until no changes
    let mut i = 0;
    for _ in 0..50 {
        println!("[INLINE] Inlining pass {}", i + 1);
        let before = result.clone();
        
        for func in funcs.values() {
            result = inline_function(&result, func);
        }
        
        if result == before {
            break;
        }
        i = i+1;
    }
    
    // Remove function declarations
    result = remove_functions(&result, &funcs);
    
    println!("[INLINE] Inlining complete");
    result
}

/// Extract all function signatures from source
fn extract_functions(source: &str) -> HashMap<String, FunctionSignature> {
    let mut funcs = HashMap::new();
    
    let re = Regex::new(r"(?s)(int|char|void)\s+(\w+)\s*\(([^)]*)\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}").unwrap();
    
    for cap in re.captures_iter(source) {
        let name = cap[2].to_string();
        
        if name == "main" {
            continue;
        }
        
        let params: Vec<String> = cap[3]
            .split(',')
            .filter_map(|p| p.trim().split_whitespace().last())
            .map(|s| s.to_string())
            .collect();
        
        let body = cap[4].trim().to_string();
        
        funcs.insert(name.clone(), FunctionSignature { name, params, body });
    }
    
    funcs
}

/// Inline all calls to a specific function
fn inline_function(source: &str, func: &FunctionSignature) -> String {
    let mut result = source.to_string();
    let mut search_pos = 0;
    
    loop {
        let Some(rel_pos) = result[search_pos..].find(&func.name) else {
            break;
        };
        
        let pos = search_pos + rel_pos;
        
        if is_declaration(&result, pos) {
            search_pos = pos + func.name.len();
            continue;
        }
        
        let after = pos + func.name.len();
        if after >= result.len() || !result[after..].trim_start().starts_with('(') {
            search_pos = pos + func.name.len();
            continue;
        }
        
        let paren_start = after + result[after..].find('(').unwrap();
        let Some(paren_end) = find_matching_paren(&result, paren_start) else {
            search_pos = pos + func.name.len();
            continue;
        };
        
        let args_str = &result[paren_start + 1..paren_end];
        let args = parse_args(args_str);
        
        // Find statement start
        let stmt_start = find_statement_start(&result, pos);
        
        // Generate inline code split into statements and result var
        let (inline_stmts, result_var) = generate_inline_split(func, &args);
        
        // Build new result: inject statements before current statement
        let mut new_result = String::new();
        new_result.push_str(&result[..stmt_start]);     // Before statement
        new_result.push_str(&inline_stmts);              // Injected statements
        new_result.push_str(&result[stmt_start..pos]);  // Statement up to call
        new_result.push_str(&result_var);                // Replace call with var
        new_result.push_str(&result[paren_end + 1..]);  // Rest after call
        
        result = new_result;
        search_pos = 0; // Restart from beginning
    }
    
    result
}

/// Find start of current statement (last ';' or '{' before pos)
fn find_statement_start(source: &str, pos: usize) -> usize {
    let chars: Vec<char> = source.chars().collect();
    
    for i in (0..pos).rev() {
        if chars[i] == ';' || chars[i] == '{' {
            // Move past the ';' or '{'
            return i + 1;
        }
    }
    
    0
}

/// Check if position is a function declaration
fn is_declaration(source: &str, pos: usize) -> bool {
    let before = &source[..pos];
    let words: Vec<&str> = before.split_whitespace().collect();
    
    if let Some(last) = words.last() {
        if *last == "int" || *last == "char" || *last == "void" {
            return true;
        }
    }
    
    false
}

/// Find matching closing parenthesis
fn find_matching_paren(source: &str, open: usize) -> Option<usize> {
    let chars: Vec<char> = source.chars().collect();
    let mut depth = 1;
    
    for i in (open + 1)..chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    
    None
}

/// Parse arguments, respecting nested calls
fn parse_args(args_str: &str) -> Vec<String> {
    if args_str.trim().is_empty() {
        return vec![];
    }
    
    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    
    for ch in args_str.chars() {
        match ch {
            '(' => {
                depth += 1;
                current.push(ch);
            }
            ')' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                args.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    
    if !current.trim().is_empty() {
        args.push(current.trim().to_string());
    }
    
    args
}

/// Generate inline code as (statements, result_variable)
fn generate_inline_split(func: &FunctionSignature, args: &[String]) -> (String, String) {
    let id = unsafe {
        INLINE_COUNTER += 1;
        INLINE_COUNTER
    };
    
    let prefix = format!("__inl_{}_{}", func.name, id);
    let mut stmts = String::new();
    
    // Declare argument temps
    let arg_vars: Vec<String> = (0..args.len())
        .map(|i| format!("{}_arg{}", prefix, i))
        .collect();
    
    for (i, arg) in args.iter().enumerate() {
        stmts.push_str(&format!("int {} = {}; ", arg_vars[i], arg));
    }
    
    // Copy body and replace params with arg vars
    let mut body = func.body.clone();
    
    for (param, arg_var) in func.params.iter().zip(arg_vars.iter()) {
        let re = Regex::new(&format!(r"\b{}\b", regex::escape(param))).unwrap();
        body = re.replace_all(&body, arg_var.as_str()).to_string();
    }
    
    // Replace return with assignment
    let ret_var = format!("{}_ret", prefix);
    let re = Regex::new(r"return\s+([^;]+);").unwrap();
    
    body = re.replace_all(&body, |caps: &regex::Captures| {
        format!("int {} = {}; ", ret_var, &caps[1])
    }).to_string();
    
    stmts.push_str(&body);
    
    (stmts, ret_var)
}

/// Remove function declarations
fn remove_functions(source: &str, funcs: &HashMap<String, FunctionSignature>) -> String {
    let mut result = source.to_string();
    
    for func in funcs.values() {
        let pattern = format!(
            r"(?s)(int|char|void)\s+{}\s*\([^)]*\)\s*\{{[^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*\}}",
            regex::escape(&func.name)
        );
        
        if let Ok(re) = Regex::new(&pattern) {
            result = re.replace_all(&result, "").to_string();
        }
    }
    
    result
}


#[derive(Debug, Clone, PartialEq)]
pub enum IRInstruction {
    // Memory operations
    Goto(usize),                    // Move pointer to specific cell
    Set(usize, i64),                // Set cell to value (clears then sets)
    Add(usize, i64),                // Add/subtract from cell
    Copy(usize, usize),             // Copy from src to dest (non-destructive)
    Move(usize, usize),             // Move from src to dest (destructive)
    
    // Control flow
    LoopStart(usize),               // [ - loop while cell != 0
    LoopEnd(usize),                 // ] - end loop
    
    // I/O
    Output(usize),                  // . - output cell value as char
    Input(usize),                   // , - input char into cell
    
    // Array operations (helper instructions)
    ArrayLoad(usize, usize, usize), // Load arr[index] -> dest (base, index_cell, dest)
    ArrayStore(usize, usize, usize),// Store value -> arr[index] (base, index_cell, src)
    
    // Debug/optimization markers
    Comment(String),                // For debugging and readability
    Label(String),                  // For tracking code sections
}

// ============================================================================
// IR GENERATOR
// ============================================================================
#[allow(dead_code)]
pub struct IRGenerator {
    // Memory management
    cell_counter: usize,
    current_pointer: usize,
    var_offsets: HashMap<String, usize>,
    array_sizes: HashMap<String, usize>,
    
    // IR output
    instructions: Vec<IRInstruction>,
    
    // Loop tracking for break/continue
    loop_stack: Vec<LoopContext>,
    
    // Temporary cell reuse pool
    temp_pool: Vec<usize>,
    temp_in_use: Vec<bool>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LoopContext {
    condition_cell: usize,
    label: String,
}

impl IRGenerator {
    pub fn new() -> Self {
        Self {
            cell_counter: 0,
            current_pointer: 0,
            var_offsets: HashMap::new(),
            array_sizes: HashMap::new(),
            instructions: Vec::new(),
            loop_stack: Vec::new(),
            temp_pool: Vec::new(),
            temp_in_use: Vec::new(),
        }
    }
    
    // ========================================================================
    // MAIN GENERATION ENTRY POINT
    // ========================================================================
    
    fn generate(&mut self, ast: ASTNode) -> Result<Vec<IRInstruction>, String> {
        self.generate_node(ast)?;
        Ok(self.instructions.clone())
    }
    
    // ========================================================================
    // NODE GENERATION DISPATCHER
    // ========================================================================
    
    fn generate_node(&mut self, node: ASTNode) -> Result<(), String> {
        match node {
            ASTNode::Program(nodes) => self.gen_program(nodes),
            ASTNode::Function { body, .. } => self.generate_node(*body),
            ASTNode::Block(nodes) => self.gen_block(nodes),
            ASTNode::VariableDeclaration { var_type: _, name, array_dims, initial_value } => {
                self.gen_var_decl(name, array_dims, initial_value)
            },
            ASTNode::Assignment { target, value } => self.gen_assignment(*target, *value),
            ASTNode::BinaryOp { op: Operations::Assign, left, right } => {
    self.gen_assignment(*left, *right)
},
            ASTNode::For { init, condition, increment, body } => {
                self.gen_for(*init, *condition, *increment, *body)
            },
            ASTNode::While { condition, body } => self.gen_while(*condition, *body),
            ASTNode::If { condition, then_branch, else_branch } => {
                self.gen_if(*condition, *then_branch, else_branch.map(|b| *b))
            },
            ASTNode::PutChar { expr } => self.gen_putchar(*expr),
            ASTNode::Break => self.gen_break(),
            ASTNode::Continue => self.gen_continue(),
            ASTNode::Return(_) => Ok(()), // Ignore for main()
            ASTNode::Empty => Ok(()),
            _ => Err(format!("Unexpected statement node: {:?}", node)),
        }
    }
    
    // ========================================================================
    // EXPRESSION EVALUATION (returns cell containing result)
    // ========================================================================
    
    fn eval_expression(&mut self, node: ASTNode) -> Result<usize, String> {
        match node {
            ASTNode::LiteralInt(n) => self.gen_literal_int(n),
            ASTNode::LiteralChar(c) => self.gen_literal_char(c),
            ASTNode::LiteralString(s) => self.gen_literal_string(&s),
            ASTNode::Identifier(name) => self.gen_identifier(&name),
            ASTNode::ArrayAccess { name, index } => self.gen_array_access(&name, *index),
            ASTNode::BinaryOp { op, left, right } => {
    // Guard: Assign is a statement, not an expression
    if op == Operations::Assign {
        return Err("Assignment is not an expression".to_string());
    }
    self.gen_binary_op(op, *left, *right)
},

            ASTNode::UnaryOp { op, expr } => self.gen_unary_op(op, *expr),
            ASTNode::TernaryOp { condition, true_expr, false_expr } => {
                self.gen_ternary(*condition, *true_expr, *false_expr)
            },
            ASTNode::FunctionCall { name, args } => self.gen_function_call(&name, args),
            _ => Err(format!("Not an expression: {:?}", node)),
        }
    }
    
    // ========================================================================
    // STATEMENT GENERATORS
    // ========================================================================
    
    fn gen_program(&mut self, nodes: Vec<ASTNode>) -> Result<(), String> {
        self.emit_comment("=== Program Start ===");
        for node in nodes {
            self.generate_node(node)?;
        }
        self.emit_comment("=== Program End ===");
        Ok(())
    }
    
    fn gen_block(&mut self, nodes: Vec<ASTNode>) -> Result<(), String> {
        for node in nodes {
            self.generate_node(node)?;
        }
        Ok(())
    }
    
    fn gen_var_decl(
        &mut self,
        name: String,
        array_dims: Option<Vec<usize>>,
        initial_value: Option<Box<ASTNode>>
    ) -> Result<(), String> {
        let size = if let Some(ref dims) = array_dims {
            dims.iter().product()
        } else {
            1
        };
        
        let base_cell = self.cell_counter;
        self.cell_counter += size;
        self.var_offsets.insert(name.clone(), base_cell);
        
        if size > 1 {
            self.array_sizes.insert(name.clone(), size);
        }
        
        self.emit_comment(&format!("Declare {} at cell {}", name, base_cell));
        
        // Initialize
        if let Some(init) = initial_value {
            if size == 1 {
                // Single variable
                let value_cell = self.eval_expression(*init)?;
                self.emit(IRInstruction::Copy(value_cell, base_cell));
                self.free_temp(value_cell);
            } else {
                // Array initialization
                match *init {
                    ASTNode::LiteralString(s) => {
                        // Initialize with string
                        for (i, ch) in s.chars().enumerate() {
                            if i < size {
                                self.emit(IRInstruction::Set(base_cell + i, ch as i64));
                            }
                        }
                    },
                    _ => {
                        // Initialize first element
                        let value_cell = self.eval_expression(*init)?;
                        self.emit(IRInstruction::Copy(value_cell, base_cell));
                        self.free_temp(value_cell);
                    }
                }
            }
        } else {
            // Zero-initialize
            for i in 0..size {
                self.emit(IRInstruction::Set(base_cell + i, 0));
            }
        }
        
        Ok(())
    }
    
    fn gen_assignment(&mut self, target: ASTNode, value: ASTNode) -> Result<(), String> {
        let value_cell = self.eval_expression(value)?;
        
        match target {
            ASTNode::Identifier(name) => {
                let target_cell = *self.var_offsets.get(&name)
                    .ok_or_else(|| format!("Undefined variable: {}", name))?;
                self.emit(IRInstruction::Copy(value_cell, target_cell));
            },
            ASTNode::ArrayAccess { name, index } => {
                let base = *self.var_offsets.get(&name)
                    .ok_or_else(|| format!("Undefined array: {}", name))?;
                let index_cell = self.eval_expression(*index)?;
                self.emit(IRInstruction::ArrayStore(base, index_cell, value_cell));
                self.free_temp(index_cell);
            },
            _ => return Err("Invalid assignment target".to_string()),
        }
        
        self.free_temp(value_cell);
        Ok(())
    }
    
    fn gen_for(
        &mut self,
        init: ASTNode,
        condition: ASTNode,
        increment: ASTNode,
        body: ASTNode
    ) -> Result<(), String> {
        self.emit_comment("For loop start");
        
        // Init
        self.generate_node(init)?;
        
        // Allocate condition cell
        let cond_cell = self.allocate_temp();
        
        // Evaluate initial condition
        let cond_result = self.eval_expression(condition.clone())?;
        self.emit(IRInstruction::Copy(cond_result, cond_cell));
        self.free_temp(cond_result);
        
        // Push loop context
        self.loop_stack.push(LoopContext {
            condition_cell: cond_cell,
            label: "for".to_string(),
        });
        
        // Loop start
        self.emit(IRInstruction::LoopStart(cond_cell));
        
        // Body
        self.generate_node(body)?;
        
        // Increment
        self.generate_node(increment)?;
        
        // Re-evaluate condition
        let cond_result = self.eval_expression(condition)?;
        self.emit(IRInstruction::Copy(cond_result, cond_cell));
        self.free_temp(cond_result);
        
        // Loop end
        self.emit(IRInstruction::LoopEnd(cond_cell));
        
        // Pop loop context
        self.loop_stack.pop();
        self.free_temp(cond_cell);
        
        self.emit_comment("For loop end");
        Ok(())
    }
    
    fn gen_while(&mut self, condition: ASTNode, body: ASTNode) -> Result<(), String> {
        self.emit_comment("While loop start");
        
        let cond_cell = self.allocate_temp();
        
        // Evaluate condition
        let cond_result = self.eval_expression(condition.clone())?;
        self.emit(IRInstruction::Copy(cond_result, cond_cell));
        self.free_temp(cond_result);
        
        // Push loop context
        self.loop_stack.push(LoopContext {
            condition_cell: cond_cell,
            label: "while".to_string(),
        });
        
        // Loop
        self.emit(IRInstruction::LoopStart(cond_cell));
        self.generate_node(body)?;
        
        // Re-evaluate condition
        let cond_result = self.eval_expression(condition)?;
        self.emit(IRInstruction::Copy(cond_result, cond_cell));
        self.free_temp(cond_result);
        
        self.emit(IRInstruction::LoopEnd(cond_cell));
        
        // Pop loop context
        self.loop_stack.pop();
        self.free_temp(cond_cell);
        
        self.emit_comment("While loop end");
        Ok(())
    }
    
    fn gen_if(
        &mut self,
        condition: ASTNode,
        then_branch: ASTNode,
        else_branch: Option<ASTNode>
    ) -> Result<(), String> {
        self.emit_comment("If statement start");
        
        let cond_cell = self.eval_expression(condition)?;
        let flag_cell = self.allocate_temp();
        
        // Copy condition to flag
        self.emit(IRInstruction::Copy(cond_cell, flag_cell));
        
        // Then branch: while(flag) { ... flag=0 }
        self.emit(IRInstruction::LoopStart(flag_cell));
        self.generate_node(then_branch)?;
        self.emit(IRInstruction::Set(flag_cell, 0));
        self.emit(IRInstruction::LoopEnd(flag_cell));
        
        // Else branch (if exists)
        if let Some(else_b) = else_branch {
            self.emit_comment("Else branch");
            let else_flag = self.allocate_temp();
            
            // else_flag = (cond_cell == 0) ? 1 : 0
            self.emit(IRInstruction::Set(else_flag, 1));
            self.emit(IRInstruction::LoopStart(cond_cell));
            self.emit(IRInstruction::Set(else_flag, 0));
            self.emit(IRInstruction::Set(cond_cell, 0));
            self.emit(IRInstruction::LoopEnd(cond_cell));
            
            self.emit(IRInstruction::LoopStart(else_flag));
            self.generate_node(else_b)?;
            self.emit(IRInstruction::Set(else_flag, 0));
            self.emit(IRInstruction::LoopEnd(else_flag));
            
            self.free_temp(else_flag);
        }
        
        self.free_temp(cond_cell);
        self.free_temp(flag_cell);
        
        self.emit_comment("If statement end");
        Ok(())
    }
    
    fn gen_putchar(&mut self, expr: ASTNode) -> Result<(), String> {
        let cell = self.eval_expression(expr)?;
        self.emit(IRInstruction::Output(cell));
        self.free_temp(cell);
        Ok(())
    }
    
    fn gen_break(&mut self) -> Result<(), String> {
        if let Some(ctx) = self.loop_stack.last() {
            self.emit(IRInstruction::Set(ctx.condition_cell, 0));
            Ok(())
        } else {
            Err("Break outside of loop".to_string())
        }
    }
    
    fn gen_continue(&mut self) -> Result<(), String> {
        // Continue is complex in BF - for now, emit comment
        // Proper implementation requires restructuring loop bodies
        self.emit_comment("Continue (requires loop restructure)");
        Ok(())
    }
    
    // ========================================================================
    // EXPRESSION GENERATORS
    // ========================================================================
    
    fn gen_literal_int(&mut self, n: i64) -> Result<usize, String> {
        let cell = self.allocate_temp();
        self.emit(IRInstruction::Set(cell, n));
        Ok(cell)
    }
    
    fn gen_literal_char(&mut self, c: char) -> Result<usize, String> {
        let cell = self.allocate_temp();
        self.emit(IRInstruction::Set(cell, c as i64));
        Ok(cell)
    }
    
    fn gen_literal_string(&mut self, s: &str) -> Result<usize, String> {
        // Allocate cells for string
        let base = self.cell_counter;
        for (i, ch) in s.chars().enumerate() {
            self.emit(IRInstruction::Set(base + i, ch as i64));
        }
        self.cell_counter += s.len();
        Ok(base)
    }
    
    fn gen_identifier(&mut self, name: &str) -> Result<usize, String> {
        // Return the cell containing the variable
        self.var_offsets.get(name)
            .copied()
            .ok_or_else(|| format!("Undefined variable: {}", name))
    }
    
    fn gen_array_access(&mut self, name: &str, index: ASTNode) -> Result<usize, String> {
        let base = *self.var_offsets.get(name)
            .ok_or_else(|| format!("Undefined array: {}", name))?;
        let index_cell = self.eval_expression(index)?;
        let result_cell = self.allocate_temp();
        
        self.emit(IRInstruction::ArrayLoad(base, index_cell, result_cell));
        self.free_temp(index_cell);
        
        Ok(result_cell)
    }
    
    fn gen_binary_op(&mut self, op: Operations, left: ASTNode, right: ASTNode) -> Result<usize, String> {
        use Operations::*;
        
        let left_cell = self.eval_expression(left)?;
        let right_cell = self.eval_expression(right)?;
        let result_cell = self.allocate_temp();
        
        match op {
            Add => self.gen_add(left_cell, right_cell, result_cell),
            Subtract => self.gen_subtract(left_cell, right_cell, result_cell),
            Multiply => self.gen_multiply(left_cell, right_cell, result_cell),
            Divide => self.gen_divide(left_cell, right_cell, result_cell),
            Modulus => self.gen_modulus(left_cell, right_cell, result_cell),
            Equal => self.gen_equal(left_cell, right_cell, result_cell),
            NotEqual => self.gen_not_equal(left_cell, right_cell, result_cell),
            LessThan => self.gen_less_than(left_cell, right_cell, result_cell),
            GreaterThan => self.gen_greater_than(left_cell, right_cell, result_cell),
            LessThanOrEqual => self.gen_less_equal(left_cell, right_cell, result_cell),
            GreaterThanOrEqual => self.gen_greater_equal(left_cell, right_cell, result_cell),
            And => self.gen_logical_and(left_cell, right_cell, result_cell),
            Or => self.gen_logical_or(left_cell, right_cell, result_cell),
            _ => return Err(format!("Unsupported binary operation: {:?}", op)),
        }?;
        
        self.free_temp(left_cell);
        self.free_temp(right_cell);
        
        Ok(result_cell)
    }
    
    fn gen_add(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
        // result = left + right
        self.emit(IRInstruction::Copy(left, result));
        
        // Add right to result (non-destructive)
        let temp = self.allocate_temp();
        self.emit(IRInstruction::Copy(right, temp));
        self.emit(IRInstruction::LoopStart(temp));
        self.emit(IRInstruction::Add(result, 1));
        self.emit(IRInstruction::Add(temp, -1));
        self.emit(IRInstruction::LoopEnd(temp));
        self.free_temp(temp);
        
        Ok(())
    }
    
    fn gen_subtract(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
        // result = left - right
        self.emit(IRInstruction::Copy(left, result));
        
        let temp = self.allocate_temp();
        self.emit(IRInstruction::Copy(right, temp));
        self.emit(IRInstruction::LoopStart(temp));
        self.emit(IRInstruction::Add(result, -1));
        self.emit(IRInstruction::Add(temp, -1));
        self.emit(IRInstruction::LoopEnd(temp));
        self.free_temp(temp);
        
        Ok(())
    }
    
    fn gen_multiply(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
        // result = left * right (using nested loops)
        self.emit(IRInstruction::Set(result, 0));
        
        let left_copy = self.allocate_temp();
        let right_copy = self.allocate_temp();
        let inner_temp = self.allocate_temp();
        
        self.emit(IRInstruction::Copy(left, left_copy));
        
        // Outer loop: for each unit in left
        self.emit(IRInstruction::LoopStart(left_copy));
        
        // Inner loop: add right to result
        self.emit(IRInstruction::Copy(right, right_copy));
        self.emit(IRInstruction::LoopStart(right_copy));
        self.emit(IRInstruction::Add(result, 1));
        self.emit(IRInstruction::Add(right_copy, -1));
        self.emit(IRInstruction::LoopEnd(right_copy));
        
        self.emit(IRInstruction::Add(left_copy, -1));
        self.emit(IRInstruction::LoopEnd(left_copy));
        
        self.free_temp(left_copy);
        self.free_temp(right_copy);
        self.free_temp(inner_temp);
        
        Ok(())
    }
    
    fn gen_divide(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
        // Division is complex - simplified version
        self.emit_comment("Division (simplified)");
        self.emit(IRInstruction::Set(result, 0));
        
        let dividend = self.allocate_temp();
        let divisor = self.allocate_temp();
        
        self.emit(IRInstruction::Copy(left, dividend));
        self.emit(IRInstruction::Copy(right, divisor));
        
        // Repeatedly subtract divisor from dividend
        // This is a simplified algorithm
        self.emit_comment("Division loop");
        
        self.free_temp(dividend);
        self.free_temp(divisor);
        
        Ok(())
    }
    
    fn gen_modulus(&mut self, left: usize, _right: usize, result: usize) -> Result<(), String> {
        // Modulus using division algorithm
        self.emit_comment("Modulus (simplified)");
        self.emit(IRInstruction::Copy(left, result));
        Ok(())
    }
    
    fn gen_equal(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
        // result = (left == right) ? 1 : 0
        self.emit_comment("Equal comparison");
        
        let diff = self.allocate_temp();
        self.gen_subtract(left, right, diff)?;
        
        // If diff == 0, then equal
        self.emit(IRInstruction::Set(result, 1));
        self.emit(IRInstruction::LoopStart(diff));
        self.emit(IRInstruction::Set(result, 0));
        self.emit(IRInstruction::Set(diff, 0));
        self.emit(IRInstruction::LoopEnd(diff));
        
        self.free_temp(diff);
        Ok(())
    }
    
    fn gen_not_equal(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
        self.gen_equal(left, right, result)?;
        
        // Invert result
        let temp = self.allocate_temp();
        self.emit(IRInstruction::Set(temp, 1));
        self.emit(IRInstruction::LoopStart(result));
        self.emit(IRInstruction::Set(temp, 0));
        self.emit(IRInstruction::Set(result, 0));
        self.emit(IRInstruction::LoopEnd(result));
        self.emit(IRInstruction::Copy(temp, result));
        self.free_temp(temp);
        
        Ok(())
    }
    
    fn gen_less_than(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
    // result = (left < right) ? 1 : 0
    let left_copy = self.allocate_temp();
    let right_copy = self.allocate_temp();
    let flag = self.allocate_temp();
    
    self.emit(IRInstruction::Copy(left, left_copy));
    self.emit(IRInstruction::Copy(right, right_copy));
    self.emit(IRInstruction::Set(result, 0));
    self.emit(IRInstruction::Set(flag, 1));
    
    // Simultaneously decrement both until one reaches zero
    self.emit(IRInstruction::LoopStart(right_copy));
    self.emit(IRInstruction::LoopStart(left_copy));
    self.emit(IRInstruction::Add(right_copy, -1));
    self.emit(IRInstruction::Add(left_copy, -1));
    self.emit(IRInstruction::Set(flag, 0));
    self.emit(IRInstruction::LoopEnd(left_copy));
    
    // If right_copy still has value, left < right
    self.emit(IRInstruction::LoopStart(flag));
    self.emit(IRInstruction::Set(flag, 0));
    self.emit(IRInstruction::LoopEnd(flag));
    
    self.emit(IRInstruction::LoopStart(right_copy));
    self.emit(IRInstruction::Set(result, 1));
    self.emit(IRInstruction::Set(right_copy, 0));
    self.emit(IRInstruction::LoopEnd(right_copy));
    
    self.emit(IRInstruction::LoopEnd(right_copy));
    
    self.free_temp(left_copy);
    self.free_temp(right_copy);
    self.free_temp(flag);
    Ok(())
}

    
    fn gen_greater_than(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
    // left > right is equivalent to right < left
    self.gen_less_than(right, left, result)
}

    
    fn gen_less_equal(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
        // result = (left <= right) ? 1 : 0
        self.gen_greater_than(left, right, result)?;
        
        // Invert
        let temp = self.allocate_temp();
        self.emit(IRInstruction::Set(temp, 1));
        self.emit(IRInstruction::LoopStart(result));
        self.emit(IRInstruction::Set(temp, 0));
        self.emit(IRInstruction::Set(result, 0));
        self.emit(IRInstruction::LoopEnd(result));
        self.emit(IRInstruction::Copy(temp, result));
        self.free_temp(temp);
        
        Ok(())
    }
    
    fn gen_greater_equal(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
        self.gen_less_than(left, right, result)?;
        
        // Invert
        let temp = self.allocate_temp();
        self.emit(IRInstruction::Set(temp, 1));
        self.emit(IRInstruction::LoopStart(result));
        self.emit(IRInstruction::Set(temp, 0));
        self.emit(IRInstruction::Set(result, 0));
        self.emit(IRInstruction::LoopEnd(result));
        self.emit(IRInstruction::Copy(temp, result));
        self.free_temp(temp);
        
        Ok(())
    }
    
    fn gen_logical_and(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
        // result = (left && right) ? 1 : 0
        self.emit(IRInstruction::Set(result, 0));
        
        let temp = self.allocate_temp();
        self.emit(IRInstruction::Copy(left, temp));
        self.emit(IRInstruction::LoopStart(temp));
        
        let temp2 = self.allocate_temp();
        self.emit(IRInstruction::Copy(right, temp2));
        self.emit(IRInstruction::LoopStart(temp2));
        self.emit(IRInstruction::Set(result, 1));
        self.emit(IRInstruction::Set(temp2, 0));
        self.emit(IRInstruction::LoopEnd(temp2));
        
        self.emit(IRInstruction::Set(temp, 0));
        self.emit(IRInstruction::LoopEnd(temp));
        
        self.free_temp(temp);
        self.free_temp(temp2);
        
        Ok(())
    }
    
    fn gen_logical_or(&mut self, left: usize, right: usize, result: usize) -> Result<(), String> {
        // result = (left || right) ? 1 : 0
        self.emit(IRInstruction::Set(result, 0));
        
        let temp_left = self.allocate_temp();
        let temp_right = self.allocate_temp();
        
        self.emit(IRInstruction::Copy(left, temp_left));
        self.emit(IRInstruction::LoopStart(temp_left));
        self.emit(IRInstruction::Set(result, 1));
        self.emit(IRInstruction::Set(temp_left, 0));
        self.emit(IRInstruction::LoopEnd(temp_left));
        
        self.emit(IRInstruction::Copy(right, temp_right));
        self.emit(IRInstruction::LoopStart(temp_right));
        self.emit(IRInstruction::Set(result, 1));
        self.emit(IRInstruction::Set(temp_right, 0));
        self.emit(IRInstruction::LoopEnd(temp_right));
        
        self.free_temp(temp_left);
        self.free_temp(temp_right);
        
        Ok(())
    }
    
    fn gen_unary_op(&mut self, op: Operations, expr: ASTNode) -> Result<usize, String> {
        use Operations::*;
        
        let expr_cell = self.eval_expression(expr)?;
        let result_cell = self.allocate_temp();
        
        match op {
            Not => {
                // result = !expr (0->1, nonzero->0)
                self.emit(IRInstruction::Set(result_cell, 1));
                
                let temp = self.allocate_temp();
                self.emit(IRInstruction::Copy(expr_cell, temp));
                self.emit(IRInstruction::LoopStart(temp));
                self.emit(IRInstruction::Set(result_cell, 0));
                self.emit(IRInstruction::Set(temp, 0));
                self.emit(IRInstruction::LoopEnd(temp));
                self.free_temp(temp);
            },
            Increment => {
                // ++expr (modifies in-place)
                self.emit(IRInstruction::Add(expr_cell, 1));
                self.free_temp(result_cell);
                return Ok(expr_cell);
            },
            Decrement => {
                // --expr (modifies in-place)
                self.emit(IRInstruction::Add(expr_cell, -1));
                self.free_temp(result_cell);
                return Ok(expr_cell);
            },
            Subtract => {
                // -expr (negate)
                self.emit(IRInstruction::Set(result_cell, 0));
                
                let temp = self.allocate_temp();
                self.emit(IRInstruction::Copy(expr_cell, temp));
                self.emit(IRInstruction::LoopStart(temp));
                self.emit(IRInstruction::Add(result_cell, -1));
                self.emit(IRInstruction::Add(temp, -1));
                self.emit(IRInstruction::LoopEnd(temp));
                self.free_temp(temp);
            },
            _ => {
                self.free_temp(result_cell);
                return Err(format!("Unsupported unary operation: {:?}", op));
            }
        }
        
        self.free_temp(expr_cell);
        Ok(result_cell)
    }
    
    fn gen_ternary(
        &mut self,
        condition: ASTNode,
        true_expr: ASTNode,
        false_expr: ASTNode
    ) -> Result<usize, String> {
        let cond_cell = self.eval_expression(condition)?;
        let result_cell = self.allocate_temp();
        let flag = self.allocate_temp();
        
        // If condition is true
        self.emit(IRInstruction::Copy(cond_cell, flag));
        self.emit(IRInstruction::LoopStart(flag));
        let true_val = self.eval_expression(true_expr)?;
        self.emit(IRInstruction::Copy(true_val, result_cell));
        self.free_temp(true_val);
        self.emit(IRInstruction::Set(flag, 0));
        self.emit(IRInstruction::LoopEnd(flag));
        
        // If condition is false
        self.emit(IRInstruction::Set(flag, 1));
        self.emit(IRInstruction::LoopStart(cond_cell));
        self.emit(IRInstruction::Set(flag, 0));
        self.emit(IRInstruction::Set(cond_cell, 0));
        self.emit(IRInstruction::LoopEnd(cond_cell));
        
        self.emit(IRInstruction::LoopStart(flag));
        let false_val = self.eval_expression(false_expr)?;
        self.emit(IRInstruction::Copy(false_val, result_cell));
        self.free_temp(false_val);
        self.emit(IRInstruction::Set(flag, 0));
        self.emit(IRInstruction::LoopEnd(flag));
        
        self.free_temp(flag);
        
        Ok(result_cell)
    }
    
    fn gen_function_call(&mut self, name: &str, _args: Vec<ASTNode>) -> Result<usize, String> {
        match name {
            "getchar" => {
                let cell = self.allocate_temp();
                self.emit(IRInstruction::Input(cell));
                Ok(cell)
            },
            _ => Err(format!("Unknown function: {}", name)),
        }
    }
    
    // ========================================================================
    // MEMORY MANAGEMENT HELPERS
    // ========================================================================
    
    fn allocate_temp(&mut self) -> usize {
        // Try to reuse a temp cell
        for (i, &in_use) in self.temp_in_use.iter().enumerate() {
            if !in_use {
                self.temp_in_use[i] = true;
                return self.temp_pool[i];
            }
        }
        
        // Allocate new temp
        let cell = self.cell_counter;
        self.cell_counter += 1;
        self.temp_pool.push(cell);
        self.temp_in_use.push(true);
        cell
    }
    
    fn free_temp(&mut self, cell: usize) {
        // Mark temp as free for reuse
        if let Some(pos) = self.temp_pool.iter().position(|&c| c == cell) {
            self.temp_in_use[pos] = false;
        }
    }
    
    // ========================================================================
    // IR EMISSION HELPERS
    // ========================================================================
    
    fn emit(&mut self, instruction: IRInstruction) {
        self.instructions.push(instruction);
    }
    
    fn emit_comment(&mut self, text: &str) {
        self.emit(IRInstruction::Comment(text.to_string()));
    }
    
    pub fn get_instructions(&self) -> &[IRInstruction] {
        &self.instructions
    }
    
    pub fn print_ir(&self) {
        println!("\\n=== IR Instructions ===\\n");
        for (i, instr) in self.instructions.iter().enumerate() {
            println!("{:4}: {:?}", i, instr);
        }
        println!("\\nTotal cells used: {}\\n", self.cell_counter);
    }
}

use std::collections::{HashSet};

// ============================================================================
// DEAD CODE ELIMINATION PASS
// ============================================================================
#[allow(dead_code)]
pub struct DeadCodeEliminator {
    live_cells: HashSet<usize>,
    used_instructions: HashSet<usize>,
    cell_last_write: HashMap<usize, usize>,
    cell_last_read: HashMap<usize, usize>,
}

impl DeadCodeEliminator {
    pub fn new() -> Self {
        Self {
            live_cells: HashSet::new(),
            used_instructions: HashSet::new(),
            cell_last_write: HashMap::new(),
            cell_last_read: HashMap::new(),
        }
    }
    
    pub fn eliminate(&mut self, instructions: Vec<IRInstruction>) -> Vec<IRInstruction> {
        println!("\n=== Dead Code Elimination Pass ===");
        println!("Input: {} instructions", instructions.len());
        
        // Pass 1: Mark live cells
        self.mark_live_cells(&instructions);
        
        // Pass 2: Remove dead writes
        let after_dead_writes = self.remove_dead_writes(instructions);
        println!("After dead write elimination: {} instructions", after_dead_writes.len());
        
        // Pass 3: Remove unreachable code
        let after_unreachable = self.remove_unreachable_code(after_dead_writes);
        println!("After unreachable code elimination: {} instructions", after_unreachable.len());
        
        // Pass 4: Remove redundant operations
        let after_redundant = self.remove_redundant_operations(after_unreachable);
        println!("After redundant operation elimination: {} instructions", after_redundant.len());
        
        // Pass 5: Remove duplicate declarations
        let final_ir = self.remove_duplicate_declarations(after_redundant);
        println!("Final: {} instructions", final_ir.len());
        
        final_ir
    }
    
    fn mark_live_cells(&mut self, instructions: &[IRInstruction]) {
        // Start with observable effects
        for (idx, instr) in instructions.iter().enumerate().rev() {
            match instr {
                IRInstruction::Output(cell) => {
                    self.live_cells.insert(*cell);
                    self.used_instructions.insert(idx);
                },
                IRInstruction::LoopStart(cell) | IRInstruction::LoopEnd(cell) => {
                    self.live_cells.insert(*cell);
                    self.used_instructions.insert(idx);
                },
                IRInstruction::ArrayStore(base, index, src) => {
                    self.live_cells.insert(*base);
                    self.live_cells.insert(*index);
                    self.live_cells.insert(*src);
                    self.used_instructions.insert(idx);
                },
                _ => {}
            }
        }
        
        // Propagate liveness backward
        for (idx, instr) in instructions.iter().enumerate().rev() {
            match instr {
                IRInstruction::Copy(src, dest) => {
                    if self.live_cells.contains(dest) {
                        self.live_cells.insert(*src);
                        self.used_instructions.insert(idx);
                    }
                },
                IRInstruction::Add(cell, _) | IRInstruction::Set(cell, _) => {
                    if self.live_cells.contains(cell) {
                        self.used_instructions.insert(idx);
                    }
                },
                _ => {}
            }
        }
    }
    
    fn remove_dead_writes(&self, instructions: Vec<IRInstruction>) -> Vec<IRInstruction> {
        let mut result = Vec::new();
        let mut cell_values: HashMap<usize, Option<i64>> = HashMap::new();
        
        for instr in instructions {
            let mut keep = true;
            
            match &instr {
                IRInstruction::Set(cell, val) => {
                    // Check if this is a dead write
                    if !self.live_cells.contains(cell) {
                        keep = false;
                    } else {
                        cell_values.insert(*cell, Some(*val));
                    }
                },
                IRInstruction::Copy(_src, dest) => {
                    if !self.live_cells.contains(dest) {
                        keep = false;
                    }
                },
                _ => {}
            }
            
            if keep {
                result.push(instr);
            }
        }
        
        result
    }
    
    fn remove_unreachable_code(&self, instructions: Vec<IRInstruction>) -> Vec<IRInstruction> {
        let mut result = Vec::new();
        
        for instr in instructions {
            match &instr {
                IRInstruction::Comment(_) | IRInstruction::Label(_) => {
                    result.push(instr);
                },
                _ => {
                    result.push(instr);
                }
            }
        }
        
        result
    }
    
    fn remove_redundant_operations(&self, instructions: Vec<IRInstruction>) -> Vec<IRInstruction> {
        let mut result = Vec::new();
        let mut cell_values: HashMap<usize, Option<i64>> = HashMap::new();
        
        for instr in instructions {
            let mut keep = true;
            
            match &instr {
                IRInstruction::Set(cell, 0) => {
                    if let Some(Some(0)) = cell_values.get(cell) {
                        keep = false; // Already 0
                    } else {
                        cell_values.insert(*cell, Some(0));
                    }
                },
                IRInstruction::Set(cell, val) => {
                    if let Some(Some(prev_val)) = cell_values.get(cell) {
                        if prev_val == val {
                            keep = false;
                        }
                    }
                    cell_values.insert(*cell, Some(*val));
                },
                IRInstruction::Add(_, 0) => {
                    keep = false; // No-op
                },
                IRInstruction::Copy(src, dest) if src == dest => {
                    keep = false; // Redundant
                },
                IRInstruction::Copy(_, dest) => {
                    cell_values.insert(*dest, None);
                },
                IRInstruction::LoopStart(_) | IRInstruction::LoopEnd(_) => {
                    cell_values.clear();
                },
                _ => {}
            }
            
            if keep {
                result.push(instr);
            }
        }
        
        result
    }
    
    fn remove_duplicate_declarations(&self, instructions: Vec<IRInstruction>) -> Vec<IRInstruction> {
        let mut result = Vec::new();
        let mut declared_vars: HashSet<String> = HashSet::new();
        
        for instr in instructions {
            match &instr {
                IRInstruction::Comment(text) => {
                    if text.starts_with("Declare ") {
                        if let Some(var_name) = text.split_whitespace().nth(1) {
                            if declared_vars.contains(var_name) {
                                continue; // Skip duplicate
                            } else {
                                declared_vars.insert(var_name.to_string());
                            }
                        }
                    }
                    result.push(instr);
                },
                _ => {
                    result.push(instr);
                }
            }
        }
        
        result
    }
}

pub fn optimize_ir(instructions: Vec<IRInstruction>) -> Vec<IRInstruction> {
    let mut optimizer = DeadCodeEliminator::new();
    optimizer.eliminate(instructions)
}

// ============================================================================
// BRAINFUCK CODE GENERATOR
// ============================================================================

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
            optimize: true,
        }
    }
    
    pub fn with_optimization(mut self, optimize: bool) -> Self {
        self.optimize = optimize;
        self
    }
    
    // MAIN ENTRY POINT
    pub fn generate(&mut self, instructions: &[IRInstruction]) -> String {
        println!("\n=== Brainfuck Code Generation ===");
        println!("IR Instructions: {}", instructions.len());
        
        for instr in instructions {
            self.generate_instruction(instr);
        }
        
        if self.optimize {
            self.output = self.peephole_optimize(self.output.clone());
        }
        
        println!("Brainfuck code length: {} characters", self.output.len());
        
        self.output.clone()
    }
    
    // INSTRUCTION DISPATCHER
    fn generate_instruction(&mut self, instr: &IRInstruction) {
        match instr {
            IRInstruction::Goto(cell) => self.gen_goto(*cell),
            IRInstruction::Set(cell, value) => self.gen_set(*cell, *value),
            IRInstruction::Add(cell, value) => self.gen_add(*cell, *value),
            IRInstruction::Copy(src, dest) => self.gen_copy(*src, *dest),
            IRInstruction::Move(src, dest) => self.gen_move(*src, *dest),
            IRInstruction::LoopStart(cell) => self.gen_loop_start(*cell),
            IRInstruction::LoopEnd(cell) => self.gen_loop_end(*cell),
            IRInstruction::Output(cell) => self.gen_output(*cell),
            IRInstruction::Input(cell) => self.gen_input(*cell),
            IRInstruction::ArrayLoad(base, index, dest) => {
                self.gen_array_load(*base, *index, *dest)
            },
            IRInstruction::ArrayStore(base, index, src) => {
                self.gen_array_store(*base, *index, *src)
            },
            IRInstruction::Comment(text) => self.gen_comment(text),
            IRInstruction::Label(label) => self.gen_label(label),
        }
    }
    
    // BASIC OPERATIONS
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
        self.emit("[-]"); // Clear cell
        
        if value > 0 {
            self.emit(&"+".repeat(value as usize));
        } else if value < 0 {
            self.emit(&"-".repeat((-value) as usize));
        }
    }
    
    fn gen_add(&mut self, cell: usize, value: i64) {
        if value == 0 {
            return;
        }
        
        self.gen_goto(cell);
        
        if value > 0 {
            self.emit(&"+".repeat(value as usize));
        } else {
            self.emit(&"-".repeat((-value) as usize));
        }
    }
    
    fn gen_copy(&mut self, src: usize, dest: usize) {
        if src == dest {
            return;
        }
        
        // Non-destructive copy using temp cells
        let temp1 = src.max(dest) + 1;
        let _temp2 = temp1 + 1;
        
        // Clear temps and dest
        self.gen_set(temp1, 0);
        self.gen_set(dest, 0);
        
        // Copy src to temp1 and dest
        self.gen_goto(src);
        self.emit("[-");
        self.gen_goto(temp1);
        self.emit("+");
        self.gen_goto(dest);
        self.emit("+");
        self.gen_goto(src);
        self.emit("]");
        
        // Restore src from temp1
        self.gen_goto(temp1);
        self.emit("[-");
        self.gen_goto(src);
        self.emit("+");
        self.gen_goto(temp1);
        self.emit("]");
    }
    
    fn gen_move(&mut self, src: usize, dest: usize) {
        if src == dest {
            return;
        }
        
        self.gen_set(dest, 0);
        self.gen_goto(src);
        self.emit("[-");
        self.gen_goto(dest);
        self.emit("+");
        self.gen_goto(src);
        self.emit("]");
    }
    
    fn gen_loop_start(&mut self, cell: usize) {
        self.gen_goto(cell);
        self.emit("[");
    }
    
    fn gen_loop_end(&mut self, cell: usize) {
        self.gen_goto(cell);
        self.emit("]");
    }
    
    fn gen_output(&mut self, cell: usize) {
        self.gen_goto(cell);
        self.emit(".");
    }
    
    fn gen_input(&mut self, cell: usize) {
        self.gen_goto(cell);
        self.emit(",");
    }
    
    // ARRAY OPERATIONS (Simplified)
    fn gen_array_load(&mut self, base: usize, _index_cell: usize, dest: usize) {
        // Simplified: for now, just copy base to dest
        // Full implementation requires complex pointer arithmetic
        self.gen_copy(base, dest);
    }
    
    fn gen_array_store(&mut self, base: usize, _index_cell: usize, src: usize) {
        // Simplified: copy src to base
        self.gen_copy(src, base);
    }
    
    // METADATA
    fn gen_comment(&mut self, text: &str) {
        // BF comments (ignored by interpreters)
        self.emit(&format!("\n# {}\n", text));
    }
    
    fn gen_label(&mut self, label: &str) {
        self.emit(&format!("\n## {}\n", label));
    }
    
    // PEEPHOLE OPTIMIZATION
    fn peephole_optimize(&self, code: String) -> String {
        let mut result = code;
        let original_len = result.len();
        
        // Remove no-ops
        result = result.replace("+-", "");
        result = result.replace("-+", "");
        result = result.replace("<>", "");
        result = result.replace("><", "");
        result = result.replace("[]", "");
        
        // Merge consecutive (simple version - can be enhanced)
        // This is a placeholder - full optimization would use RLE
        
        let optimized_len = result.len();
        println!("Optimized: {} -> {} chars", original_len, optimized_len);
        
        result
    }
    
    fn emit(&mut self, code: &str) {
        self.output.push_str(code);
    }
    
    pub fn get_output(&self) -> &str {
        &self.output
    }
}

// PUBLIC API
pub fn generate_brainfuck(instructions: Vec<IRInstruction>) -> Result<String, String> {
    let mut generator = BrainfuckGenerator::new();
    let code = generator.generate(&instructions);
    Ok(code)
}

pub fn compile_to_brainfuck(source: &str) -> Result<String, String> {
    // PASS 1: Parse
    let ast = parse_program(source).ok_or("Parsing failed")?;
    
    let mut ast_vec = if let ASTNode::Program(decls) = ast {
        decls
    } else {
        vec![ast]
    };
    
    // PASS 2: Semantic Analysis - FIX HERE
    let _validated_ast = run_ast(&mut ast_vec)
        .map_err(|e| format!("{}", e))?;  // Convert SemanticError to String
    
    // PASS 3: Function Inlining (source-to-source)
    let inlined_source = clean_source(source);
    let inlined_ast = parse_program(&inlined_source).ok_or("Parse inlined failed")?;
    
    let mut inlined_ast_vec = if let ASTNode::Program(decls) = inlined_ast {
        decls
    } else {
        vec![inlined_ast]
    };
    
    // FIX HERE TOO
    let final_ast = run_ast(&mut inlined_ast_vec)
        .map_err(|e| format!("{}", e))?;  // Convert SemanticError to String
    
    // PASS 4: Generate IR
    let mut ir_gen = IRGenerator::new();
    let program_node = ASTNode::Program(final_ast);
    ir_gen.generate(program_node)?;
    ir_gen.print_ir();
    let ir = ir_gen.get_instructions().to_vec();
    
    // PASS 5: Dead Code Elimination
    let optimized_ir = optimize_ir(ir);
    
    // PASS 6: Generate Brainfuck
    let brainfuck = generate_brainfuck(optimized_ir)?;
    
    Ok(brainfuck)
}


fn main() {
    let source = r"
        int main(){
        int i =65;
        while(i < 91){
        putchar(i);
        i = i +1;
}
        return 0;
}
    ";
    let brainfuck = compile_to_brainfuck(source).expect("Compilation failed");
    
    // Output
    println!("\n=== BRAINFUCK CODE ===\n{}", brainfuck);
}
