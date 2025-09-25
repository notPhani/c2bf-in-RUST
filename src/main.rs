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

// next is to implement the inlining function : since we already walked the tree once it's best to pattern match and 
/*
(*
 * inline : program -> program
 *
 * remove all functions, assuming that the program is well typed
 *)
let inline =
    (* for temporary variables in expressions *)
    let tmp_counter = ref 0 in

    let get_tmp_var () =
        incr tmp_counter;
        sprintf "%d_tmp" (!tmp_counter)
    in

    (* for local variables inside functions *)
    let call_counter = ref 0 in

    let name_local_var call_id name = sprintf "%d_loc_%s" call_id name in

    (* inline_expression : inline_environment -> expression -> program * expression *)
    let rec inline_expression env expr =
        let inline_unary_expr expr construct =
            let pre_expr, new_expr = inline_expression env expr in
            pre_expr, construct new_expr
        in
        let inline_binary_expr left right construct =
            let pre_left, new_left = inline_expression env left in
            let pre_right, new_right = inline_expression env right in
            pre_left @ pre_right, construct new_left new_right
        in

This as far as my knowledge does this :: 
 */
fn main() {
    let extra_test_cases = vec![
        // Fix your first test - it uses undeclared x
        ("Basic valid code fixed", r#"int main() { int x = 42; return x; }"#),
        
        // More type conversion edge cases
        ("Char literal assignment", r#"int main() { char c = 'Z'; int x = c; return x; }"#),
        ("Large char value", r#"int main() { char c = 128; return c; }"#),
        ("String length check", r#"int main() { string s = "hello world"; return 0; }"#),
        
        // Function edge cases
        ("Recursive function", r#"int fib(int n) { if (n <= 1) return n; return fib(n-1) + fib(n-2); } int main() { return fib(5); }"#),
        ("Function with no params", r#"int get_magic() { return 42; } int main() { return get_magic(); }"#),
        ("Function returning char", r#"char get_letter() { return 'A'; } int main() { char c = get_letter(); return c; }"#),
        
        // Array madness
        ("Multi-dimensional array", r#"int main() { int arr[3][3]; arr[1][2] = 5; return arr[1][2]; }"#),
        ("Array initialization", r#"int main() { int arr[3] = {1, 2, 3}; return arr[1]; }"#),
        ("String as array", r#"int main() { string s = "test"; return 0; }"#),
        ("Array bounds with variables", r#"int main() { int arr[5]; int i = 2; return arr[i]; }"#),
        
        // Control flow chaos
        ("Nested loops with break", r#"int main() { int x = 0; for(int i = 0; i < 3; i = i + 1) { while(x < 10) { x = x + 1; if(x == 5) break; } } return x; }"#),
        ("Complex if-else chain", r#"int main() { int x = 5; if(x > 10) return 1; else if(x > 5) return 2; else if(x == 5) return 3; else return 4; }"#),
        ("For loop with complex increment", r#"int main() { int sum = 0; for(int i = 0; i < 10; i = i + 2) { sum = sum + i; } return sum; }"#),
        
        // Expression complexity
        ("Complex arithmetic", r#"int main() { int x = (5 + 3) * 2 - 1; return x; }"#),
        ("Ternary operator", r#"int main() { int x = 5; int y = (x > 3) ? 10 : 20; return y; }"#),
        ("Mixed type expressions", r#"int main() { int x = 5; char c = 'A'; int result = x + c; return result; }"#),
        
        // Scope and shadowing tests
        ("Deep variable shadowing", r#"int x = 1; int main() { int x = 2; { int x = 3; { int x = 4; return x; } } }"#),
        ("Function parameter shadowing", r#"int x = 100; int test(int x) { return x + 1; } int main() { return test(5); }"#),
        ("Loop variable scope", r#"int main() { int x = 0; for(int x = 1; x < 3; x = x + 1) { } return x; }"#),
        
        // Initialization edge cases
        ("Multiple uninitialized variables", r#"int main() { int a, b, c; a = 1; return a + b; }"#),
        ("Assignment in declaration", r#"int main() { int x = 5, y = x + 1; return y; }"#),
        ("Array element initialization", r#"int main() { int arr[3]; arr[0] = 10; int x = arr[1]; return x; }"#),
        
        // Error combinations
        ("Function and array errors", r#"int main() { int x[5]; ghost_func(x["hello"]); return 0; }"#),
        ("Type and scope errors", r#"int main() { unknown_var = "string"; return unknown_var; }"#),
        ("Complex nested errors", r#"int f() { return; } int main() { int x; x = f(); return x[0]; }"#),
        
        // Edge cases for meme value
        ("Empty function body", r#"int main() { }"#),
        ("Only comments", r#"int main() { /* nothing here */ return 0; }"#),
        ("Weird but valid", r#"int main() { int x = 0; x = x = x = 42; return x; }"#),
    ];

    // Combine with your existing test_cases
    let  all_tests = extra_test_cases;

    for (i, (description, source)) in all_tests.iter().enumerate() {
        println!("Test {}: {}", i + 1, description);
        println!("Code: {}", source.trim());
        
        match parse_program(source) {
            Some(ast) => {
                match run_ast(&mut vec![ast]) {
                    Ok(_) => println!("Result: PASSED semantic analysis"),
                    Err(e) => println!("Result: FAILED - {}", e),
                }
            },
            None => println!("Result: FAILED - Parse error"),
        } 
        println!("{}", "-".repeat(60));
    }
}
