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
fn parse_assignment(tokens: &mut Vec<Token>) -> Option<ASTNode> {
    //Handles only array element assignment and identifier assignment, not variable declaration nor, array declaration.
    let left = parse_expression(tokens)?;
    
    if let Some(Token::Operator(Operations::Assign)) = tokens.first() {
        tokens.remove(0); // consume '='
        
        match left {
            ASTNode::Identifier(_) | ASTNode::ArrayAccess { .. } => {
                let right = parse_assignment(tokens)?; 
                Some(ASTNode::Assignment { 
                    target: Box::new(left), 
                    value: Box::new(right)
                })
            },
            _ => None,
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
    foldable_value: Option<ASTNode>,   // For constant propagation
    is_array: bool,                   // Array type checking
    array_dims: Option<Vec<usize>>,   // Array bounds validation
    initial_value: Option<Box<ASTNode>>, // Initialization storage
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FuncInfo {
    return_type: Keywords,            // Function return type
    params: Vec<(Keywords, String)>,  // Parameter validation & mapping
    body: Box<ASTNode>,              // Function inlining source
}
#[derive(Debug, Clone)]
struct Scope {
    label: String,                   // Hierarchical scope identification  
    variables: HashMap<String, VarInfo>, // Variable symbol table
    functions: HashMap<String, FuncInfo>, // Function symbol table
    inside_loop: bool,               // Loop context for break/continue
}

// Error types for semantic analysis
#[derive(Debug, Clone)]
enum SemanticError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    TypeMismatch { expected: Keywords, found: Keywords },
    DuplicateDeclaration(String),
    ArgumentCountMismatch { expected: usize, found: usize },
    BreakOutsideLoop,
    ContinueOutsideLoop,
    //ReturnTypeMismatch { expected: Keywords, found: Keywords },
    ArrayIndexNotInt,
    InvalidArraySize,
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
                write!(f, "This array index isnt an integer. Arrays like whole numbers only.")
            },
            SemanticError::InvalidArraySize => {
                write!(f, "This array size is invalid. Must be > 0; zero is a no-go here.")
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
    
    // Initialize global scope
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
        )?;
        rebuilt_ast.push(processed_node);
    }
    
    Ok(rebuilt_ast)
}

fn process_node(
    node: ASTNode,
    all_scopes: &mut Vec<Scope>,
    scope_stack: &mut Vec<String>,
    block_counter: &mut usize,
) -> Result<ASTNode, SemanticError> {
    match node {
        // Variable declarations - add to current scope
        ASTNode::VariableDeclaration { var_type, name, array_dims, initial_value } => {
            let current_scope_label = scope_stack.last().unwrap().clone();
            
            // Check for duplicate declaration in CURRENT scope only (shadowing allowed)
            if let Some(current_scope) = all_scopes.iter().find(|s| s.label == current_scope_label) {
                if current_scope.variables.contains_key(&name) {
                    return Err(SemanticError::DuplicateDeclaration(name));
                }
            }
            
            // Validate array dimensions
            let is_array = array_dims.is_some();
            if let Some(ref dims) = array_dims {
                for &size in dims {
                    if size == 0 {
                        return Err(SemanticError::InvalidArraySize);
                    }
                }
            }
            
            // Process initial value if present
            let processed_initial = if let Some(init_val) = initial_value {
                let processed = process_node(*init_val, all_scopes, scope_stack, block_counter)?;
                
                // Type check the initial value
                let init_type = get_expr_type(&processed, all_scopes, scope_stack)?;
                if !types_compatible(&var_type, &init_type) {
                    return Err(SemanticError::TypeMismatch { 
                        expected: var_type, 
                        found: init_type 
                    });
                }
                
                Some(Box::new(processed))
            } else {
                None
            };
            
            // Add to symbol table
            let var_info = VarInfo {
                var_type: var_type.clone(),
                foldable_value: None,
                is_array,
                array_dims: array_dims.clone(),
                initial_value: processed_initial.clone(),
            };
            
            // Find and update the current scope
            if let Some(current_scope) = all_scopes.iter_mut().find(|s| s.label == current_scope_label) {
                current_scope.variables.insert(name.clone(), var_info);
            }
            
            Ok(ASTNode::VariableDeclaration { var_type, name, array_dims, initial_value: processed_initial })
        },
        
        // Function declarations - add to current scope
        ASTNode::Function { return_type, name, params, body } => {
            let current_scope_label = scope_stack.last().unwrap().clone();
            
            // Check for duplicate function declaration
            if let Some(current_scope) = all_scopes.iter().find(|s| s.label == current_scope_label) {
                if current_scope.functions.contains_key(&name) {
                    return Err(SemanticError::DuplicateDeclaration(name));
                }
            }
            
            // Create function scope for parameters and body
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
                };
                func_scope.variables.insert(param_name.clone(), param_info);
            }
            
            all_scopes.push(func_scope);
            scope_stack.push(func_scope_label);
            
            // Process function body - FIXED DEREFERENCING HERE!
            let processed_body = if matches!(*body, ASTNode::Empty) {
                body // Function declaration without body
            } else {
                Box::new(process_node(*body, all_scopes, scope_stack, block_counter)?)
            };
            
            scope_stack.pop();
            
            // Add function to symbol table
            let func_info = FuncInfo {
                return_type: return_type.clone(),
                params: params.clone(),
                body: processed_body.clone(),
            };
            
            // Find and update the current scope
            if let Some(current_scope) = all_scopes.iter_mut().find(|s| s.label == current_scope_label) {
                current_scope.functions.insert(name.clone(), func_info);
            }
            
            Ok(ASTNode::Function { return_type, name, params, body: processed_body })
        },
        
        // Variable usage - check if exists
        ASTNode::Identifier(name) => {
            if lookup_variable(&name, all_scopes, scope_stack).is_none() {
                return Err(SemanticError::UndefinedVariable(name));
            }
            Ok(ASTNode::Identifier(name))
        },
        
        // Function calls - validate existence and arguments
        ASTNode::FunctionCall { name, args } => {
            let func_info = lookup_function(&name, all_scopes, scope_stack)
                .ok_or_else(|| SemanticError::UndefinedFunction(name.clone()))?;
            
            // Check argument count
            if args.len() != func_info.params.len() {
                return Err(SemanticError::ArgumentCountMismatch { 
                    expected: func_info.params.len(), 
                    found: args.len() 
                });
            }
            
            // Process and type-check arguments
            let mut processed_args = Vec::new();
            for (i, arg) in args.into_iter().enumerate() {
                let processed_arg = process_node(arg, all_scopes, scope_stack, block_counter)?;
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
        
        // Array access - validate index type
        ASTNode::ArrayAccess { name, index } => {
            let var_info = lookup_variable(&name, all_scopes, scope_stack)
                .ok_or_else(|| SemanticError::UndefinedVariable(name.clone()))?;
            
            if !var_info.is_array {
                return Err(SemanticError::TypeMismatch { 
                    expected: Keywords::Arr, 
                    found: var_info.var_type 
                });
            }
            
            let processed_index = process_node(*index, all_scopes, scope_stack, block_counter)?;
            let index_type = get_expr_type(&processed_index, all_scopes, scope_stack)?;
            
            if !matches!(index_type, Keywords::Int) {
                return Err(SemanticError::ArrayIndexNotInt);
            }
            
            Ok(ASTNode::ArrayAccess { name, index: Box::new(processed_index) })
        },
        
        // Assignments - type checking
        ASTNode::Assignment { target, value } => {
            let processed_target = process_node(*target, all_scopes, scope_stack, block_counter)?;
            let processed_value = process_node(*value, all_scopes, scope_stack, block_counter)?;
            
            let target_type = get_expr_type(&processed_target, all_scopes, scope_stack)?;
            let value_type = get_expr_type(&processed_value, all_scopes, scope_stack)?;
            
            if !types_compatible(&target_type, &value_type) {
                return Err(SemanticError::TypeMismatch { 
                    expected: target_type, 
                    found: value_type 
                });
            }
            
            Ok(ASTNode::Assignment { 
                target: Box::new(processed_target), 
                value: Box::new(processed_value) 
            })
        },
        
        // Control flow - break/continue validation
        ASTNode::Break => {
            // Check if we're inside any loop by traversing scope hierarchy
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
            // Check if we're inside any loop by traversing scope hierarchy
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
        
        // If statements - process all branches
        ASTNode::If { condition, then_branch, else_branch } => {
            let processed_condition = process_node(*condition, all_scopes, scope_stack, block_counter)?;
            let processed_then = process_node(*then_branch, all_scopes, scope_stack, block_counter)?;
            let processed_else = if let Some(else_node) = else_branch {
                Some(Box::new(process_node(*else_node, all_scopes, scope_stack, block_counter)?))
            } else {
                None
            };
            
            Ok(ASTNode::If {
                condition: Box::new(processed_condition),
                then_branch: Box::new(processed_then),
                else_branch: processed_else,
            })
        },
        
        // Loops - create new scope and set loop flag
        ASTNode::While { condition, body } => {
            *block_counter += 1;
            let current_scope_label = scope_stack.last().unwrap();
            let while_scope_label = format!("{}.while{}", current_scope_label, block_counter);
            
            let while_scope = Scope {
                label: while_scope_label.clone(),
                variables: HashMap::new(),
                functions: HashMap::new(),
                inside_loop: true, // THIS is where the loop magic happens!
            };
            
            all_scopes.push(while_scope);
            scope_stack.push(while_scope_label);
            
            let processed_condition = process_node(*condition, all_scopes, scope_stack, block_counter)?;
            let processed_body = process_node(*body, all_scopes, scope_stack, block_counter)?;
            
            scope_stack.pop();
            
            Ok(ASTNode::While { 
                condition: Box::new(processed_condition), 
                body: Box::new(processed_body) 
            })
        },
        
        // For loops
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
            
            let processed_init = process_node(*init, all_scopes, scope_stack, block_counter)?;
            let processed_condition = process_node(*condition, all_scopes, scope_stack, block_counter)?;
            let processed_increment = process_node(*increment, all_scopes, scope_stack, block_counter)?;
            let processed_body = process_node(*body, all_scopes, scope_stack, block_counter)?;
            
            scope_stack.pop();
            
            Ok(ASTNode::For { 
                init: Box::new(processed_init),
                condition: Box::new(processed_condition),
                increment: Box::new(processed_increment),
                body: Box::new(processed_body) 
            })
        },
        
        // Blocks - create new scope
        ASTNode::Block(statements) => {
            *block_counter += 1;
            let current_scope_label = scope_stack.last().unwrap();
            let block_scope_label = format!("{}.block{}", current_scope_label, block_counter);
            
            // Inherit loop context from parent scope
            let parent_in_loop = all_scopes.iter()
                .find(|s| &s.label == current_scope_label)
                .map(|s| s.inside_loop)
                .unwrap_or(false);
            
            let block_scope = Scope {
                label: block_scope_label.clone(),
                variables: HashMap::new(),
                functions: HashMap::new(),
                inside_loop: parent_in_loop, // Inherit loop context
            };
            
            all_scopes.push(block_scope);
            scope_stack.push(block_scope_label);
            
            let mut processed_statements = Vec::new();
            for stmt in statements {
                let processed = process_node(stmt, all_scopes, scope_stack, block_counter)?;
                processed_statements.push(processed);
            }
            
            scope_stack.pop();
            
            Ok(ASTNode::Block(processed_statements))
        },
        
        // Binary operations - process both sides
        ASTNode::BinaryOp { op, left, right } => {
            let processed_left = process_node(*left, all_scopes, scope_stack, block_counter)?;
            let processed_right = process_node(*right, all_scopes, scope_stack, block_counter)?;
            
            Ok(ASTNode::BinaryOp { 
                op, 
                left: Box::new(processed_left), 
                right: Box::new(processed_right) 
            })
        },
        
        // Unary operations - process expression
        ASTNode::UnaryOp { op, expr } => {
            let processed_expr = process_node(*expr, all_scopes, scope_stack, block_counter)?;
            
            Ok(ASTNode::UnaryOp {
                op,
                expr: Box::new(processed_expr)
            })
        },
        
        // Ternary operations - process all branches
        ASTNode::TernaryOp { condition, true_expr, false_expr } => {
            let processed_condition = process_node(*condition, all_scopes, scope_stack, block_counter)?;
            let processed_true = process_node(*true_expr, all_scopes, scope_stack, block_counter)?;
            let processed_false = process_node(*false_expr, all_scopes, scope_stack, block_counter)?;
            
            Ok(ASTNode::TernaryOp {
                condition: Box::new(processed_condition),
                true_expr: Box::new(processed_true),
                false_expr: Box::new(processed_false),
            })
        },
        
        // PutChar - process expression
        ASTNode::PutChar { expr } => {
            let processed_expr = process_node(*expr, all_scopes, scope_stack, block_counter)?;
            Ok(ASTNode::PutChar { expr: Box::new(processed_expr) })
        },
        
        // Return statements - type checking (TODO: match with function return type)
        ASTNode::Return(expr) => {
            if let Some(return_expr) = expr {
                let processed_expr = process_node(*return_expr, all_scopes, scope_stack, block_counter)?;
                // TODO: Check return type matches function return type
                Ok(ASTNode::Return(Some(Box::new(processed_expr))))
            } else {
                Ok(ASTNode::Return(None))
            }
        },
        
        // Program - process all top-level declarations
        ASTNode::Program(declarations) => {
            let mut processed_declarations = Vec::new();
            for decl in declarations {
                let processed = process_node(decl, all_scopes, scope_stack, block_counter)?;
                processed_declarations.push(processed);
            }
            Ok(ASTNode::Program(processed_declarations))
        },
        
        // Literals and simple nodes - pass through
        node @ (ASTNode::LiteralInt(_) | ASTNode::LiteralChar(_) | ASTNode::LiteralString(_) | ASTNode::Empty) => {
            Ok(node)
        },
    }
}


// Helper functions
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
            Ok(var_info.var_type)
        },
        ASTNode::ArrayAccess { name, .. } => {
            let var_info = lookup_variable(name, all_scopes, scope_stack)
                .ok_or_else(|| SemanticError::UndefinedVariable(name.clone()))?;
            Ok(var_info.var_type) // Return element type
        },
        ASTNode::FunctionCall { name, .. } => {
            let func_info = lookup_function(name, all_scopes, scope_stack)
                .ok_or_else(|| SemanticError::UndefinedFunction(name.clone()))?;
            Ok(func_info.return_type)
        },
        // Add more type inference cases as needed
        _ => Ok(Keywords::Int), // Default fallback
    }
}

fn types_compatible(expected: &Keywords, found: &Keywords) -> bool {
    expected == found || 
    (matches!(expected, Keywords::Int) && matches!(found, Keywords::Char)) ||
    (matches!(expected, Keywords::Char) && matches!(found, Keywords::Int))
}

fn main() {
    println!("🔥 EXTREME C-TO-BRAINFUCK COMPILER TORTURE TEST 🔥");
    println!("Time to separate the wheat from the chaff...\n");

    // Real-world nightmare scenarios
    torture_test("Uninitialized Variable Usage", r#"
        int main() {
            int x;
            int y = x + 5;  // Using uninitialized x!
            return y;
        }
    "#);

    torture_test("Array Out-of-Bounds Access", r#"
        int main() {
            int arr[5];
            int dangerous = arr[10];  // Way out of bounds!
            return dangerous;
        }
    "#);

    torture_test("Complex Type Mismatch Chain", r#"
        int main() {
            int x = 42;
            char c = 'A';
            x = "this is a string";  // Massive type mismatch!
            c = x;
            return c;
        }
    "#);

    torture_test("Invalid Array Size Edge Cases", r#"
        int main() {
            int arr1[0];      // Zero size - should fail
            int arr2[-5];     // Negative size - should fail
            return 0;
        }
    "#);

    torture_test("Deep Nested Scope Madness", r#"
        int main() {
            int x = 1;
            {
                int x = 2;  // Shadow outer x
                {
                    int x = 3;  // Shadow again
                    {
                        int y = x;  // Should use innermost x (3)
                        {
                            int z = w;  // w is undefined!
                        }
                    }
                }
            }
            return x;  // Should be 1
        }
    "#);

    torture_test("Function Call Hell", r#"
        int multiply(int a, int b) {
            return a * b;
        }
        
        int main() {
            int result1 = multiply(5);        // Too few args
            int result2 = multiply(1, 2, 3);  // Too many args
            int result3 = ghost_function(42); // Undefined function
            return result1;
        }
    "#);

    torture_test("Loop Control Flow Chaos", r#"
        int main() {
            int x = 5;
            break;      // Break outside loop - error!
            
            while (x > 0) {
                continue; // This is valid
                x = x - 1;
                if (x == 2) {
                    break; // This is valid
                }
            }
            
            continue;   // Continue outside loop - error!
            return x;
        }
    "#);

    torture_test("Mixed Array and Function Madness", r#"
        int process(int value) {
            return value * 2;
        }
        
        int main() {
            int numbers[3];
            char letter = 'x';
            
            numbers[letter] = 42;        // Char as index (should work)
            numbers["hello"] = 10;       // String as index - error!
            
            int result = process(numbers[2]);
            return result;
        }
    "#);

    torture_test("Return Type Validation", r#"
        char get_letter() {
            return 65;  // Should work (int -> char)
        }
        
        int get_number() {
            return "not a number";  // Type mismatch error!
        }
        
        int main() {
            char c = get_letter();
            int n = get_number();
            return n;
        }
    "#);

    torture_test("Recursive Declaration Nightmare", r#"
        int factorial(int n) {
            int factorial = 5;  // Variable same name as function
            if (n <= 1) {
                return factorial;
            }
            return n * factorial(n - 1);
        }
        
        int main() {
            return factorial(5);
        }
    "#);

    torture_test("Complex Expression Folding Test", r#"
        int main() {
            int x = 2 + 3 * 4 - 1;           // Should fold to 13
            int y = (5 > 3) ? (10 + 5) : 0;  // Should fold to 15
            int z = !0 && (4 < 10);          // Should fold to 1
            
            int arr[x];  // Using folded constant as array size
            return x + y + z;  // Should be 29
        }
    "#);

    torture_test("Extreme Nesting with All Features", r#"
        int helper(int a, int b) {
            return a + b;
        }
        
        int main() {
            int global_var = 100;
            
            for (int i = 0; i < 3; i = i + 1) {
                int loop_var = i * 2;
                
                if (loop_var > 2) {
                    int if_var = loop_var + 1;
                    
                    while (if_var > 0) {
                        if_var = if_var - 1;
                        
                        if (if_var == 1) {
                            int deep_var = helper(if_var, global_var);
                            break;
                        }
                    }
                } else {
                    continue;
                }
            }
            
            return global_var;
        }
    "#);

    torture_test("Assignment Chain Type Checking", r#"
        int main() {
            int a, b, c;
            char x, y;
            
            a = b = c = 42;        // Should work
            x = y = 'Z';           // Should work
            a = x = c;             // Mixed types - should work with conversion
            x = "string literal";  // Type mismatch - should fail
            
            return a;
        }
    "#);

    torture_test("Edge Case Array Initializers", r#"
        int main() {
            int arr1[3] = {1, 2, 3};           // Valid
            int arr2[5] = {1, 2};              // Partial init - valid
            char str[10] = "hello";            // String init - valid
            int arr3[2] = {1, 2, 3, 4, 5};     // Too many elements - error
            
            return arr1[0];
        }
    "#);

    torture_test("Pathological Identifier Shadowing", r#"
        int x = 42;  // Global
        
        int main() {
            int x = 1;   // Shadow global
            {
                char x = 'A';  // Shadow local
                {
                    int x = 999;  // Shadow again
                    return x;     // Should return 999
                }
                return x;  // Should return 'A' (65)
            }
            return x;  // Should return 1
        }
    "#);

    println!("\n💀 TORTURE TEST COMPLETE! 💀");
    println!("If your compiler survived this gauntlet, it's ready for the real world!");
    println!("Time to generate some EPIC brainfuck code! 🧠🔥");
}

fn torture_test(name: &str, source_code: &str) {
    println!("💀 TORTURE TEST: {}", name);
    println!("💀 Code under torture:");
    println!("{}", source_code);
    println!("💀 Result:");

    // Parse and analyze
    let ast = match parse_program(source_code) {
        Some(ast) => ast,
        None => {
            println!(" PARSER DEATH: Failed to parse");
            return;
        }
    };

    match run_ast(&mut vec![ast]) {
        Ok(_) => {
            println!("   ✅ SURVIVED: Code passed semantic analysis");
            println!("   🎯 Ready for brainfuck generation!");
        },
        Err(e) => {
            println!("   💥 SEMANTIC DEATH: {}", e);
            println!("   😈 Your compiler caught the bug like a boss!");
        }
    }
    
}
