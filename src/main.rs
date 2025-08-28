
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
    /* 
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
    Continue,*/
    BinaryOp { op: Operations, left: Box<ASTNode>, right: Box<ASTNode> },
    UnaryOp { op: Operations, expr: Box<ASTNode> },
    TernaryOp { condition: Box<ASTNode>, true_expr: Box<ASTNode>, false_expr: Box<ASTNode> },
    FunctionCall { name: String, args: Vec<ASTNode> },
    LiteralInt(i64),
    LiteralChar(char),
    LiteralString(String),
    Identifier(String),
    ArrayAccess { name: String, index: Box<ASTNode> },
    //Block(Vec<ASTNode>),
    //Empty,
}


fn parse_literal(tokens: &mut Vec<Token>)->Option<Vec<ASTNode>>{
    if tokens.is_empty() {
        return None;
    }
    else{
        let token = tokens.remove(0);
        match token {
            Token::Number(n) => Some(vec![ASTNode::LiteralInt(n)]),
            Token::CharLiteral(c) => Some(vec![ASTNode::LiteralChar(c)]),
            Token::StringLiteral(s) => Some(vec![ASTNode::LiteralString(s)]),
            _ => None,
        }
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


fn parse_identifier(tokens: &mut Vec<Token>) -> Option<Vec<ASTNode>> {
    if let Some(Token::Identifier(name)) = tokens.first() {
        let name = name.clone(); // clone so we can consume
        tokens.remove(0);         // consume the token
        Some(vec![ASTNode::Identifier(name)])
    } else {
        None
    }
}
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

fn parse_ternary_operation(tokens: &mut Vec<Token>, condition: ASTNode) -> Option<Vec<ASTNode>> {
    if let Some(Token::Operator(Operations::Ternary)) = tokens.first() {
        tokens.remove(0); // consume '?'
        let true_expr = parse_expression(tokens)?;
        if let Some(Token::Punctuator(Punctuators::Colon)) = tokens.first() {
            tokens.remove(0); // consume ':'
            let false_expr = parse_expression(tokens)?;
            Some(vec![ASTNode::TernaryOp {
                condition: Box::new(condition),
                true_expr: Box::new(true_expr[0].clone()),
                false_expr: Box::new(false_expr[0].clone())
            }])
        } else {
            None // Error: Expected ':'
        }
    } else {
        Some(vec![condition]) // Not a ternary operation, return original condition
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
        // Handle parentheses first
        Token::Punctuator(Punctuators::LeftParen) => {
            tokens.remove(0); // consume '('
            let expr = parse_expression(tokens)?;
            if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightParen))) {
                tokens.remove(0); // consume ')'
                expr[0].clone()
            } else {
                return None; // missing ')'
            }
        },
        // Handle unary operators
        Token::Operator(op) if matches!(op, Operations::Subtract | Operations::Not | Operations::BitwiseNot) => {
            let op = if let Token::Operator(o) = tokens.remove(0) { o } else { unreachable!() };
            let expr = parse_primary(tokens)?; // Use parse_primary instead of parse_expression
            ASTNode::UnaryOp { op, expr: Box::new(expr) }
        },
        // Handle literals
        Token::Number(_) | Token::CharLiteral(_) | Token::StringLiteral(_) => {
            parse_literal(tokens)?.pop()?
        },
        // Handle identifiers
        Token::Identifier(_) => {
            parse_identifier(tokens)?.pop()?
        },
        _ => return None,
    };

    // Handle postfix operations (array access, function calls)
    loop {
        match tokens.first() {
            Some(Token::Punctuator(Punctuators::LeftBracket)) => {
                tokens.remove(0); // consume '['
                let index = parse_expression(tokens)?;
                if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightBracket))) {
                    tokens.remove(0); // consume ']'
                    if let ASTNode::Identifier(name) = node {
                        node = ASTNode::ArrayAccess { name, index: Box::new(index[0].clone()) };
                    } else {
                        return None;
                    }
                } else { 
                    return None; 
                }
            },
            Some(Token::Punctuator(Punctuators::LeftParen)) => {
                tokens.remove(0); // consume '('
                let mut args = Vec::new();
                
                if !matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightParen))) {
                    loop {
                        let expr = parse_expression(tokens)?;
                        args.push(expr[0].clone());
                        
                        if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightParen))) {
                            break;
                        } else if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::Comma))) {
                            tokens.remove(0); // consume ','
                        } else {
                            return None; // expected ',' or ')'
                        }
                    }
                }
                
                if matches!(tokens.first(), Some(Token::Punctuator(Punctuators::RightParen))) {
                    tokens.remove(0); // consume ')'
                    if let ASTNode::Identifier(name) = node {
                        node = ASTNode::FunctionCall { name, args };
                    } else {
                        return None;
                    }
                } else { 
                    return None; 
                }
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
            right = tern[0].clone();
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

fn parse_expression(tokens: &mut Vec<Token>) -> Option<Vec<ASTNode>> {
    let mut node = parse_binary_ops(tokens, 1)?;
    if let Some(Token::Operator(Operations::Ternary)) = tokens.first() {
        let tern = parse_ternary_operation(tokens, node)?;
        node = tern[0].clone();
    }
    if is_foldable(&node) {
        // If foldable, evaluate it to a literal right away
        if let Some(folded_node) = eval_tree(node.clone()) {
            println!("Folded expression to: {:?}", folded_node);
            node = folded_node;
        }
    }
    Some(vec![node])
}

fn main() {
    let examples = vec![
        "1 + 2 * 3",           // Should fold to: LiteralInt(7)
        "10 - 4 / 2",          // Should fold to: LiteralInt(8)  
        "5 > 3 ? 100 : 200",   // Should fold to: LiteralInt(100)
        "-42 + 2",             // Should fold to: LiteralInt(-40)
        "a + 2 * 3",           // Won't fold (contains identifier)//Being conservative here
    ];
    
    for source in examples {
        println!("Source: {}", source);
        let mut tokens = tokenize(source);
        match parse_expression(&mut tokens) {
            Some(ast) => println!("Final AST: {:#?}", ast),
            None => println!("Parsing failed"),
        }
        println!("-----------------------");
    }
}

