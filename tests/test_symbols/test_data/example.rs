use std::collections::HashMap;

pub struct Config {
    pub name: String,
    pub timeout: u64,
}

pub trait Handler {
    fn handle(&self, event: &str);
}

pub enum Status {
    Active,
    Inactive,
}

impl Handler for Config {
    fn handle(&self, event: &str) {
        println!("{}: {}", self.name, event);
    }
}

pub fn compute(a: i32, b: i32) -> i32 {
    a + b
}
