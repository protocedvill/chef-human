import { readFile } from "fs";
import { join } from "path";

function greet(name) {
    return `Hello, ${name}!`;
}

class Calculator {
    constructor(factor) {
        this.factor = factor;
    }

    multiply(x) {
        return x * this.factor;
    }
}
