// Sample JavaScript code for testing

function greet(name) {
    return `Hello, ${name}!`;
}

function calculateSum(a, b) {
    return a + b;
}

function calculateProduct(a, b) {
    return a * b;
}

class Calculator {
    constructor() {
        this.history = [];
    }

    add(x, y) {
        const result = x + y;
        this.history.push(`${x} + ${y} = ${result}`);
        return result;
    }

    multiply(x, y) {
        const result = x * y;
        this.history.push(`${x} * ${y} = ${result}`);
        return result;
    }

    getHistory() {
        return [...this.history];
    }
}

module.exports = {
    greet,
    calculateSum,
    calculateProduct,
    Calculator
};
