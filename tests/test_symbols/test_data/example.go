package main

import "fmt"

type Config struct {
    Name    string
    Timeout int
}

type Handler interface {
    Handle(event string)
}

func (c *Config) Handle(event string) {
    fmt.Println(c.Name, event)
}

func compute(a int, b int) int {
    return a + b
}
