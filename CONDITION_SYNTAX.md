# Condition Syntax

Natural language conditions supported by the `NaturalLanguageExecutor` can use
logical operators, grouping and a number of comparison synonyms. The parser
accepts:

## Logical operators

- `and`, `or` (case insensitive)
- Parentheses `(` and `)` for nested expressions

## Comparisons

| Phrase(s)                               | Python operator |
|-----------------------------------------|-----------------|
| `equals`, `is equal to`                 | `==`            |
| `is not equal to`                       | `!=`            |
| `is greater than`                       | `>`             |
| `is less than`                          | `<`             |
| `is greater than or equal to`           | `>=`            |
| `is less than or equal to`              | `<=`            |
| `contains`, `is in`                     | `in`            |
| `is not in`, `not in`                   | `not in`        |

The resulting expression is validated with `ast.parse` before execution.  A
`ValueError` is raised if the condition cannot be parsed.
