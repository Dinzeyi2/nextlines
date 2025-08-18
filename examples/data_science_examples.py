"""Example script demonstrating natural language to Python execution."""

from codefull import ExecutionTemplate, ParameterType, NaturalLanguageExecutor


def build_executor():
    nl = NaturalLanguageExecutor()
    nl.templates = [
        ExecutionTemplate(
            "set {var} to {value}",
            "execute_assignment",
            parameters={"var": ParameterType.IDENTIFIER, "value": ParameterType.VALUE},
        ),
        ExecutionTemplate(
            "create a list named {var} with {value}",
            "execute_list_creation",
            parameters={"var": ParameterType.IDENTIFIER, "value": ParameterType.COLLECTION},
        ),
        ExecutionTemplate(
            "add {value} to {var}",
            "execute_add_to_var",
            parameters={"var": ParameterType.IDENTIFIER, "value": ParameterType.VALUE},
        ),
        ExecutionTemplate(
            "print {value}",
            "execute_print",
            parameters={"value": ParameterType.IDENTIFIER},
        ),
    ]
    return nl


def main():
    nl = build_executor()
    print(nl.execute("set count to 10"))
    print(nl.execute("add 5 to count"))
    print(nl.execute("print count"))
    print(nl.execute("create a list named data with 1,2,3"))
    print(nl.execute("print data"))


if __name__ == "__main__":
    main()
