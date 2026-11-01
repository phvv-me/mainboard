# The node types the deletions below name.
STATEMENT = "expression_statement"
FUNCTION = "function_definition"
DECORATED = "decorated_definition"

# A C function whose `if` holds an inert statement and another one in braces.
BRACED_IF = """int f(int x) {
    if (x) {
        x == 1;
        x = 2;
    }
    return x;
}
"""
# A CUDA kernel whose `for` has an inert statement as its whole body, with no braces.
UNBRACED_LOOP = """__global__ void k(int *p, int n) {
    for (int i = 0; i < n; i++)
        p[0] == 1;
    p[1] = 2;
}
"""
