"""Rules over descriptor and attribute use in classes (mcmr's PY-CLAS descriptors family)."""

from ...catalog import Example, Rule


class DirectMethodDescriptorCallCount(Rule, id="PY-CLAS0008"):
    """`staticmethod` and `classmethod` calls used without decorator syntax."""

    query = """
            SELECT path, start_line, start_column, qualified_name FROM core_call
            WHERE qualified_name IN ('builtins.staticmethod', 'builtins.classmethod')
              AND top_type IS DISTINCT FROM 'decorator'
        """
    message = (
        "`{qualified_name}` constructs a descriptor directly instead of using decorator syntax"
    )


class DuplicateComponentAttributeAliasCount(Rule, id="PY-CLAS0009"):
    """Classes copying attributes from a component they already retain."""

    query = "SELECT * FROM class_classes WHERE duplicate_component_alias_count > 0"
    message = (
        "`{name}` copies {duplicate_component_alias_count} attributes from a component it "
        "already retains"
    )


class DynamicSuperReceiver(Rule, id="PY-CLAS0010", applicability="safe"):
    """`super` calls in methods whose first argument is computed from the receiver."""

    query = """
            SELECT s.path, s.start_line, s.start_column, f.qualname, r.name AS receiver,
                   s.fact_id, s.ordinal AS call_ordinal, f.start_line AS callable_line
            FROM syntax_nodes s
            JOIN syntax_nodes m ON m.node_id = s.parent_id AND m.kind = 'member'
            JOIN syntax_children c
              ON c.fact_id = s.fact_id AND c.parent_ordinal = s.ordinal AND c.child_order = 1
            JOIN syntax_nodes r
              ON r.fact_id = c.fact_id AND r.ordinal = c.child_ordinal
             AND r.name IN ('type', '__class__')
            JOIN syntax_facts f ON f.fact_id = s.fact_id
            WHERE s.kind = 'call' AND s.name = 'super'
              AND f.kind = 'callable' AND contains(f.qualname, '.')
        """
    message = "`{qualname}` computes the first `super` argument from `{receiver}`"
    # `super(type(self), self)` and `super(self.__class__, self)` recurse forever once a subclass
    # inherits the method; `super()` names the defining class. Only where the zero-argument form
    # works: the call sits directly in a method (no lambda or comprehension between) whose first
    # parameter is the `self` the call passes.
    fix = """
        WITH arguments AS (
          SELECT call_id, count(*) AS argument_count,
                 any_value(text) FILTER (WHERE root_ordinal = 0) AS first_text,
                 any_value(text) FILTER (WHERE root_ordinal = 1) AS second_text
          FROM call_expressions WHERE root_relation = 'argument' AND depth = 0
          GROUP BY call_id
        )
        SELECT f.finding_key, c.node_path AS path, c.node_start_line AS start_line,
               c.node_start_column AS start_column, c.node_end_line AS end_line,
               c.node_end_column AS end_column, 'super()' AS text
        FROM finding f
        JOIN call_calls c ON c.node_path = f.path AND c.node_start_line = f.start_line
          AND c.node_start_column = f.start_column AND c.qualified_name = 'builtins.super'
        JOIN arguments a USING (call_id)
        JOIN function_functions m ON m.path = f.path AND m.start_line = f.callable_line
          AND m.scope = 'method'
        JOIN function_parameters p ON p.function_id = m.entity_id AND p.ordinal = 0
        WHERE a.argument_count = 2 AND p.name = a.second_text
          AND a.first_text IN ('type(' || a.second_text || ')', a.second_text || '.__class__')
          AND NOT EXISTS (
            SELECT 1 FROM syntax_nodes inner_scope
            WHERE inner_scope.fact_id = f.fact_id AND inner_scope.ordinal > 0
              AND inner_scope.kind IN ('callable', 'comprehension')
              AND inner_scope.ordinal < f.call_ordinal
              AND f.call_ordinal < inner_scope.subtree_end)
    """
    examples = (
        Example(
            bad=(
                "class Base:\n    def run(self):\n        pass\n\n\n"
                "class Child(Base):\n    def run(self):\n        super(type(self), self).run()\n"
            ),
            good=(
                "class Base:\n    def run(self):\n        pass\n\n\n"
                "class Child(Base):\n    def run(self):\n        super().run()\n"
            ),
        ),
    )
